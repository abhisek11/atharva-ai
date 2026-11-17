"""Aggregation daemon. Run alongside the API: `python coordinator/aggregator.py`

Async, windowed aggregation:
  - every AGGREGATE_INTERVAL_S, drain whatever deltas arrived
  - weighted-average by steps_completed
  - apply with an outer optimizer (plain averaging here; swap in Nesterov
    momentum on the averaged delta for the actual DiLoCo recipe)
  - write model_v{N+1}.pt, bump model:version

Never blocks on stragglers: a slow worker's delta just lands in the next window.
"""
import json
import os
import sys
import time

import redis
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from worker.compression import decompress_state_delta
from worker.model import build_model  # single source of truth for architecture

AGGREGATE_INTERVAL_S = int(os.getenv("AGGREGATE_INTERVAL_S", "60"))
MODEL_DIR = os.getenv("MODEL_DIR", "/data/models")
OUTER_LR = float(os.getenv("OUTER_LR", "0.7"))  # DiLoCo outer step size

r = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), decode_responses=True)


def load_current():
    version = int(r.get("model:version") or 0)
    path = os.path.join(MODEL_DIR, f"model_v{version}.pt")
    model = build_model()
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location="cpu"))
    return model, version


def aggregate_once():
    n = r.llen("round:deltas")
    if n == 0:
        return

    model, version = load_current()
    params = dict(model.named_parameters())

    accum = {name: torch.zeros(p.numel()) for name, p in params.items()}
    total_steps = 0

    for _ in range(n):
        raw = r.lpop("round:deltas")
        if raw is None:
            break
        up = json.loads(raw)
        weight = up["steps_completed"]
        sparse = decompress_state_delta(up["payload_b64"])
        for name, flat in sparse.items():
            if name in accum:
                accum[name] += flat * weight
        total_steps += weight

    if total_steps == 0:
        return

    with torch.no_grad():
        for name, p in params.items():
            avg_delta = (accum[name] / total_steps).view_as(p)
            p.add_(OUTER_LR * avg_delta)

    new_version = version + 1
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(MODEL_DIR, f"model_v{new_version}.pt"))
    r.set("model:version", new_version)
    print(f"[aggregator] v{version} -> v{new_version} "
          f"({n} deltas, {total_steps} worker-steps)")


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(MODEL_DIR, "model_v0.pt")):
        torch.save(build_model().state_dict(),
                   os.path.join(MODEL_DIR, "model_v0.pt"))
        r.set("model:version", 0)
        print("[aggregator] initialized model_v0")
    while True:
        try:
            aggregate_once()
        except Exception as e:
            print(f"[aggregator] error: {e}")
        time.sleep(AGGREGATE_INTERVAL_S)
