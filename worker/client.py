"""atharva.ai worker client.

Usage:
    COORDINATOR=https://atharva.ai python worker/client.py

Loop:
  register -> [pull model if stale -> get work -> train local_steps ->
               compress (params_after - params_before) -> upload] -> repeat

Error feedback: coordinates dropped by top-k are accumulated in a residual
buffer and added back before the next compression pass, so nothing is lost
permanently — only delayed a round or two.
"""
import io
import os
import platform
import sys
import threading
import time

import requests
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from worker.compression import compress_state_delta
from worker.model import build_model

COORDINATOR = os.getenv("COORDINATOR", "http://localhost:8000")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKOFF_MAX_S = 120


class Worker:
    def __init__(self):
        self.model = build_model().to(DEVICE)
        self.model_version = -1
        self.residual = {}  # error-feedback buffer, name -> flat tensor
        self._register()
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()

    def _register(self):
        gpu = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"
        vram = (torch.cuda.get_device_properties(0).total_memory / 1e9
                if DEVICE == "cuda" else 0.0)
        resp = requests.post(f"{COORDINATOR}/register", json={
            "hostname": platform.node(), "gpu_name": gpu,
            "vram_gb": round(vram, 1), "torch_version": torch.__version__,
        }, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        self.worker_id, self.token = data["worker_id"], data["token"]
        self.hb_interval = data["heartbeat_interval_s"]
        print(f"[worker] registered as {self.worker_id} on {gpu}")

    def _headers(self):
        return {"x-token": self.token}

    def _heartbeat_loop(self):
        while True:
            try:
                requests.post(f"{COORDINATOR}/heartbeat/{self.worker_id}",
                              headers=self._headers(), timeout=10)
            except requests.RequestException:
                pass
            time.sleep(self.hb_interval)

    def _sync_model(self, target_version: int):
        if target_version == self.model_version:
            return
        resp = requests.get(f"{COORDINATOR}/model/{target_version}", timeout=300)
        resp.raise_for_status()
        state = torch.load(io.BytesIO(resp.content), map_location=DEVICE)
        self.model.load_state_dict(state)
        self.model_version = target_version
        print(f"[worker] synced to model v{target_version}")

    def _fetch_shard(self, shard_url: str) -> torch.Tensor:
        """Return a LongTensor [num_seqs, seq_len] of token ids.
        Skeleton: random tokens. Replace with real tokenized shards
        (e.g. .npy files on R2/S3) — the rest of the loop is unchanged."""
        return torch.randint(0, 50257, (256, 128))

    def _train(self, work: dict) -> tuple[str, float]:
        data = self._fetch_shard(work["shard_url"])
        before = {n: p.detach().clone()
                  for n, p in self.model.named_parameters()}

        opt = torch.optim.AdamW(self.model.parameters(), lr=work["lr"])
        self.model.train()
        losses, bs = [], work["batch_size"]
        for step in range(work["local_steps"]):
            i = (step * bs) % max(1, len(data) - bs)
            batch = data[i:i + bs].to(DEVICE)
            logits = self.model(batch[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch[:, 1:].reshape(-1),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        # delta = local - global, plus error-feedback residual
        delta = {}
        for n, p in self.model.named_parameters():
            d = (p.detach() - before[n]).flatten().cpu()
            if n in self.residual:
                d = d + self.residual[n]
            delta[n] = d
        payload = compress_state_delta(
            {n: d for n, d in delta.items()}
        )
        # residual = what top-k dropped (recompute cheaply via decompress round-trip
        # is wasteful; track it directly)
        from worker.compression import decompress_state_delta
        kept = decompress_state_delta(payload)
        self.residual = {n: delta[n] - kept[n] for n in delta}

        # reset local params to global — the coordinator owns the trajectory
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                p.copy_(before[n])

        return payload, sum(losses) / len(losses)

    def run(self):
        backoff = 5
        while True:
            try:
                resp = requests.get(f"{COORDINATOR}/work/{self.worker_id}",
                                    headers=self._headers(), timeout=30)
                if resp.status_code == 404:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, BACKOFF_MAX_S)
                    continue
                resp.raise_for_status()
                backoff = 5
                work = resp.json()

                self._sync_model(work["model_version"])
                payload, avg_loss = self._train(work)

                ack = requests.post(f"{COORDINATOR}/gradients",
                                    headers=self._headers(), json={
                    "job_id": work["job_id"],
                    "worker_id": self.worker_id,
                    "model_version": self.model_version,
                    "steps_completed": work["local_steps"],
                    "loss": avg_loss,
                    "payload_b64": payload,
                }, timeout=120).json()
                print(f"[worker] job {work['job_id'][:8]} loss={avg_loss:.4f} "
                      f"accepted={ack['accepted']}")
            except requests.RequestException as e:
                print(f"[worker] network error: {e}, retrying in {backoff}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, BACKOFF_MAX_S)


if __name__ == "__main__":
    Worker().run()
