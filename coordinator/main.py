"""atharva.ai coordinator.

Redis layout:
  workers:{id}            hash  {gpu, last_seen, token}
  jobs:pending            list  of job_id (FIFO)
  jobs:leased             zset  job_id -> lease_expiry_ts
  job:{id}                hash  {shard_url, model_version, ...}
  model:version           int
  round:deltas            list  of accepted GradientUpload JSON (drained by aggregator)

Model weights themselves live on disk (MODEL_DIR) and should be fronted by
nginx/CDN in production — never stream multi-GB files through FastAPI workers.
"""
import json
import os
import secrets
import time
import uuid

import redis
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.version import VERSION
from coordinator.license import LicenseError, check_gpu_count, load_and_verify
from shared.schemas import (
    AckResponse, GradientUpload, RegisterRequest, RegisterResponse, WorkItem,
)

LEASE_SECONDS = int(os.getenv("LEASE_SECONDS", "600"))
LOCAL_STEPS = int(os.getenv("LOCAL_STEPS", "200"))   # DiLoCo inner steps
MODEL_DIR = os.getenv("MODEL_DIR", "/data/models")
STALE_VERSION_TOLERANCE = 1  # accept deltas computed against version >= current-1

r = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), decode_responses=True)
app = FastAPI(title="atharva.ai coordinator", version=VERSION)

_license = None


@app.on_event("startup")
def _validate_license():
    global _license
    if os.getenv("ATHARVA_DEV_MODE") == "1":
        _license = {"customer": "dev", "max_gpus": 9999,
                    "expires": "9999-01-01", "plan": "dev", "_grace": False}
        return
    _license = load_and_verify()   # raises -> coordinator refuses to start
    if _license["_grace"]:
        print(f"[license] WARNING: expired {_license['expires']}, in grace period")


def _auth(worker_id: str, token: str):
    stored = r.hget(f"workers:{worker_id}", "token")
    if stored is None or stored != token:
        raise HTTPException(401, "bad worker credentials")


def _reclaim_expired_leases():
    now = time.time()
    expired = r.zrangebyscore("jobs:leased", 0, now)
    for job_id in expired:
        r.zrem("jobs:leased", job_id)
        r.rpush("jobs:pending", job_id)   # back of the queue, idempotent job spec


@app.post("/register", response_model=RegisterResponse)
def register(req: RegisterRequest):
    active = len(r.keys("workers:*"))
    try:
        check_gpu_count(_license, active + 1)
    except LicenseError as e:
        raise HTTPException(403, str(e))
    worker_id = str(uuid.uuid4())
    token = secrets.token_urlsafe(32)
    r.hset(f"workers:{worker_id}", mapping={
        "token": token, "gpu": req.gpu_name, "vram": req.vram_gb,
        "hostname": req.hostname, "last_seen": time.time(),
    })
    return RegisterResponse(worker_id=worker_id, token=token)


@app.post("/heartbeat/{worker_id}")
def heartbeat(worker_id: str, x_token: str = Header()):
    _auth(worker_id, x_token)
    r.hset(f"workers:{worker_id}", "last_seen", time.time())
    return {"ok": True}


@app.get("/work/{worker_id}", response_model=WorkItem)
def get_work(worker_id: str, x_token: str = Header()):
    _auth(worker_id, x_token)
    _reclaim_expired_leases()

    job_id = r.lpop("jobs:pending")
    if job_id is None:
        raise HTTPException(404, "no work available, retry with backoff")

    lease_expiry = time.time() + LEASE_SECONDS
    r.zadd("jobs:leased", {job_id: lease_expiry})
    job = r.hgetall(f"job:{job_id}")
    return WorkItem(
        job_id=job_id,
        model_version=int(r.get("model:version") or 0),
        shard_url=job["shard_url"],
        local_steps=LOCAL_STEPS,
        lr=float(job.get("lr", 3e-4)),
        batch_size=int(job.get("batch_size", 8)),
        lease_expires_at=lease_expiry,
    )


@app.post("/gradients", response_model=AckResponse)
def submit_gradients(up: GradientUpload, x_token: str = Header()):
    _auth(up.worker_id, x_token)
    current = int(r.get("model:version") or 0)

    # Reject deltas computed against weights that are too stale to average safely.
    if up.model_version < current - STALE_VERSION_TOLERANCE:
        r.zrem("jobs:leased", up.job_id)
        r.rpush("jobs:pending", up.job_id)
        return AckResponse(accepted=False, reason="stale model version",
                           latest_model_version=current)

    r.rpush("round:deltas", up.model_dump_json())
    r.zrem("jobs:leased", up.job_id)          # job complete, release lease
    r.hset(f"workers:{up.worker_id}", "last_seen", time.time())
    return AckResponse(accepted=True, latest_model_version=current)


@app.get("/model/{version}")
def get_model(version: int):
    path = os.path.join(MODEL_DIR, f"model_v{version}.pt")
    if not os.path.exists(path):
        raise HTTPException(404, "unknown version")
    return FileResponse(path)  # front with nginx sendfile / CDN in prod


@app.get("/")
def landing_page():
    index_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"service": "atharva.ai coordinator", "version": VERSION}


@app.get("/status")
def status():
    _reclaim_expired_leases()
    return {
        "atharva_version": VERSION,
        "license": {k: _license[k] for k in ("customer", "plan", "expires", "_grace")},
        "model_version": int(r.get("model:version") or 0),
        "pending_jobs": r.llen("jobs:pending"),
        "leased_jobs": r.zcard("jobs:leased"),
        "unaggregated_deltas": r.llen("round:deltas"),
    }
