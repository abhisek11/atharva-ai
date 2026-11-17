"""atharva.ai coordinator with LAN Auto-Scan & Capacity-Ranked Scheduling.

Redis layout:
  workers:{id}            hash  {gpu, last_seen, token, ip_address, device_type,
                                 vram_total_gb, vram_free_gb, ram_total_gb, ram_free_gb,
                                 cpu_cores, cpu_load_pct, gpu_load_pct, is_free, capacity_score}
  jobs:pending            list  of job_id (FIFO)
  jobs:leased             zset  job_id -> lease_expiry_ts
  job:{id}                hash  {shard_url, model_version, ...}
  model:version           int
  round:deltas            list  of accepted GradientUpload JSON (drained by aggregator)

Model weights themselves live on disk (MODEL_DIR) and should be fronted by
nginx/CDN in production — never stream multi-GB files through FastAPI workers.
"""
import os
import secrets
import time
import uuid
from typing import Dict, List, Optional

import redis
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.version import VERSION
from coordinator.license import LicenseError, check_gpu_count, load_and_verify
from coordinator.scanner import get_local_subnet, rank_devices, scan_lan_devices
from shared.schemas import (
    AckResponse, GradientUpload, HeartbeatRequest, NetworkScanResponse,
    RankedNode, RegisterRequest, RegisterResponse, WorkItem, compute_capacity_score,
)

LEASE_SECONDS = int(os.getenv("LEASE_SECONDS", "600"))
LOCAL_STEPS = int(os.getenv("LOCAL_STEPS", "200"))   # DiLoCo inner steps
MODEL_DIR = os.getenv("MODEL_DIR", "/data/models")
STALE_VERSION_TOLERANCE = 1  # accept deltas computed against version >= current-1
WORKER_STALE_S = 120         # workers silent >120s are pruned from active rankings

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
        if r.zrem("jobs:leased", job_id):
            r.rpush("jobs:pending", job_id)   # atomic guard so only 1 worker re-queues


def _collect_active_workers() -> List[Dict]:
    """Gather all active registered workers from Redis."""
    now = time.time()
    nodes: List[Dict] = []
    for key in r.keys("workers:*"):
        w = r.hgetall(key)
        if not w:
            continue
        last_seen = float(w.get("last_seen", 0))
        if now - last_seen > WORKER_STALE_S:
            continue
        worker_id = key.split("workers:", 1)[1]
        nodes.append({
            "worker_id": worker_id,
            "hostname": w.get("hostname", worker_id),
            "ip_address": w.get("ip_address", "127.0.0.1"),
            "device_type": w.get("device_type", "cpu"),
            "gpu_name": w.get("gpu", "cpu"),
            "vram_total_gb": float(w.get("vram_total_gb", w.get("vram", 0.0))),
            "vram_free_gb": float(w.get("vram_free_gb", w.get("vram", 0.0))),
            "ram_total_gb": float(w.get("ram_total_gb", 8.0)),
            "ram_free_gb": float(w.get("ram_free_gb", 4.0)),
            "cpu_cores": int(w.get("cpu_cores", 4)),
            "cpu_load_pct": float(w.get("cpu_load_pct", 0.0)),
            "gpu_load_pct": float(w.get("gpu_load_pct", 0.0)),
            "is_free": w.get("is_free", "True") == "True",
        })
    return nodes


def _get_ranked_roster(max_selected: Optional[int] = None) -> List[RankedNode]:
    limit = max_selected or (_license["max_gpus"] if _license else None)
    return rank_devices(_collect_active_workers(), max_selected=limit)


@app.post("/register", response_model=RegisterResponse)
def register(req: RegisterRequest):
    active_nodes = _collect_active_workers()
    try:
        check_gpu_count(_license, len(active_nodes) + 1)
    except LicenseError as e:
        raise HTTPException(403, str(e))

    worker_id = str(uuid.uuid4())
    token = secrets.token_urlsafe(32)
    vram_free = req.vram_free_gb if req.vram_free_gb > 0 else req.vram_gb
    ram_total = req.ram_total_gb if req.ram_total_gb > 0 else 8.0
    ram_free = req.ram_free_gb if req.ram_free_gb > 0 else ram_total * 0.6

    score = compute_capacity_score(
        device_type=req.device_type,
        vram_free_gb=vram_free,
        ram_free_gb=ram_free,
        cpu_cores=req.cpu_cores,
        cpu_load_pct=req.cpu_load_pct,
        gpu_load_pct=req.gpu_load_pct,
        is_free=req.is_free,
    )

    r.hset(f"workers:{worker_id}", mapping={
        "token": token,
        "hostname": req.hostname,
        "ip_address": req.ip_address,
        "device_type": req.device_type,
        "gpu": req.gpu_name,
        "vram": req.vram_gb,
        "vram_total_gb": req.vram_gb,
        "vram_free_gb": vram_free,
        "ram_total_gb": ram_total,
        "ram_free_gb": ram_free,
        "cpu_cores": req.cpu_cores,
        "cpu_load_pct": req.cpu_load_pct,
        "gpu_load_pct": req.gpu_load_pct,
        "is_free": str(req.is_free),
        "capacity_score": score,
        "last_seen": time.time(),
    })

    roster = _get_ranked_roster()
    my_rank = next((n.rank for n in roster if n.worker_id == worker_id), 1)
    return RegisterResponse(
        worker_id=worker_id,
        token=token,
        rank=my_rank,
        capacity_score=score,
    )


@app.post("/heartbeat/{worker_id}")
def heartbeat(
    worker_id: str,
    hb: Optional[HeartbeatRequest] = None,
    x_token: str = Header(),
):
    _auth(worker_id, x_token)
    updates = {"last_seen": time.time()}
    if hb is not None:
        w = r.hgetall(f"workers:{worker_id}")
        score = compute_capacity_score(
            device_type=w.get("device_type", "cpu"),
            vram_free_gb=hb.vram_free_gb,
            ram_free_gb=hb.ram_free_gb,
            cpu_cores=int(w.get("cpu_cores", 4)),
            cpu_load_pct=hb.cpu_load_pct,
            gpu_load_pct=hb.gpu_load_pct,
            is_free=hb.is_free,
        )
        updates.update({
            "vram_free_gb": hb.vram_free_gb,
            "ram_free_gb": hb.ram_free_gb,
            "cpu_load_pct": hb.cpu_load_pct,
            "gpu_load_pct": hb.gpu_load_pct,
            "is_free": str(hb.is_free),
            "capacity_score": score,
        })
    r.hset(f"workers:{worker_id}", mapping=updates)
    return {"ok": True}


@app.get("/nodes/ranked", response_model=NetworkScanResponse)
def get_ranked_nodes(top_k: Optional[int] = Query(default=None)):
    """Return all registered devices ranked from highest free capacity (#1) to lowest."""
    ranked = _get_ranked_roster(max_selected=top_k)
    free_count = sum(1 for n in ranked if n.is_free)
    return NetworkScanResponse(
        subnet=get_local_subnet(),
        total_discovered=len(ranked),
        free_devices=free_count,
        ranked_nodes=ranked,
    )


@app.get("/nodes/scan", response_model=NetworkScanResponse)
def scan_and_rank_network(top_k: Optional[int] = Query(default=None)):
    """Actively scan the local LAN via UDP broadcast + /24 subnet sweep, combine
    with registered workers, identify which PCs/laptops are free, and rank them
    by maximum available capacity."""
    lan_discovered = scan_lan_devices(timeout_s=0.6)
    active_registered = _collect_active_workers()

    merged_by_host: Dict[str, Dict] = {}
    for d in lan_discovered + active_registered:
        key = f"{d.get('hostname')}@{d.get('ip_address')}"
        merged_by_host[key] = d

    ranked = rank_devices(list(merged_by_host.values()), max_selected=top_k)
    free_count = sum(1 for n in ranked if n.is_free)
    return NetworkScanResponse(
        subnet=get_local_subnet(),
        total_discovered=len(ranked),
        free_devices=free_count,
        ranked_nodes=ranked,
    )


@app.get("/work/{worker_id}", response_model=WorkItem)
def get_work(worker_id: str, x_token: str = Header()):
    _auth(worker_id, x_token)
    _reclaim_expired_leases()

    # Check this device's capacity rank & whether it is selected among free top-ranked nodes
    roster = _get_ranked_roster()
    node_entry = next((n for n in roster if n.worker_id == worker_id), None)
    if node_entry and not node_entry.selected_for_training:
        raise HTTPException(
            429,
            f"device rank #{node_entry.rank} is currently busy or outranked by higher-capacity free nodes",
        )

    job_id = r.lpop("jobs:pending")
    if job_id is None:
        raise HTTPException(404, "no work available, retry with backoff")

    lease_expiry = time.time() + LEASE_SECONDS
    r.zadd("jobs:leased", {job_id: lease_expiry})
    job = r.hgetall(f"job:{job_id}")

    # Scale batch_size and local_steps to the device's ranked capacity
    bs = node_entry.assigned_batch_size if node_entry else int(job.get("batch_size", 8))
    steps = node_entry.assigned_local_steps if node_entry else LOCAL_STEPS
    rank_num = node_entry.rank if node_entry else 1
    score = node_entry.capacity_score if node_entry else 0.0

    return WorkItem(
        job_id=job_id,
        model_version=int(r.get("model:version") or 0),
        shard_url=job["shard_url"],
        local_steps=steps,
        lr=float(job.get("lr", 3e-4)),
        batch_size=bs,
        worker_rank=rank_num,
        capacity_score=score,
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
    ranked = _get_ranked_roster()
    return {
        "atharva_version": VERSION,
        "license": {k: _license[k] for k in ("customer", "plan", "expires", "_grace")},
        "model_version": int(r.get("model:version") or 0),
        "pending_jobs": r.llen("jobs:pending"),
        "leased_jobs": r.zcard("jobs:leased"),
        "unaggregated_deltas": r.llen("round:deltas"),
        "ranked_nodes": [n.model_dump() for n in ranked],
    }
