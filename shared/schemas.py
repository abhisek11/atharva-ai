"""Shared wire schemas between coordinator and workers."""
from pydantic import BaseModel
from typing import Optional


class RegisterRequest(BaseModel):
    hostname: str
    gpu_name: str
    vram_gb: float
    torch_version: str


class RegisterResponse(BaseModel):
    worker_id: str
    token: str
    heartbeat_interval_s: int = 30


class WorkItem(BaseModel):
    job_id: str
    model_version: int
    shard_url: str            # data shard the worker should train on
    local_steps: int          # DiLoCo-style: steps before sync
    lr: float
    batch_size: int
    lease_expires_at: float   # unix ts; coordinator reclaims after this


class GradientUpload(BaseModel):
    job_id: str
    worker_id: str
    model_version: int        # version the delta was computed against
    steps_completed: int
    loss: float
    # top-k sparse delta, base64 of the packed payload (see worker/compression.py)
    payload_b64: str


class AckResponse(BaseModel):
    accepted: bool
    reason: Optional[str] = None
    latest_model_version: int
