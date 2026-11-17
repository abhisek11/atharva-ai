"""Shared wire schemas and capacity-ranking logic between coordinator and workers."""
import json
from typing import List, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    def Field(default=None, default_factory=None):
        return default_factory() if default_factory is not None else default

    class BaseModel:  # Lightweight fallback when running CLI without pydantic
        def __init__(self, **kwargs):
            for k, v in self.__class__.__dict__.items():
                if not k.startswith("_") and not callable(v):
                    setattr(self, k, v)
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self) -> dict:
            return {
                k: (v.model_dump() if hasattr(v, "model_dump") else v)
                for k, v in self.__dict__.items()
            }

        def model_dump_json(self) -> str:
            return json.dumps(self.model_dump())


def compute_capacity_score(
    device_type: str,
    vram_free_gb: float,
    ram_free_gb: float,
    cpu_cores: int,
    cpu_load_pct: float = 0.0,
    gpu_load_pct: float = 0.0,
    is_free: bool = True,
) -> float:
    """Compute a unified capacity score so heterogeneous PCs/laptops on a LAN
    can be ranked from highest available capacity (#1) to lowest.

    Fixes unified memory double-counting:
      - Discrete GPU/TPU (`cuda`, `tpu`): scores discrete `vram_free_gb * 12.0` +
        host `ram_free_gb * 2.5` + accelerator bonus (`+50`).
      - Apple Silicon (`mps`): GPU and CPU share ONE unified memory pool (`ram_free_gb`).
        Never adds `vram_free_gb` on top of `ram_free_gb` — scores `ram_free_gb * 8.0`
        once + unified GPU bonus (`+25`).
      - CPU-only (`cpu`): scores `ram_free_gb * 2.5` + `cpu_cores * 1.5`.
    """
    dev = (device_type or "cpu").lower()

    if dev == "mps":
        # Unified memory pool: count free unified RAM once (do NOT double-count as VRAM + RAM)
        unified_free = max(vram_free_gb, ram_free_gb)
        raw = 25.0 + (max(0.0, unified_free) * 8.0) + (max(1, cpu_cores) * 1.5)
    elif dev in ("cuda", "tpu"):
        raw = (
            50.0
            + (max(0.0, vram_free_gb) * 12.0)
            + (max(0.0, ram_free_gb) * 2.5)
            + (max(1, cpu_cores) * 1.5)
        )
    else:
        raw = (max(0.0, ram_free_gb) * 2.5) + (max(1, cpu_cores) * 1.5)

    primary_load = gpu_load_pct if dev in ("cuda", "tpu", "mps") else cpu_load_pct
    headroom = max(0.1, 1.0 - (min(100.0, max(0.0, primary_load)) / 100.0) * 0.7)
    availability_mult = 1.0 if is_free else 0.15

    return round(raw * headroom * availability_mult, 2)


class RegisterRequest(BaseModel):
    hostname: str
    ip_address: str = "127.0.0.1"
    device_type: str = "cpu"          # "cuda" | "tpu" | "mps" | "cpu"
    gpu_name: str
    vram_gb: float                    # discrete VRAM GB (0.0 for unified MPS / CPU)
    vram_free_gb: float = 0.0         # currently free discrete VRAM GB
    ram_total_gb: float = 0.0         # total system (or unified) RAM GB
    ram_free_gb: float = 0.0          # currently free system (or unified) RAM GB
    unified_memory: bool = False      # True on Apple Silicon (M1/M2/M3/M4)
    cpu_cores: int = 1
    cpu_load_pct: float = 0.0
    gpu_load_pct: float = 0.0
    is_free: bool = True
    torch_version: str


class RegisterResponse(BaseModel):
    worker_id: str
    token: str
    rank: int = 1
    capacity_score: float = 0.0
    heartbeat_interval_s: int = 30


class HeartbeatRequest(BaseModel):
    vram_free_gb: float = 0.0
    ram_free_gb: float = 0.0
    cpu_load_pct: float = 0.0
    gpu_load_pct: float = 0.0
    is_free: bool = True


class RankedNode(BaseModel):
    rank: int
    worker_id: str
    hostname: str
    ip_address: str
    mac_address: str = ""
    device_type: str
    gpu_name: str
    unified_memory: bool = False
    vram_total_gb: float
    vram_free_gb: float
    ram_total_gb: float
    ram_free_gb: float
    cpu_cores: int
    cpu_load_pct: float
    gpu_load_pct: float
    is_free: bool
    agent_online: bool = True
    capacity_score: float
    selected_for_training: bool = True
    assigned_batch_size: int = 8
    assigned_local_steps: int = 200


class NetworkScanResponse(BaseModel):
    subnet: str
    total_discovered: int
    free_devices: int
    ranked_nodes: List[RankedNode] = Field(default_factory=list)


class DataFeedRequest(BaseModel):
    dataset_name: str = "custom-feed"
    raw_text: str                         # text/corpus fed from the main interface
    num_batches: int = 4                  # how many distributed batches to split into
    lr: float = 3e-4
    local_steps: int = 200


class DataFeedResponse(BaseModel):
    dataset_id: str
    dataset_name: str
    total_tokens: int
    batches_created: int
    job_ids: List[str] = Field(default_factory=list)


class WorkItem(BaseModel):
    job_id: str
    model_version: int
    shard_url: str            # data shard URL or inline batch reference
    dataset_name: str = "default"
    batch_index: int = 0
    total_batches: int = 1
    token_ids: Optional[List[int]] = None  # direct batch tokens fed from main UI
    local_steps: int          # DiLoCo-style: steps before sync (scaled by rank/capacity)
    lr: float
    batch_size: int           # scaled to device free VRAM/RAM capacity
    worker_rank: int = 1
    capacity_score: float = 0.0
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
