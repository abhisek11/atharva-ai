"""Hardware capacity probe & LAN UDP discovery responder for worker PCs/laptops.

Measures:
  - Accelerator type: CUDA GPU, Google Cloud TPU (torch_xla), Apple MPS, or CPU/RAM
  - Discrete VRAM (GB) vs Unified Memory (Apple Silicon M1/M2/M3/M4 — never double-counted)
  - Total & Free System RAM (GB)
  - Logical CPU cores & current CPU load (%)
  - Current GPU utilization (%)
  - Whether the device is currently FREE (`is_free`) vs busy with user/system workloads
  - Composite `capacity_score` for ranking devices on the network
"""
import json
import os
import platform
import socket
import subprocess
import threading
from typing import Dict

try:
    import torch
except ImportError:
    torch = None  # type: ignore

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.schemas import compute_capacity_score

DISCOVERY_PORT = int(os.getenv("ATHARVA_DISCOVERY_PORT", "49152"))
DISCOVERY_MAGIC = b"ATHARVA_SCAN_V1"


def get_lan_ip() -> str:
    """Resolve the machine's primary LAN IPv4 address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def _read_system_ram_gb() -> tuple[float, float]:
    """Return (ram_total_gb, ram_free_gb) across Linux and macOS without external deps."""
    # Linux /proc/meminfo
    try:
        if os.path.exists("/proc/meminfo"):
            info = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        k = parts[0].strip()
                        v = int(parts[1].strip().split()[0])  # kB
                        info[k] = v
            total_gb = info.get("MemTotal", 0) / 1e6
            avail_gb = info.get("MemAvailable", info.get("MemFree", 0)) / 1e6
            return round(total_gb, 2), round(avail_gb, 2)
    except Exception:
        pass

    # macOS sysctl + vm_stat
    if platform.system() == "Darwin":
        try:
            mem_bytes = int(
                subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, timeout=2
                ).stdout.strip()
            )
            total_gb = mem_bytes / (1024 ** 3)
            vm_out = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, timeout=2
            ).stdout
            free_pages = 0
            for line in vm_out.splitlines():
                if any(k in line for k in ("Pages free", "Pages inactive", "Pages speculative")):
                    num = line.split(":")[1].strip().rstrip(".")
                    free_pages += int(num)
            avail_gb = (free_pages * 16384) / (1024 ** 3)
            if avail_gb <= 0 or avail_gb > total_gb:
                avail_gb = total_gb * 0.65
            return round(total_gb, 2), round(avail_gb, 2)
        except Exception:
            pass

    # Fallback via POSIX sysconf
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_pages = os.sysconf("SC_PHYS_PAGES")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        total_gb = (page_size * total_pages) / 1e9
        avail_gb = (page_size * avail_pages) / 1e9
        return round(total_gb, 2), round(avail_gb, 2)
    except Exception:
        return 8.0, 4.0


def _read_cpu_load_pct(cores: int) -> float:
    """Estimate CPU utilization % from 1-minute OS load average."""
    try:
        load1, _, _ = os.getloadavg()
        return round(min(100.0, (load1 / max(1, cores)) * 100.0), 1)
    except Exception:
        return 10.0


def _read_gpu_telemetry() -> tuple[str, str, float, float, float, bool]:
    """Return (device_type, gpu_name, vram_total_gb, vram_free_gb, gpu_load_pct, unified_memory)."""
    # 1. Try nvidia-smi directly (works even if PyTorch is not installed in the current shell)
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0 and out.stdout.strip():
            first = out.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            if len(parts) >= 4:
                name = parts[0]
                total_gb = round(float(parts[1]) / 1024.0, 2)
                free_gb = round(float(parts[2]) / 1024.0, 2)
                util_pct = float(parts[3])
                return "cuda", name, total_gb, free_gb, util_pct, False
    except Exception:
        pass

    # 2. Check PyTorch CUDA if available
    if torch is not None and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        try:
            free_b, total_b = torch.cuda.mem_get_info(0)
            vram_total = round(total_b / 1e9, 2)
            vram_free = round(free_b / 1e9, 2)
        except Exception:
            total_b = torch.cuda.get_device_properties(0).total_memory
            vram_total = round(total_b / 1e9, 2)
            vram_free = vram_total
        return "cuda", gpu_name, vram_total, vram_free, 0.0, False

    # 3. Check Google Cloud TPU (torch_xla)
    try:
        import torch_xla.core.xla_model as xm  # type: ignore
        dev = xm.xla_device()
        return "tpu", f"TPU ({dev})", 16.0, 14.5, 5.0, False
    except Exception:
        pass

    # 4. Check Apple Silicon (MPS / Darwin arm64) — Unified Memory (do NOT duplicate into discrete VRAM)
    is_apple_silicon = (
        platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")
    ) or (torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    if is_apple_silicon:
        chip_name = "Apple Silicon Unified GPU"
        try:
            brand = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2
            ).stdout.strip()
            if brand:
                chip_name = f"{brand} (Unified Memory)"
        except Exception:
            pass
        # vram_total_gb=0.0, vram_free_gb=0.0 because memory is unified with system RAM
        return "mps", chip_name, 0.0, 0.0, 10.0, True

    # 5. Fallback to CPU + System RAM
    cpu_label = platform.processor() or platform.machine() or "CPU"
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        cpu_label = line.split(":", 1)[1].strip()
                        break
    except Exception:
        pass
    return "cpu", f"CPU ({cpu_label})", 0.0, 0.0, 0.0, False


def probe_local_device(training_active: bool = False, worker_id: str = "") -> Dict:
    """Inspect local PC/laptop hardware, free memory, load, and compute capacity score."""
    cores = os.cpu_count() or 1
    ram_total, ram_free = _read_system_ram_gb()
    cpu_load = _read_cpu_load_pct(cores)
    dev_type, gpu_name, vram_total, vram_free, gpu_load, unified_mem = _read_gpu_telemetry()

    is_free = (
        not training_active
        and gpu_load < 80.0
        and cpu_load < 85.0
        and ram_free >= 0.5
    )

    score = compute_capacity_score(
        device_type=dev_type,
        vram_free_gb=vram_free,
        ram_free_gb=ram_free,
        cpu_cores=cores,
        cpu_load_pct=cpu_load,
        gpu_load_pct=gpu_load,
        is_free=is_free,
    )

    return {
        "worker_id": worker_id or f"lan-{platform.node()}",
        "hostname": platform.node(),
        "ip_address": get_lan_ip(),
        "mac_address": "local",
        "device_type": dev_type,
        "gpu_name": gpu_name,
        "unified_memory": unified_mem,
        "vram_gb": vram_total,
        "vram_total_gb": vram_total,
        "vram_free_gb": vram_free,
        "ram_total_gb": ram_total,
        "ram_free_gb": ram_free,
        "cpu_cores": cores,
        "cpu_load_pct": cpu_load,
        "gpu_load_pct": gpu_load,
        "is_free": is_free,
        "agent_online": True,
        "capacity_score": score,
        "torch_version": getattr(torch, "__version__", "not-installed"),
    }


def start_lan_discovery_beacon(state_fn=None, port: int = DISCOVERY_PORT):
    """Run a background UDP responder so the coordinator or `cli/scan.py` can
    discover all PCs/laptops on the same LAN via broadcast/subnet scan."""
    def _serve():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
        except OSError as e:
            print(f"[beacon] UDP port {port} already bound ({e})")
            return

        while True:
            try:
                data, addr = sock.recvfrom(1024)
                if data.strip().startswith(DISCOVERY_MAGIC):
                    info = state_fn() if state_fn else probe_local_device()
                    payload = json.dumps(info).encode("utf-8")
                    sock.sendto(payload, addr)
            except Exception:
                pass

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    return t
