"""LAN Device Scanner & Capacity Ranker for atharva.ai.

Discovers multiple PCs/laptops on the same network via UDP broadcast + `/24`
subnet sweep, inspects which devices are currently FREE (`is_free=True`) and
how much free GPU VRAM / TPU / System RAM they have, and ranks them from
Rank #1 (maximum capacity) downward.
"""
import ipaddress
import json
import os
import socket
import time
from typing import Dict, List, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.schemas import RankedNode, compute_capacity_score
from worker.probe import DISCOVERY_MAGIC, DISCOVERY_PORT, get_lan_ip


def get_local_subnet() -> str:
    ip = get_lan_ip()
    if ip.startswith("127."):
        return "127.0.0.1/32"
    parts = ip.split(".")
    return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"


def scan_lan_devices(timeout_s: float = 0.8, port: int = DISCOVERY_PORT) -> List[Dict]:
    """Scan the local network via UDP broadcast + /24 unicast sweep to find all
    PCs and laptops running the Atharva worker beacon."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(0.08)

    subnet_str = get_local_subnet()
    targets = ["255.255.255.255", "127.0.0.1"]

    try:
        net = ipaddress.ip_network(subnet_str, strict=False)
        targets.append(str(net.broadcast_address))
        for host in net.hosts():
            targets.append(str(host))
    except Exception:
        pass

    # Send lightweight UDP probe to broadcast + all /24 hosts on the LAN
    for ip in targets:
        try:
            sock.sendto(DISCOVERY_MAGIC, (ip, port))
        except Exception:
            pass

    discovered_by_host: Dict[str, Dict] = {}
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            data, addr = sock.recvfrom(4096)
            info = json.loads(data.decode("utf-8"))
            info["ip_address"] = addr[0] if addr[0] != "127.0.0.1" else info.get("ip_address", "127.0.0.1")
            key = f"{info.get('hostname', addr[0])}:{info.get('worker_id', '')}"
            discovered_by_host[key] = info
        except socket.timeout:
            continue
        except Exception:
            continue

    sock.close()
    return list(discovered_by_host.values())


def allocate_workload_for_score(score: float, device_type: str, is_free: bool) -> tuple[int, int]:
    """Scale (batch_size, local_steps) proportional to the device's ranked capacity."""
    if not is_free:
        return 0, 0
    if score >= 250 or (device_type in ("cuda", "tpu") and score >= 160):
        return 16, 250   # Tier 1: High-capacity GPU/TPU workstation
    if score >= 100:
        return 8, 200    # Tier 2: Mid-range GPU / Apple Silicon / High-RAM PC
    if score >= 45:
        return 4, 120    # Tier 3: Standard multi-core laptop
    return 2, 60         # Tier 4: Lightweight RAM node


def rank_devices(
    raw_nodes: List[Dict],
    max_selected: Optional[int] = None,
) -> List[RankedNode]:
    """Filter and rank discovered/registered PCs & laptops on the network:
      1. Free devices (`is_free=True`) are always ranked above busy devices.
      2. Within free devices, sort by `capacity_score` descending (maximum capacity = Rank #1).
      3. Select the top-ranked free devices (up to `max_selected` if specified).
    """
    enriched = []
    for n in raw_nodes:
        dev_type = str(n.get("device_type", "cpu")).lower()
        vram_total = float(n.get("vram_total_gb", n.get("vram_gb", 0.0)))
        vram_free = float(n.get("vram_free_gb", vram_total))
        ram_total = float(n.get("ram_total_gb", 8.0))
        ram_free = float(n.get("ram_free_gb", ram_total * 0.6))
        cores = int(n.get("cpu_cores", 4))
        cpu_load = float(n.get("cpu_load_pct", 0.0))
        gpu_load = float(n.get("gpu_load_pct", 0.0))
        is_free = str(n.get("is_free", True)).lower() not in ("false", "0", "no")

        score = compute_capacity_score(
            device_type=dev_type,
            vram_free_gb=vram_free,
            ram_free_gb=ram_free,
            cpu_cores=cores,
            cpu_load_pct=cpu_load,
            gpu_load_pct=gpu_load,
            is_free=is_free,
        )
        enriched.append({
            "worker_id": str(n.get("worker_id", n.get("hostname", "unknown"))),
            "hostname": str(n.get("hostname", "unknown")),
            "ip_address": str(n.get("ip_address", "127.0.0.1")),
            "device_type": dev_type,
            "gpu_name": str(n.get("gpu_name", "cpu")),
            "vram_total_gb": vram_total,
            "vram_free_gb": vram_free,
            "ram_total_gb": ram_total,
            "ram_free_gb": ram_free,
            "cpu_cores": cores,
            "cpu_load_pct": cpu_load,
            "gpu_load_pct": gpu_load,
            "is_free": is_free,
            "capacity_score": score,
        })

    # Sort: Free nodes first (`is_free == True`), then highest `capacity_score`
    enriched.sort(key=lambda x: (1 if x["is_free"] else 0, x["capacity_score"]), reverse=True)

    ranked_models: List[RankedNode] = []
    free_selected_count = 0
    for idx, item in enumerate(enriched, start=1):
        can_select = item["is_free"] and (max_selected is None or free_selected_count < max_selected)
        if can_select:
            free_selected_count += 1
        bs, steps = allocate_workload_for_score(
            item["capacity_score"], item["device_type"], can_select
        )
        ranked_models.append(
            RankedNode(
                rank=idx,
                selected_for_training=can_select,
                assigned_batch_size=bs,
                assigned_local_steps=steps,
                **item,
            )
        )
    return ranked_models
