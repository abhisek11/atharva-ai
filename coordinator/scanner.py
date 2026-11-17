"""Real LAN Device Scanner & Capacity Ranker for atharva.ai.

Performs two-layer discovery on the local subnet:
  1. UDP Port 49152 Beacon Scan (`ATHARVA_SCAN_V1`) for live Atharva worker agents.
  2. ICMP / ARP neighbor table (`ip neigh show` + reverse DNS) to discover real
     PCs and laptops on the LAN (e.g. `MacBookAir.lan`, `UI-5CG350652G.lan`, `abhisek`)
     even before `worker/client.py` is started on them, while filtering out routers,
     set-top boxes, cameras, and phones.
"""
import concurrent.futures
import ipaddress
import json
import os
import socket
import subprocess
import time
from typing import Dict, List, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.schemas import RankedNode, compute_capacity_score
from worker.probe import DISCOVERY_MAGIC, DISCOVERY_PORT, get_lan_ip, probe_local_device

# Hostname substrings for non-PC appliances (routers, set-top boxes, IoT cameras, phones)
NON_PC_PATTERNS = (
    "jiofiber", "router", "gateway", "settopbox", "stb", "qubo",
    "iphone", "ipad", "samsung", "android", "pixel", "oneplus", "xiaomi", "redmi",
)


def get_local_subnet() -> str:
    ip = get_lan_ip()
    if ip.startswith("127."):
        return "127.0.0.1/32"
    parts = ip.split(".")
    return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"


def _is_non_compute_appliance(ip: str, hostname: str) -> bool:
    if ip.endswith(".1") or ip.endswith(".254"):
        return True
    h = hostname.lower()
    return any(p in h for p in NON_PC_PATTERNS)


def _infer_lan_pc_profile(ip: str, hostname: str, mac: str) -> Dict:
    """For a real PC/laptop discovered on the LAN via ARP/DNS that isn't running
    the UDP beacon yet, infer a sensible baseline hardware profile from its hostname."""
    h = hostname.lower()
    if "macbook" in h or "imac" in h or "macstudio" in h or "mac-mini" in h or "mac" in h:
        # Apple Silicon MacBook Air / Pro (Unified Memory — do NOT double-count VRAM)
        return {
            "worker_id": f"lan-{hostname.split('.')[0]}",
            "hostname": hostname,
            "ip_address": ip,
            "mac_address": mac,
            "device_type": "mps",
            "gpu_name": "Apple M1/M-Series (Unified Memory)",
            "unified_memory": True,
            "vram_total_gb": 0.0,
            "vram_free_gb": 0.0,
            "ram_total_gb": 8.0,
            "ram_free_gb": 5.4,
            "cpu_cores": 8,
            "cpu_load_pct": 12.0,
            "gpu_load_pct": 8.0,
            "is_free": True,
            "agent_online": False,
        }
    return {
        "worker_id": f"lan-{hostname.split('.')[0]}",
        "hostname": hostname,
        "ip_address": ip,
        "mac_address": mac,
        "device_type": "cpu",
        "gpu_name": "Laptop / PC CPU (Agent Not Running)",
        "unified_memory": False,
        "vram_total_gb": 0.0,
        "vram_free_gb": 0.0,
        "ram_total_gb": 16.0,
        "ram_free_gb": 10.2,
        "cpu_cores": 8,
        "cpu_load_pct": 18.0,
        "gpu_load_pct": 0.0,
        "is_free": True,
        "agent_online": False,
    }


def _discover_arp_lan_hosts(subnet_str: str, include_appliances: bool = False) -> Dict[str, Dict]:
    """Sweep the local /24 subnet and read the OS neighbor table (`ip neigh show`)
    to find real connected hosts on the LAN."""
    try:
        net = ipaddress.ip_network(subnet_str, strict=False)
        hosts = [str(h) for h in net.hosts()]
    except Exception:
        hosts = []

    def _ping(ip: str):
        try:
            subprocess.run(
                ["ping", "-c", "1", "-W", "1", ip],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=1.2,
            )
        except Exception:
            pass

    if hosts and len(hosts) <= 256:
        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as ex:
            list(ex.map(_ping, hosts))

    found: Dict[str, Dict] = {}
    try:
        out = subprocess.run(
            ["ip", "neigh", "show"], capture_output=True, text=True, timeout=3
        ).stdout
        for line in out.splitlines():
            if "lladdr" not in line:
                continue
            parts = line.split()
            ip = parts[0]
            if ":" in ip:  # skip IPv6 link-local entries
                continue
            mac = parts[parts.index("lladdr") + 1] if "lladdr" in parts else ""
            try:
                hostname = socket.gethostbyaddr(ip)[0]
            except Exception:
                hostname = f"host-{ip.split('.')[-1]}.lan"

            if not include_appliances and _is_non_compute_appliance(ip, hostname):
                continue

            found[ip] = _infer_lan_pc_profile(ip, hostname, mac)
    except Exception:
        pass

    return found


def scan_lan_devices(
    timeout_s: float = 0.6,
    port: int = DISCOVERY_PORT,
    include_arp_pcs: bool = True,
    include_appliances: bool = False,
) -> List[Dict]:
    """Scan the local network via UDP broadcast + /24 ARP/neighbor discovery."""
    subnet_str = get_local_subnet()
    local_hw = probe_local_device()
    discovered_by_ip: Dict[str, Dict] = {local_hw["ip_address"]: local_hw}

    # 1. Discover real PCs/laptops on the LAN via ARP + reverse DNS
    if include_arp_pcs and not subnet_str.startswith("127."):
        arp_hosts = _discover_arp_lan_hosts(subnet_str, include_appliances=include_appliances)
        for ip, info in arp_hosts.items():
            if ip not in discovered_by_ip:
                discovered_by_ip[ip] = info

    # 2. Query UDP Port 49152 for live Atharva worker beacons (overrides inferred profile with exact telemetry)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(0.08)

    targets = ["255.255.255.255", "127.0.0.1"] + list(discovered_by_ip.keys())
    try:
        net = ipaddress.ip_network(subnet_str, strict=False)
        targets.append(str(net.broadcast_address))
    except Exception:
        pass

    for ip in set(targets):
        try:
            sock.sendto(DISCOVERY_MAGIC, (ip, port))
        except Exception:
            pass

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            data, addr = sock.recvfrom(4096)
            info = json.loads(data.decode("utf-8"))
            real_ip = addr[0] if addr[0] != "127.0.0.1" else info.get("ip_address", local_hw["ip_address"])
            info["ip_address"] = real_ip
            info["agent_online"] = True
            discovered_by_ip[real_ip] = info
        except socket.timeout:
            continue
        except Exception:
            continue

    sock.close()
    return list(discovered_by_ip.values())


def allocate_workload_for_score(score: float, device_type: str, is_free: bool) -> tuple[int, int]:
    """Scale (batch_size, local_steps) proportional to the device's ranked capacity."""
    if not is_free:
        return 0, 0
    if score >= 250 or (device_type in ("cuda", "tpu") and score >= 160):
        return 16, 250   # Tier 1: High-capacity GPU/TPU workstation
    if score >= 70:
        return 8, 160    # Tier 2: Mid-range GPU / Apple Silicon M-series / High-RAM PC
    if score >= 35:
        return 4, 100    # Tier 3: Standard multi-core laptop
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
        unified_mem = bool(n.get("unified_memory", dev_type == "mps"))
        # Never duplicate unified memory into discrete VRAM
        vram_total = 0.0 if unified_mem else float(n.get("vram_total_gb", n.get("vram_gb", 0.0)))
        vram_free = 0.0 if unified_mem else float(n.get("vram_free_gb", vram_total))
        ram_total = float(n.get("ram_total_gb", 8.0))
        ram_free = float(n.get("ram_free_gb", ram_total * 0.6))
        cores = int(n.get("cpu_cores", 4))
        cpu_load = float(n.get("cpu_load_pct", 0.0))
        gpu_load = float(n.get("gpu_load_pct", 0.0))
        is_free = str(n.get("is_free", True)).lower() not in ("false", "0", "no")
        agent_online = str(n.get("agent_online", True)).lower() not in ("false", "0", "no")

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
            "mac_address": str(n.get("mac_address", "")),
            "device_type": dev_type,
            "gpu_name": str(n.get("gpu_name", "cpu")),
            "unified_memory": unified_mem,
            "vram_total_gb": vram_total,
            "vram_free_gb": vram_free,
            "ram_total_gb": ram_total,
            "ram_free_gb": ram_free,
            "cpu_cores": cores,
            "cpu_load_pct": cpu_load,
            "gpu_load_pct": gpu_load,
            "is_free": is_free,
            "agent_online": agent_online,
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
