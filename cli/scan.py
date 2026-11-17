"""`atharva scan` — Scan the local network for PCs/laptops, inspect which devices
are FREE and how much GPU/TPU/RAM capacity they have, and rank them from
Rank #1 (maximum available capacity) downward.

Usage:
    python cli/scan.py                  # live LAN scan + local machine probe
    python cli/scan.py --top 3          # select top 3 highest-capacity free devices
    python cli/scan.py --demo           # include sample LAN PCs/laptops to preview ranking
    python cli/scan.py --json           # output ranked JSON
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coordinator.scanner import get_local_subnet, rank_devices, scan_lan_devices
from worker.probe import probe_local_device


DEMO_LAN_PCS = [
    {
        "worker_id": "lan-workstation-01",
        "hostname": "rtx4090-rig",
        "ip_address": "192.168.1.14",
        "device_type": "cuda",
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "vram_total_gb": 24.0,
        "vram_free_gb": 22.4,
        "ram_total_gb": 64.0,
        "ram_free_gb": 51.0,
        "cpu_cores": 24,
        "cpu_load_pct": 12.0,
        "gpu_load_pct": 5.0,
        "is_free": True,
    },
    {
        "worker_id": "lan-tpu-gateway",
        "hostname": "tpu-node-02",
        "ip_address": "192.168.1.22",
        "device_type": "tpu",
        "gpu_name": "Cloud TPU v5e Pod Slice",
        "vram_total_gb": 16.0,
        "vram_free_gb": 15.2,
        "ram_total_gb": 32.0,
        "ram_free_gb": 26.0,
        "cpu_cores": 16,
        "cpu_load_pct": 8.0,
        "gpu_load_pct": 4.0,
        "is_free": True,
    },
    {
        "worker_id": "lan-macbook-pro",
        "hostname": "macbook-m3-max",
        "ip_address": "192.168.1.35",
        "device_type": "mps",
        "gpu_name": "Apple M3 Max Unified GPU",
        "vram_total_gb": 36.0,
        "vram_free_gb": 24.0,
        "ram_total_gb": 36.0,
        "ram_free_gb": 24.0,
        "cpu_cores": 14,
        "cpu_load_pct": 18.0,
        "gpu_load_pct": 10.0,
        "is_free": True,
    },
    {
        "worker_id": "lan-office-laptop",
        "hostname": "thinkpad-ryzen7",
        "ip_address": "192.168.1.48",
        "device_type": "cpu",
        "gpu_name": "CPU (AMD Ryzen 7 7840U)",
        "vram_total_gb": 0.0,
        "vram_free_gb": 0.0,
        "ram_total_gb": 32.0,
        "ram_free_gb": 21.5,
        "cpu_cores": 16,
        "cpu_load_pct": 15.0,
        "gpu_load_pct": 0.0,
        "is_free": True,
    },
    {
        "worker_id": "lan-gaming-pc-busy",
        "hostname": "rtx3080-desktop",
        "ip_address": "192.168.1.61",
        "device_type": "cuda",
        "gpu_name": "NVIDIA GeForce RTX 3080",
        "vram_total_gb": 10.0,
        "vram_free_gb": 1.1,
        "ram_total_gb": 32.0,
        "ram_free_gb": 8.4,
        "cpu_cores": 16,
        "cpu_load_pct": 78.0,
        "gpu_load_pct": 94.0,
        "is_free": False,  # Busy! Should be ranked at bottom & skipped
    },
]


def main():
    ap = argparse.ArgumentParser(description="Scan LAN for free PCs/laptops and rank by capacity.")
    ap.add_argument("--top", type=int, default=None, help="Select top-K highest capacity free devices")
    ap.add_argument("--timeout", type=float, default=0.8, help="UDP LAN scan timeout in seconds")
    ap.add_argument("--demo", action="store_true", help="Include sample LAN PCs/laptops alongside this host")
    ap.add_argument("--json", action="store_true", help="Output raw JSON")
    args = ap.parse_args()

    subnet = get_local_subnet()
    discovered = scan_lan_devices(timeout_s=args.timeout)

    # Always include the local machine if no remote beacon replied for it yet
    local_hw = probe_local_device()
    if not any(d.get("hostname") == local_hw["hostname"] for d in discovered):
        discovered.append(local_hw)

    if args.demo:
        prefix = ".".join(local_hw["ip_address"].split(".")[:3]) if "." in local_hw["ip_address"] else "192.168.1"
        for idx, demo_node in enumerate(DEMO_LAN_PCS, start=10):
            copy_node = dict(demo_node)
            copy_node["ip_address"] = f"{prefix}.{idx * 7}"
            discovered.append(copy_node)

    ranked = rank_devices(discovered, max_selected=args.top)

    if args.json:
        print(json.dumps({
            "subnet": subnet,
            "total_discovered": len(ranked),
            "free_devices": sum(1 for n in ranked if n.is_free),
            "ranked_nodes": [n.model_dump() for n in ranked],
        }, indent=2))
        return

    print(f"\n[atharva.ai LAN Scanner] Subnet: {subnet} | Discovered: {len(ranked)} PC/Laptop(s)\n")
    header = (
        f"{'RANK':<6} {'STATUS':<9} {'HOSTNAME (IP)':<28} {'ACCELERATOR / CPU':<28} "
        f"{'FREE VRAM':<11} {'FREE RAM':<10} {'LOAD':<7} {'SCORE':<8} {'CHOSEN (BS/STEPS)'}"
    )
    print(header)
    print("-" * len(header))

    for n in ranked:
        status_str = "FREE" if n.is_free else "BUSY"
        host_ip = f"{n.hostname} ({n.ip_address})"
        vram_str = f"{n.vram_free_gb:.1f}/{n.vram_total_gb:.0f}GB" if n.vram_total_gb > 0 else "N/A (RAM)"
        ram_str = f"{n.ram_free_gb:.1f}GB"
        load_val = n.gpu_load_pct if n.device_type in ("cuda", "tpu", "mps") else n.cpu_load_pct
        load_str = f"{load_val:.0f}%"
        chosen_str = (
            f"YES (bs={n.assigned_batch_size}, steps={n.assigned_local_steps})"
            if n.selected_for_training
            else "SKIPPED"
        )
        print(
            f"#{n.rank:<5} {status_str:<9} {host_ip:<28.28} {n.gpu_name:<28.28} "
            f"{vram_str:<11} {ram_str:<10} {load_str:<7} {n.capacity_score:<8.1f} {chosen_str}"
        )
    print()


if __name__ == "__main__":
    main()
