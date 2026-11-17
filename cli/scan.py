"""`atharva scan` — Scan the local network for real PCs/laptops, inspect which
devices are FREE and how much GPU/TPU/RAM capacity they have (without
double-counting Apple Silicon unified memory), and rank them from Rank #1
(maximum available capacity) downward.

Usage:
    python3 cli/scan.py                  # live LAN scan of real PCs/laptops on your subnet
    python3 cli/scan.py --all-lan        # also include routers/phones/IoT found in ARP table
    python3 cli/scan.py --top 3          # select top 3 highest-capacity free devices
    python3 cli/scan.py --json           # output ranked JSON
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coordinator.scanner import get_local_subnet, rank_devices, scan_lan_devices


def main():
    ap = argparse.ArgumentParser(description="Scan LAN for free PCs/laptops and rank by capacity.")
    ap.add_argument("--top", type=int, default=None, help="Select top-K highest capacity free devices")
    ap.add_argument("--timeout", type=float, default=0.6, help="UDP LAN scan timeout in seconds")
    ap.add_argument("--all-lan", action="store_true", help="Include non-PC appliances/phones from ARP table")
    ap.add_argument("--json", action="store_true", help="Output raw JSON")
    args = ap.parse_args()

    subnet = get_local_subnet()
    discovered = scan_lan_devices(
        timeout_s=args.timeout,
        include_arp_pcs=True,
        include_appliances=args.all_lan,
    )
    ranked = rank_devices(discovered, max_selected=args.top)

    if args.json:
        print(json.dumps({
            "subnet": subnet,
            "total_discovered": len(ranked),
            "free_devices": sum(1 for n in ranked if n.is_free),
            "ranked_nodes": [n.model_dump() for n in ranked],
        }, indent=2))
        return

    print(f"\n[atharva.ai Real LAN Scanner] Subnet: {subnet} | Discovered: {len(ranked)} PC/Laptop(s)\n")
    header = (
        f"{'RANK':<6} {'STATUS':<8} {'HOSTNAME':<22} {'IP ADDRESS':<16} "
        f"{'ACCELERATOR / CPU':<34} {'DISCRETE VRAM':<15} {'FREE RAM':<14} "
        f"{'LOAD':<6} {'SCORE':<7} {'CHOSEN (BS/STEPS)'}"
    )
    print(header)
    print("-" * len(header))

    for n in ranked:
        status_str = "FREE" if n.is_free else "BUSY"
        if n.unified_memory or n.device_type == "mps":
            vram_str = "Unified w/ RAM"
            ram_str = f"{n.ram_free_gb:.1f}/{n.ram_total_gb:.0f}GB (Uni)"
        elif n.vram_total_gb > 0:
            vram_str = f"{n.vram_free_gb:.1f}/{n.vram_total_gb:.0f}GB"
            ram_str = f"{n.ram_free_gb:.1f}/{n.ram_total_gb:.0f}GB"
        else:
            vram_str = "None (CPU)"
            ram_str = f"{n.ram_free_gb:.1f}/{n.ram_total_gb:.0f}GB"

        load_val = n.gpu_load_pct if n.device_type in ("cuda", "tpu", "mps") else n.cpu_load_pct
        load_str = f"{load_val:.0f}%"
        chosen_str = (
            f"YES (bs={n.assigned_batch_size}, steps={n.assigned_local_steps})"
            if n.selected_for_training
            else "SKIPPED"
        )
        print(
            f"#{n.rank:<5} {status_str:<8} {n.hostname:<22} {n.ip_address:<16} "
            f"{n.gpu_name:<34.34} {vram_str:<15} {ram_str:<14} "
            f"{load_str:<6} {n.capacity_score:<7.1f} {chosen_str}"
        )
    print()


if __name__ == "__main__":
    main()
