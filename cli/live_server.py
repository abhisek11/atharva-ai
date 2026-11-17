"""Live HTTP server for atharva.ai Landing Page & Dashboard (`:8080`).

Serves static files (`index.html`, `dashboard.html`) AND provides a real-time
`/api/scan` endpoint that executes `scan_lan_devices()` + `rank_devices()`
against the host's actual LAN (`192.168.31.0/24`).
"""
import http.server
import json
import os
import sys
import urllib.parse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from coordinator.scanner import get_local_subnet, rank_devices, scan_lan_devices


class AtharvaHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT_DIR, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/scan":
            qs = urllib.parse.parse_qs(parsed.query)
            top_k = int(qs["top"][0]) if "top" in qs and qs["top"][0].isdigit() else None
            all_lan = qs.get("all_lan", ["0"])[0] == "1"

            discovered = scan_lan_devices(
                timeout_s=0.4,
                include_arp_pcs=True,
                include_appliances=all_lan,
            )
            ranked = rank_devices(discovered, max_selected=top_k)
            payload = {
                "subnet": get_local_subnet(),
                "total_discovered": len(ranked),
                "free_devices": sum(1 for n in ranked if n.is_free),
                "ranked_nodes": [n.model_dump() for n in ranked],
            }
            data = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return

        if parsed.path == "/dashboard":
            self.path = "/dashboard.html"
        return super().do_GET()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), AtharvaHandler)
    print(f"[atharva live server] Serving on http://localhost:{port} (with live /api/scan)")
    server.serve_forever()
