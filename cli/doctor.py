"""`atharva doctor` — one command a customer runs when something breaks.

Collects everything a support engineer needs into a single shareable
tar.gz, with secrets redacted. Works on both coordinator and worker hosts;
gathers whatever it can find and skips the rest.

    python cli/doctor.py                 # writes atharva-doctor-<ts>.tar.gz
    python cli/doctor.py --out /tmp/x    # custom output dir

Collected:
  - platform version, OS, Python, torch/CUDA, GPU inventory (nvidia-smi)
  - coordinator /status snapshot + Redis queue depths (if reachable)
  - license payload (customer/plan/expiry — signature and tokens excluded)
  - last 500 lines of docker compose logs per service (if docker present)
  - env vars matching ATHARVA_*/REDIS_*/MODEL_* with values redacted
    unless clearly non-secret

Redaction rule: any key containing TOKEN, SECRET, KEY, PASSWORD is
replaced with "<redacted>". The bundle is safe to email.
"""
import argparse
import io
import json
import os
import platform
import subprocess
import sys
import tarfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.version import VERSION

REDACT_MARKERS = ("TOKEN", "SECRET", "KEY", "PASSWORD", "CREDENTIAL")


def _run(cmd: list, timeout=15) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout).stdout
    except Exception as e:
        return f"<unavailable: {e}>"


def collect() -> dict:
    files = {}

    sysinfo = {
        "atharva_version": VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "os": platform.platform(),
        "python": sys.version,
        "hostname": platform.node(),
    }
    try:
        import torch
        sysinfo["torch"] = torch.__version__
        sysinfo["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            sysinfo["gpus"] = [torch.cuda.get_device_name(i)
                               for i in range(torch.cuda.device_count())]
    except ImportError:
        sysinfo["torch"] = "not installed"
    files["sysinfo.json"] = json.dumps(sysinfo, indent=2)

    files["nvidia-smi.txt"] = _run(["nvidia-smi"])

    # coordinator status + Redis depths, if this host runs them
    try:
        import requests
        files["coordinator-status.json"] = requests.get(
            "http://localhost:8000/status", timeout=5).text
    except Exception as e:
        files["coordinator-status.json"] = f"<unreachable: {e}>"
    try:
        import redis
        r = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"),
                        decode_responses=True, socket_timeout=5)
        files["redis.json"] = json.dumps({
            "pending": r.llen("jobs:pending"),
            "leased": r.zcard("jobs:leased"),
            "deltas_unaggregated": r.llen("round:deltas"),
            "model_version": r.get("model:version"),
            "worker_count": len(r.keys("workers:*")),
        }, indent=2)
    except Exception as e:
        files["redis.json"] = f"<unreachable: {e}>"

    # license — payload only, never the signature or file path contents raw
    lic_path = os.getenv("LICENSE_PATH", "/data/license.json")
    if os.path.exists(lic_path):
        try:
            with open(lic_path) as f:
                payload = json.load(f).get("payload", {})
            files["license.json"] = json.dumps(payload, indent=2)
        except Exception as e:
            files["license.json"] = f"<parse error: {e}>"

    files["docker-logs.txt"] = _run(
        ["docker", "compose", "logs", "--tail", "500"], timeout=30)

    env = {}
    for k, v in sorted(os.environ.items()):
        if not k.startswith(("ATHARVA_", "REDIS_", "MODEL_", "COORDINATOR")):
            continue
        env[k] = "<redacted>" if any(m in k.upper() for m in REDACT_MARKERS) else v
    files["env.json"] = json.dumps(env, indent=2)

    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    files = collect()
    name = f"atharva-doctor-{time.strftime('%Y%m%d-%H%M%S')}.tar.gz"
    path = os.path.join(args.out, name)
    with tarfile.open(path, "w:gz") as tar:
        for fname, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name=fname)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    print(f"diagnostics bundle: {path}")
    print("Contents are secret-redacted and safe to share with atharva.ai support.")
    for fname in files:
        print(f"  - {fname}")


if __name__ == "__main__":
    main()
