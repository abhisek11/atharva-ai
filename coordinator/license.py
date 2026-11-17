"""Signed license files — works fully offline (air-gapped customers).

Model:
  - You (atharva.ai) hold an Ed25519 private key. NEVER ships to customers.
  - Each customer gets license.json: payload + signature over the payload.
  - Coordinator verifies the signature against the public key baked into
    this file at build time. No network call required.

Payload:
  {"customer": "Acme GCC", "max_gpus": 100,
   "expires": "2027-07-11", "plan": "enterprise"}

Grace period: after expiry, WARN for GRACE_DAYS before refusing new jobs —
never brick a customer's training run at midnight on renewal day.

Issue a license (your side, not shipped):
    python coordinator/license.py issue --customer "Acme" --max-gpus 100 \
        --expires 2027-07-11 --private-key ~/.atharva/signing.key
Generate a keypair once:
    python coordinator/license.py keygen --out ~/.atharva/signing.key
"""
import argparse
import base64
import json
import os
import sys
from datetime import date, datetime, timedelta

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey,
)

# Baked in at build time. Replace with your real public key (base64, 32 bytes).
ATHARVA_PUBLIC_KEY_B64 = os.getenv("ATHARVA_PUBLIC_KEY_B64", "REPLACE_ME")

LICENSE_PATH = os.getenv("LICENSE_PATH", "/data/license.json")
GRACE_DAYS = 14


class LicenseError(Exception):
    pass


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def load_and_verify(path: str = LICENSE_PATH) -> dict:
    """Returns the payload if valid. Raises LicenseError otherwise.
    Sets payload['_grace'] = True if expired but inside the grace window."""
    if not os.path.exists(path):
        raise LicenseError(f"no license file at {path}")
    with open(path) as f:
        blob = json.load(f)

    payload, sig_b64 = blob["payload"], blob["signature"]
    pub = Ed25519PublicKey.from_public_bytes(
        base64.b64decode(ATHARVA_PUBLIC_KEY_B64))
    try:
        pub.verify(base64.b64decode(sig_b64), _canonical(payload))
    except Exception:
        raise LicenseError("license signature invalid")

    expires = date.fromisoformat(payload["expires"])
    today = date.today()
    if today > expires + timedelta(days=GRACE_DAYS):
        raise LicenseError(f"license expired {expires} (grace period over)")
    payload["_grace"] = today > expires
    return payload


def check_gpu_count(payload: dict, active_workers: int):
    if active_workers > payload["max_gpus"]:
        raise LicenseError(
            f"{active_workers} active GPUs exceeds licensed {payload['max_gpus']}")


# ---- issuer-side commands (never deployed to customers) ----

def _keygen(out: str):
    key = Ed25519PrivateKey.generate()
    from cryptography.hazmat.primitives import serialization as ser
    priv = key.private_bytes(ser.Encoding.Raw, ser.PrivateFormat.Raw,
                             ser.NoEncryption())
    pub = key.public_key().public_bytes(ser.Encoding.Raw,
                                        ser.PublicFormat.Raw)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(priv)
    os.chmod(out, 0o600)
    print(f"private key -> {out}")
    print(f"PUBLIC key (bake into license.py / env): {base64.b64encode(pub).decode()}")


def _issue(args):
    with open(args.private_key, "rb") as f:
        key = Ed25519PrivateKey.from_private_bytes(f.read())
    payload = {
        "customer": args.customer, "max_gpus": args.max_gpus,
        "expires": args.expires, "plan": args.plan,
        "issued": datetime.utcnow().isoformat() + "Z",
    }
    sig = key.sign(_canonical(payload))
    out = {"payload": payload, "signature": base64.b64encode(sig).decode()}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    kg = sub.add_parser("keygen"); kg.add_argument("--out", required=True)
    iss = sub.add_parser("issue")
    iss.add_argument("--customer", required=True)
    iss.add_argument("--max-gpus", type=int, required=True)
    iss.add_argument("--expires", required=True)
    iss.add_argument("--plan", default="enterprise")
    iss.add_argument("--private-key", required=True)
    a = ap.parse_args()
    _keygen(a.out) if a.cmd == "keygen" else _issue(a)
