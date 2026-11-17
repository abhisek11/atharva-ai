"""atharva.ai worker client with LAN Auto-Discovery Beacon & Live Capacity Telemetry.

Usage:
    COORDINATOR=https://atharva.ai python worker/client.py

Loop:
  start LAN UDP beacon -> register with full GPU/TPU/RAM capacity ->
  [pull model if stale -> get work (scaled to rank) -> train local_steps ->
   compress (params_after - params_before) -> upload] -> repeat
"""
import io
import os
import sys
import threading
import time

import requests
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from worker.compression import compress_state_delta, decompress_state_delta
from worker.model import build_model
from worker.probe import probe_local_device, start_lan_discovery_beacon

COORDINATOR = os.getenv("COORDINATOR", "http://localhost:8000")
BACKOFF_MAX_S = 120


def _resolve_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _resolve_torch_device()


class Worker:
    def __init__(self):
        self.model = build_model().to(DEVICE)
        self.model_version = -1
        self.residual = {}  # error-feedback buffer, name -> flat tensor
        self.training_active = False
        self.worker_id = ""
        self.rank = 1
        self.capacity_score = 0.0

        # Start UDP LAN discovery beacon so network scans find this PC/laptop
        start_lan_discovery_beacon(
            state_fn=lambda: probe_local_device(
                training_active=self.training_active,
                worker_id=self.worker_id,
            )
        )

        self._register()
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()

    def _register(self):
        hw = probe_local_device(training_active=False)
        resp = requests.post(f"{COORDINATOR}/register", json={
            "hostname": hw["hostname"],
            "ip_address": hw["ip_address"],
            "device_type": hw["device_type"],
            "gpu_name": hw["gpu_name"],
            "vram_gb": hw["vram_total_gb"],
            "vram_free_gb": hw["vram_free_gb"],
            "ram_total_gb": hw["ram_total_gb"],
            "ram_free_gb": hw["ram_free_gb"],
            "cpu_cores": hw["cpu_cores"],
            "cpu_load_pct": hw["cpu_load_pct"],
            "gpu_load_pct": hw["gpu_load_pct"],
            "is_free": hw["is_free"],
            "torch_version": hw["torch_version"],
        }, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        self.worker_id = data["worker_id"]
        self.token = data["token"]
        self.rank = data.get("rank", 1)
        self.capacity_score = data.get("capacity_score", hw["capacity_score"])
        self.hb_interval = data["heartbeat_interval_s"]
        print(
            f"[worker] registered as {self.worker_id} on {hw['gpu_name']} "
            f"(IP={hw['ip_address']}, Rank=#{self.rank}, Score={self.capacity_score})"
        )

    def _headers(self):
        return {"x-token": self.token}

    def _heartbeat_loop(self):
        while True:
            try:
                hw = probe_local_device(
                    training_active=self.training_active,
                    worker_id=self.worker_id,
                )
                requests.post(
                    f"{COORDINATOR}/heartbeat/{self.worker_id}",
                    headers=self._headers(),
                    json={
                        "vram_free_gb": hw["vram_free_gb"],
                        "ram_free_gb": hw["ram_free_gb"],
                        "cpu_load_pct": hw["cpu_load_pct"],
                        "gpu_load_pct": hw["gpu_load_pct"],
                        "is_free": hw["is_free"],
                    },
                    timeout=10,
                )
            except requests.RequestException:
                pass
            time.sleep(self.hb_interval)

    def _sync_model(self, target_version: int):
        if target_version == self.model_version:
            return
        resp = requests.get(f"{COORDINATOR}/model/{target_version}", timeout=300)
        resp.raise_for_status()
        state = torch.load(io.BytesIO(resp.content), map_location=DEVICE)
        self.model.load_state_dict(state)
        self.model_version = target_version
        print(f"[worker] synced to model v{target_version}")

    def _fetch_shard(self, work: dict) -> torch.Tensor:
        """Return a LongTensor [num_seqs, seq_len] of token ids from the fed batch."""
        token_ids = work.get("token_ids")
        if token_ids and len(token_ids) >= 128:
            t = torch.tensor(token_ids, dtype=torch.long) % 50257
            repeats = max(1, (256 * 128 + len(t) - 1) // len(t))
            t = t.repeat(repeats)[: 256 * 128]
            return t.view(256, 128)
        return torch.randint(0, 50257, (256, 128))

    def _train(self, work: dict) -> tuple[str, float, dict]:
        self.training_active = True
        try:
            data = self._fetch_shard(work)
            before = {n: p.detach().clone()
                      for n, p in self.model.named_parameters()}

            opt = torch.optim.AdamW(self.model.parameters(), lr=work["lr"])
            self.model.train()
            losses, bs = [], work["batch_size"]
            for step in range(work["local_steps"]):
                i = (step * bs) % max(1, len(data) - bs)
                batch = data[i:i + bs].to(DEVICE)
                logits = self.model(batch[:, :-1])
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch[:, 1:].reshape(-1),
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())

            # delta = local - global, plus error-feedback residual
            delta = {}
            for n, p in self.model.named_parameters():
                d = (p.detach() - before[n]).flatten().cpu()
                if n in self.residual:
                    d = d + self.residual[n]
                delta[n] = d
            payload = compress_state_delta({n: d for n, d in delta.items()})
            kept = decompress_state_delta(payload)
            new_residual = {n: delta[n] - kept[n] for n in delta}

            # reset local params to global — the coordinator owns the trajectory
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    p.copy_(before[n])

            return payload, sum(losses) / len(losses), new_residual
        finally:
            self.training_active = False

    def run(self):
        backoff = 5
        while True:
            try:
                resp = requests.get(f"{COORDINATOR}/work/{self.worker_id}",
                                    headers=self._headers(), timeout=30)
                if resp.status_code in (404, 429):
                    time.sleep(backoff)
                    backoff = min(backoff * 2, BACKOFF_MAX_S)
                    continue
                resp.raise_for_status()
                backoff = 5
                work = resp.json()
                self.rank = work.get("worker_rank", self.rank)

                self._sync_model(work["model_version"])
                payload, avg_loss, new_residual = self._train(work)

                ack = requests.post(f"{COORDINATOR}/gradients",
                                    headers=self._headers(), json={
                    "job_id": work["job_id"],
                    "worker_id": self.worker_id,
                    "model_version": self.model_version,
                    "steps_completed": work["local_steps"],
                    "loss": avg_loss,
                    "payload_b64": payload,
                }, timeout=120).json()
                if ack.get("accepted"):
                    self.residual = new_residual
                print(
                    f"[worker rank=#{self.rank}] job {work['job_id'][:8]} "
                    f"(bs={work['batch_size']}, steps={work['local_steps']}) "
                    f"loss={avg_loss:.4f} accepted={ack['accepted']}"
                )
            except requests.RequestException as e:
                print(f"[worker] network error: {e}, retrying in {backoff}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, BACKOFF_MAX_S)


if __name__ == "__main__":
    Worker().run()
