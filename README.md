# atharva.ai — volunteer distributed training

Coordinator + worker skeleton for DiLoCo-style local-SGD training over
home internet connections.

## Deploy the coordinator (your E2E box)

```bash
docker compose up -d --build
docker compose exec api python coordinator/seed_jobs.py --shards 100
curl localhost:8000/status
```

Put nginx + certbot in front of :8000 (same pattern as roadlink.co.in) and
serve `/model/*` with `sendfile` or push weights to R2 — don't stream
checkpoints through uvicorn once the model is non-trivial.

## Run a worker (friends' machines)

```bash
pip install -r requirements-worker.txt
COORDINATOR=https://atharva.ai python worker/client.py
```

## How it flows

1. Worker registers, gets an id + token, heartbeats every 30s.
2. `GET /work` leases a job (Redis zset, 10-min lease). Crash → lease
   expires → job re-queued. Jobs are idempotent.
3. Worker syncs to the latest weights if stale, trains `LOCAL_STEPS`
   (default 200) on its shard, computes `local − global` delta.
4. Delta is top-k sparsified (1%) + fp16 → ~100x smaller. Dropped
   coordinates go into an error-feedback residual, added back next round.
5. `POST /gradients` — coordinator accepts unless the delta is >1 version
   stale, in which case the job is re-queued.
6. Aggregator drains deltas every 60s, weighted-averages by steps,
   applies with OUTER_LR, writes `model_v{N+1}.pt`, bumps the version.
   Stragglers land in the next window; nothing blocks.

## What to replace before scaling

- `_fetch_shard` — random tokens now; point at tokenized `.npy` shards on R2/S3.
- `build_model` — 10M-param toy; validate the pipeline end-to-end first.
- Outer step — plain averaging; the full DiLoCo recipe uses Nesterov
  momentum on the averaged delta (big convergence win, ~15 lines).
- Verification — duplicate an occasional shard to two workers, compare
  delta norms, ban outliers.
- Auth — tokens live in Redis; fine for friends, add an allowlist before
  opening registration publicly.

## Self-hosted operations

- **License**: coordinator refuses to start without a valid signed
  `license.json` (Ed25519, verified fully offline — air-gap safe).
  14-day grace period after expiry so renewals never brick a training run.
  Set `ATHARVA_DEV_MODE=1` to skip during development.
  Issue licenses: `python coordinator/license.py keygen|issue` (your side only —
  the private key never ships).
- **Diagnostics**: customers run `python cli/doctor.py` when something breaks.
  Produces a secret-redacted tar.gz (versions, GPU inventory, queue depths,
  docker logs, license payload) safe to email to support. This is your
  support workflow for deployments you can't SSH into.
- **Versioning**: `shared/version.py` is stamped into /status, heartbeats,
  and doctor bundles. Bump every release; you cannot hotfix self-hosted
  customers, so migration scripts and disciplined versioning replace
  "just redeploy".
