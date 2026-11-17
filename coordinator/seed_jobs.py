"""Seed the work queue. Run once (or on a cron) to enqueue data shards.

    python coordinator/seed_jobs.py --shards 100
"""
import argparse
import os
import uuid

import redis

r = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), decode_responses=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=100)
    ap.add_argument("--shard-base-url", default="https://data.atharva.ai/shards")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    for i in range(args.shards):
        job_id = str(uuid.uuid4())
        r.hset(f"job:{job_id}", mapping={
            "shard_url": f"{args.shard_base_url}/shard_{i:05d}.npy",
            "lr": args.lr,
            "batch_size": args.batch_size,
        })
        r.rpush("jobs:pending", job_id)
    print(f"seeded {args.shards} jobs")
