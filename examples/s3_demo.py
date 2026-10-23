"""Demo against a local S3-compatible server (SeaweedFS or MinIO from docker-compose.yml).

    docker compose up -d
    python examples/s3_demo.py

Nothing here talks to AWS: the endpoint is localhost and the keys are local-only
placeholders from .env.example.
"""

from __future__ import annotations

import os
import threading

import boto3

from s3_agent_memory import ConflictError
from s3_agent_memory import MemoryStore
from s3_agent_memory.backends.s3 import S3Backend

ENDPOINT = os.environ.get("S3_AGENT_MEMORY_ENDPOINT_URL", "http://localhost:8333")
BUCKET = os.environ.get("S3_AGENT_MEMORY_BUCKET", "agent-memory")
NS = "users/alice"


def main() -> None:
    client = boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        region_name="us-east-1",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "local-dev"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "local-dev-only"),
    )
    try:
        client.create_bucket(Bucket=BUCKET)
    except client.exceptions.BucketAlreadyOwnedByYou:
        pass
    except Exception as exc:  # already exists under another error code on some servers
        if "BucketAlready" not in type(exc).__name__ and "BucketAlready" not in str(exc):
            raise
    store = MemoryStore(S3Backend(BUCKET, client=client), prefix="demo")

    # 1. Remember a few typed memories.
    store.remember(NS, "Alice prefers metric units", kind="procedural", importance=0.9)
    store.remember(NS, "Alice lives in Lisbon and works on data pipelines")
    store.remember(NS, "Alice asked about the Q3 roadmap on Monday", kind="episodic")

    # 2. Eight writers at once: conditional writes on the index keep every entry.
    threads = [
        threading.Thread(target=store.remember, args=(NS, f"parallel note {i} about topic{i}"))
        for i in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("memories after 8 concurrent writers:", store.stats(NS)["memories"])

    # 3. Optimistic concurrency on one memory: the second writer with a stale version loses.
    m = store.remember(NS, "Alice's favourite editor is vim")
    store.update(NS, m.id, content="Alice's favourite editor is helix", expected_version=1)
    try:
        store.update(NS, m.id, content="Alice's favourite editor is emacs", expected_version=1)
    except ConflictError as exc:
        print("stale update refused:", exc)

    # 4. Recall with provenance.
    for hit in store.recall(NS, "what units does Alice use?", limit=3):
        print(f"{hit.score:.3f}  {hit.memory.kind:<10}  {hit.memory.content}  <- {hit.key}")

    # 5. Forget leaves a tombstone without content.
    store.forget(NS, m.id, reason="demo")
    print("tombstones:", [t["reason"] for t in store.tombstones(NS)])


if __name__ == "__main__":
    main()
