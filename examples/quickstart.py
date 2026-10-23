"""Quickstart on the local filesystem backend: no server, no keys, no network.

python examples/quickstart.py
"""

from __future__ import annotations

import tempfile

from s3_agent_memory import MemoryStore
from s3_agent_memory.consolidate import LongestSummariser


def main() -> None:
    with tempfile.TemporaryDirectory() as root:
        store = MemoryStore.from_url(f"file://{root}")
        ns = "users/alice"

        store.remember(ns, "Alice prefers metric units", kind="procedural", importance=0.9)
        store.remember(ns, "Alice prefers metric units in every answer", kind="procedural")
        store.remember(ns, "Alice is vegetarian", tags=["food"])
        store.remember(ns, "Alice asked for a ramen recipe", kind="episodic", ttl=7 * 86_400)

        print("recall 'what should I cook for Alice?'")
        for hit in store.recall(ns, "what should I cook for Alice? vegetarian recipe"):
            print(f"  {hit.score:.3f}  {hit.memory.kind:<10} {hit.memory.content}")

        report = store.consolidate(ns, summariser=LongestSummariser())
        print("consolidated:", [(m["from"], m["content"]) for m in report.merged])
        print("stats:", store.stats(ns))
        print("objects:")
        for key in store.backend.list(""):
            print("  ", key)


if __name__ == "__main__":
    main()
