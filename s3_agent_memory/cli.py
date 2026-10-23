"""``s3-agent-memory``: remember | recall | get | update | forget | list | consolidate |
prune | reindex | stats | namespaces | export | import | serve.

Every command prints JSON. Exit codes: 0 ok, 1 not found / nothing to do,
2 usage or validation error, 3 conflict or backend error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from typing import Any

from s3_agent_memory import __version__
from s3_agent_memory.errors import AgentMemoryError
from s3_agent_memory.errors import BackendError
from s3_agent_memory.errors import ConflictError
from s3_agent_memory.keys import KINDS
from s3_agent_memory.model import Memory
from s3_agent_memory.store import UNSET
from s3_agent_memory.store import MemoryStore
from s3_agent_memory.vectors import embedder_from_name

DEFAULT_URL = "file://./.agent-memory"


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _memory(store: MemoryStore, memory: Memory) -> dict[str, Any]:
    out = memory.to_dict()
    out["key"] = store.key_of(memory)
    return out


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--url",
        default=None,
        help=f"store URL: s3://bucket/prefix, file://DIR, memory:// "
        f"(env S3_AGENT_MEMORY_URL, default {DEFAULT_URL})",
    )
    common.add_argument(
        "--namespace",
        "-n",
        default=None,
        help="namespace such as users/alice (env S3_AGENT_MEMORY_NAMESPACE, default 'default')",
    )
    common.add_argument(
        "--embeddings",
        choices=["none", "fastembed", "sentence-transformers"],
        default=None,
        help="local embedding model for hybrid search (env S3_AGENT_MEMORY_EMBEDDINGS)",
    )
    common.add_argument(
        "--embedding-model", default=None, help="model name (env S3_AGENT_MEMORY_EMBEDDING_MODEL)"
    )

    parser = argparse.ArgumentParser(
        prog="s3-agent-memory",
        description="Long-term agent memory as plain objects in S3 (or a directory).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("remember", parents=[common], help="store a memory")
    p.add_argument("content")
    p.add_argument("--kind", choices=KINDS, default="semantic")
    p.add_argument("--tag", action="append", default=[], dest="tags")
    p.add_argument("--importance", type=float, default=0.5)
    p.add_argument("--ttl-days", type=float, default=None)
    p.add_argument("--source", default="cli")
    p.add_argument("--id", default=None)

    p = sub.add_parser("recall", parents=[common], help="hybrid search")
    p.add_argument("query")
    p.add_argument("--kind", choices=KINDS, default=None)
    p.add_argument("--tag", action="append", default=None, dest="tags")
    p.add_argument("--limit", type=int, default=5)

    p = sub.add_parser("get", parents=[common], help="one memory by id")
    p.add_argument("id")

    p = sub.add_parser("update", parents=[common], help="change a memory (If-Match on its ETag)")
    p.add_argument("id")
    p.add_argument("--content", default=None)
    p.add_argument("--importance", type=float, default=None)
    p.add_argument("--tag", action="append", default=None, dest="tags")
    p.add_argument("--expected-version", type=int, default=None)

    p = sub.add_parser("forget", parents=[common], help="hard delete + tombstone")
    p.add_argument("id")
    p.add_argument("--reason", default="cli")

    p = sub.add_parser("list", parents=[common], help="recent memories")
    p.add_argument("--kind", choices=KINDS, default=None)
    p.add_argument("--tag", action="append", default=None, dest="tags")
    p.add_argument("--limit", type=int, default=20)

    p = sub.add_parser("consolidate", parents=[common], help="merge near-duplicates")
    p.add_argument("--kind", choices=KINDS, default=None)
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument(
        "--summariser",
        choices=["ollama", "longest"],
        default="ollama",
        help="ollama (local model, default) or longest (no model)",
    )
    p.add_argument("--model", default=None, help="ollama model (env S3_AGENT_MEMORY_OLLAMA_MODEL)")
    p.add_argument("--dry-run", action="store_true")

    sub.add_parser("prune", parents=[common], help="forget expired memories")
    sub.add_parser("reindex", parents=[common], help="rebuild the index from the objects")
    sub.add_parser("stats", parents=[common], help="counts for the namespace")
    p = sub.add_parser("namespaces", parents=[common], help="list namespaces")
    p.add_argument("prefix", nargs="?", default="")

    p = sub.add_parser("export", parents=[common], help="write JSONL")
    p.add_argument("-o", "--output", default="-")
    p.add_argument("--include-expired", action="store_true")

    p = sub.add_parser("import", parents=[common], help="read JSONL into the namespace")
    p.add_argument("input")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resurrect", action="store_true", help="also import tombstoned ids")

    p = sub.add_parser("serve", parents=[common], help="MCP server on stdio")
    p.add_argument("--read-only", action="store_true", help="expose recall and list only")
    return parser


def open_store(args: argparse.Namespace) -> MemoryStore:
    url = args.url or os.environ.get("S3_AGENT_MEMORY_URL") or DEFAULT_URL
    embedder = embedder_from_name(
        args.embeddings or os.environ.get("S3_AGENT_MEMORY_EMBEDDINGS"),
        args.embedding_model or os.environ.get("S3_AGENT_MEMORY_EMBEDDING_MODEL"),
    )
    return MemoryStore.from_url(url, embedder=embedder)


def run(args: argparse.Namespace) -> int:
    store = open_store(args)
    ns = args.namespace or os.environ.get("S3_AGENT_MEMORY_NAMESPACE") or "default"
    cmd = args.command
    if cmd == "remember":
        memory = store.remember(
            ns,
            args.content,
            kind=args.kind,
            tags=args.tags,
            importance=args.importance,
            ttl=None if args.ttl_days is None else args.ttl_days * 86_400,
            source=args.source,
            id=args.id,
        )
        _print(_memory(store, memory))
    elif cmd == "recall":
        hits = store.recall(ns, args.query, kind=args.kind, tags=args.tags, limit=args.limit)
        _print([h.to_dict() for h in hits])
    elif cmd == "get":
        found = store.get(ns, args.id)
        if found is None:
            _print({"error": "not found", "id": args.id})
            return 1
        _print(_memory(store, found))
    elif cmd == "update":
        updated = store.update(
            ns,
            args.id,
            content=UNSET if args.content is None else args.content,
            importance=UNSET if args.importance is None else args.importance,
            tags=UNSET if args.tags is None else args.tags,
            expected_version=args.expected_version,
        )
        _print(_memory(store, updated))
    elif cmd == "forget":
        forgotten = store.forget(ns, args.id, reason=args.reason)
        _print({"id": args.id, "forgotten": forgotten})
        return 0 if forgotten else 1
    elif cmd == "list":
        memories = store.list(ns, kind=args.kind, tags=args.tags, limit=args.limit)
        _print([_memory(store, m) for m in memories])
    elif cmd == "consolidate":
        from s3_agent_memory.consolidate import LongestSummariser
        from s3_agent_memory.consolidate import OllamaSummariser
        from s3_agent_memory.consolidate import Summariser

        summariser: Summariser = (
            LongestSummariser() if args.summariser == "longest" else OllamaSummariser(args.model)
        )
        report = store.consolidate(
            ns,
            kind=args.kind,
            threshold=args.threshold,
            summariser=summariser,
            dry_run=args.dry_run,
        )
        _print(report.as_dict())
    elif cmd == "prune":
        _print({"forgotten": store.prune_expired(ns)})
    elif cmd == "reindex":
        _print(store.reindex(ns).as_dict())
    elif cmd == "stats":
        _print(store.stats(ns))
    elif cmd == "namespaces":
        _print(["/".join(n) for n in store.namespaces(args.prefix or ())])
    elif cmd == "export":
        if args.output == "-":
            count = store.export_jsonl(ns, sys.stdout, include_expired=args.include_expired)
        else:
            with open(args.output, "w", encoding="utf-8") as fh:
                count = store.export_jsonl(ns, fh, include_expired=args.include_expired)
        print(json.dumps({"exported": count}), file=sys.stderr)
    elif cmd == "import":
        if args.input == "-":
            result = store.import_jsonl(
                ns, sys.stdin, overwrite=args.overwrite, resurrect=args.resurrect
            )
        else:
            with open(args.input, encoding="utf-8") as fh:
                result = store.import_jsonl(
                    ns, fh, overwrite=args.overwrite, resurrect=args.resurrect
                )
        _print(result.as_dict())
    elif cmd == "serve":  # pragma: no cover - blocks on stdio
        from s3_agent_memory.server import serve

        serve(store, ns, read_only=args.read_only)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except ConflictError as exc:
        print(json.dumps({"error": "conflict", "detail": str(exc)}), file=sys.stderr)
        return 3
    except BackendError as exc:
        print(json.dumps({"error": "backend", "detail": str(exc)}), file=sys.stderr)
        return 3
    except (AgentMemoryError, ValueError) as exc:
        print(json.dumps({"error": type(exc).__name__, "detail": str(exc)}), file=sys.stderr)
        return 2
