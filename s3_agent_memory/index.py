"""The per-namespace index object (``_index.json``).

The index is a cache of what search needs - term frequencies and ranking
metadata, never the memory text - so recall reads one object instead of every
memory. Memory objects stay the source of truth: :meth:`MemoryStore.reindex`
rebuilds the index from them at any time.

The index is updated incrementally with compare-and-swap (read with ETag,
modify, ``PutObject If-Match``) and retried on conflict, so concurrent writers
never lose each other's entries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import Any

from s3_agent_memory.errors import ValidationError
from s3_agent_memory.keys import Namespace
from s3_agent_memory.keys import validate_id
from s3_agent_memory.keys import validate_kind
from s3_agent_memory.model import Memory
from s3_agent_memory.model import iso
from s3_agent_memory.model import parse_time
from s3_agent_memory.text import term_frequencies

INDEX_SCHEMA = 1


@dataclass
class IndexEntry:
    kind: str
    tf: dict[str, int]
    created_at: datetime
    updated_at: datetime
    importance: float
    tags: tuple[str, ...]
    expires_at: datetime | None
    version: int

    @classmethod
    def of(cls, memory: Memory) -> IndexEntry:
        text = memory.content + (" " + " ".join(memory.tags) if memory.tags else "")
        return cls(
            kind=memory.kind,
            tf=term_frequencies(text),
            created_at=memory.created_at,
            updated_at=memory.updated_at,
            importance=float(memory.importance),
            tags=memory.tags,
            expires_at=memory.expires_at,
            version=memory.version,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "k": self.kind,
            "tf": self.tf,
            "c": iso(self.created_at),
            "u": iso(self.updated_at),
            "i": self.importance,
            "t": list(self.tags),
            "x": iso(self.expires_at) if self.expires_at else None,
            "v": self.version,
        }

    @classmethod
    def from_json(cls, raw: object) -> IndexEntry:
        if not isinstance(raw, dict):
            raise ValidationError("index entry must be an object")
        tf = raw.get("tf")
        if not isinstance(tf, dict) or not all(
            isinstance(k, str) and isinstance(v, int) and v > 0 for k, v in tf.items()
        ):
            raise ValidationError("index entry tf must map terms to positive ints")
        tags = raw.get("t") or []
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            raise ValidationError("index entry tags must be strings")
        importance = raw.get("i", 0.5)
        if not isinstance(importance, int | float) or isinstance(importance, bool):
            raise ValidationError("index entry importance must be a number")
        version = raw.get("v", 1)
        if not isinstance(version, int):
            raise ValidationError("index entry version must be an int")
        return cls(
            kind=validate_kind(raw.get("k")),
            tf=tf,
            created_at=parse_time(raw.get("c"), "c"),
            updated_at=parse_time(raw.get("u"), "u"),
            importance=float(importance),
            tags=tuple(tags),
            expires_at=parse_time(raw["x"], "x") if raw.get("x") else None,
            version=version,
        )

    def is_expired(self, now: datetime) -> bool:
        return self.expires_at is not None and self.expires_at <= now


@dataclass
class NamespaceIndex:
    namespace: Namespace
    docs: dict[str, IndexEntry] = field(default_factory=dict)
    updated_at: datetime | None = None

    def add(self, memory: Memory) -> None:
        self.docs[memory.id] = IndexEntry.of(memory)

    def remove(self, memory_id: str) -> bool:
        return self.docs.pop(memory_id, None) is not None

    def to_bytes(self) -> bytes:
        payload = {
            "schema": INDEX_SCHEMA,
            "namespace": list(self.namespace),
            "updated_at": iso(self.updated_at) if self.updated_at else None,
            "docs": {doc_id: self.docs[doc_id].to_json() for doc_id in sorted(self.docs)},
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()

    @classmethod
    def from_bytes(cls, data: bytes, namespace: Namespace) -> NamespaceIndex:
        try:
            raw = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValidationError("index object is not valid JSON") from exc
        if not isinstance(raw, dict) or raw.get("schema") != INDEX_SCHEMA:
            raise ValidationError("unsupported index schema")
        if tuple(raw.get("namespace") or ()) != namespace:
            raise ValidationError("index object belongs to another namespace")
        docs_raw = raw.get("docs")
        if not isinstance(docs_raw, dict):
            raise ValidationError("index docs must be an object")
        docs = {validate_id(doc_id): IndexEntry.from_json(e) for doc_id, e in docs_raw.items()}
        updated = raw.get("updated_at")
        return cls(namespace, docs, parse_time(updated, "updated_at") if updated else None)
