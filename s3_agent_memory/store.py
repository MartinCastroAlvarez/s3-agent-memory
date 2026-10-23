""":class:`MemoryStore`: remember, recall, update, forget, consolidate, export, import.

Writes follow one rule: every object that more than one writer can touch is
changed with compare-and-swap on its ETag (``If-None-Match: *`` to create,
``If-Match`` to replace) and retried with jittered backoff when another writer
won. Memory objects are the source of truth; the index and vector shard are
caches that :meth:`MemoryStore.reindex` can rebuild.
"""

from __future__ import annotations

import builtins
import json
import logging
import secrets
import time
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from datetime import datetime
from datetime import timedelta
from enum import Enum
from typing import IO
from typing import Any
from typing import TypeVar

from s3_agent_memory.backends import ObjectStore
from s3_agent_memory.backends import open_backend
from s3_agent_memory.errors import ConflictError
from s3_agent_memory.errors import InvalidKeyError
from s3_agent_memory.errors import LimitExceededError
from s3_agent_memory.errors import NotFoundError
from s3_agent_memory.errors import ValidationError
from s3_agent_memory.index import NamespaceIndex
from s3_agent_memory.keys import KINDS
from s3_agent_memory.keys import Kind
from s3_agent_memory.keys import Namespace
from s3_agent_memory.keys import index_key
from s3_agent_memory.keys import kind_dir
from s3_agent_memory.keys import memory_key
from s3_agent_memory.keys import namespace_dir
from s3_agent_memory.keys import namespace_from_index_key
from s3_agent_memory.keys import tombstone_key
from s3_agent_memory.keys import tombstones_dir
from s3_agent_memory.keys import validate_id
from s3_agent_memory.keys import validate_kind
from s3_agent_memory.keys import validate_namespace
from s3_agent_memory.keys import validate_prefix
from s3_agent_memory.keys import vectors_key
from s3_agent_memory.limits import Limits
from s3_agent_memory.model import Memory
from s3_agent_memory.model import RecallHit
from s3_agent_memory.model import iso
from s3_agent_memory.model import new_id
from s3_agent_memory.model import utcnow
from s3_agent_memory.ranking import Ranking
from s3_agent_memory.text import BM25
from s3_agent_memory.text import tokenize
from s3_agent_memory.vectors import Embedder
from s3_agent_memory.vectors import VectorShard

log = logging.getLogger(__name__)

T = TypeVar("T")
S = TypeVar("S")
_rng = secrets.SystemRandom()


class _Unset(Enum):
    UNSET = "unset"


UNSET = _Unset.UNSET
"""Sentinel for :meth:`MemoryStore.update`: leave the field as it is."""

NamespaceLike = str | Sequence[str]


@dataclass
class ImportReport:
    imported: int = 0
    skipped_existing: int = 0
    skipped_tombstoned: int = 0
    errors: list[tuple[int, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "imported": self.imported,
            "skipped_existing": self.skipped_existing,
            "skipped_tombstoned": self.skipped_tombstoned,
            "errors": [{"line": n, "error": e} for n, e in self.errors],
        }


@dataclass
class ReindexReport:
    indexed: int = 0
    invalid: int = 0
    vectors: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"indexed": self.indexed, "invalid": self.invalid, "vectors": self.vectors}


def _expiry(
    now: datetime, ttl: float | timedelta | None, expires_at: datetime | None
) -> datetime | None:
    if ttl is not None and expires_at is not None:
        raise ValidationError("pass ttl or expires_at, not both")
    if ttl is None:
        return expires_at
    seconds = ttl.total_seconds() if isinstance(ttl, timedelta) else float(ttl)
    if seconds <= 0:
        raise ValidationError("ttl must be positive")
    return now + timedelta(seconds=seconds)


class MemoryStore:
    """Long-term memory on an :class:`~s3_agent_memory.backends.ObjectStore`.

    Every method takes a ``namespace`` (``"alice"``, ``"users/alice"`` or
    ``("users", "alice")``) and only ever touches keys under it.
    """

    def __init__(
        self,
        backend: ObjectStore,
        *,
        prefix: str = "",
        embedder: Embedder | None = None,
        limits: Limits | None = None,
        ranking: Ranking | None = None,
        max_retries: int = 12,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self.backend = backend
        self.prefix = validate_prefix(prefix)
        self.embedder = embedder
        self.limits = limits or Limits()
        self.ranking = ranking or Ranking()
        self.max_retries = max_retries
        self.clock = clock

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> MemoryStore:
        """``s3://bucket/prefix``, ``file:///dir`` or ``memory://`` (see :func:`open_backend`)."""
        backend, prefix = open_backend(url)
        kwargs.setdefault("prefix", prefix)
        return cls(backend, **kwargs)

    def __repr__(self) -> str:
        return f"MemoryStore({self.backend!r}, prefix={self.prefix!r})"

    # ------------------------------------------------------------------ #
    # Compare-and-swap plumbing
    # ------------------------------------------------------------------ #
    def _backoff(self, attempt: int) -> None:
        time.sleep(min(0.25, 0.002 * 2**attempt) * _rng.uniform(0.5, 1.5))

    def _cas(
        self,
        key: str,
        load: Callable[[bytes | None], S],
        dump: Callable[[S], bytes],
        fn: Callable[[S], tuple[T, bool]],
        *,
        max_bytes: int,
        content_type: str = "application/json",
    ) -> T:
        """Read-modify-write ``key`` until our conditional write wins.

        ``fn`` mutates the loaded state and returns ``(result, changed)``; it is
        re-run on fresh state after every lost race, so it must not have side
        effects outside the state.
        """
        for attempt in range(self.max_retries):
            current = self.backend.get(key, max_bytes=max_bytes)
            state = load(current.data if current else None)
            result, changed = fn(state)
            if not changed:
                return result
            try:
                self.backend.put(
                    key,
                    dump(state),
                    if_match=current.etag if current else None,
                    if_none_match=current is None,
                    content_type=content_type,
                )
                return result
            except ConflictError:
                self._backoff(attempt)
        raise ConflictError(f"{key}: gave up after {self.max_retries} conflicting writes")

    def _mutate_index(self, ns: Namespace, fn: Callable[[NamespaceIndex], tuple[T, bool]]) -> T:
        def load(data: bytes | None) -> NamespaceIndex:
            return NamespaceIndex(ns) if data is None else NamespaceIndex.from_bytes(data, ns)

        def dump(idx: NamespaceIndex) -> bytes:
            idx.updated_at = self.clock()
            return idx.to_bytes()

        return self._cas(
            index_key(self.prefix, ns), load, dump, fn, max_bytes=self.limits.max_index_bytes
        )

    def _read_index(self, ns: Namespace) -> NamespaceIndex:
        obj = self.backend.get(index_key(self.prefix, ns), max_bytes=self.limits.max_index_bytes)
        return NamespaceIndex(ns) if obj is None else NamespaceIndex.from_bytes(obj.data, ns)

    def _read_vectors(self, ns: Namespace) -> VectorShard | None:
        if self.embedder is None:
            return None
        obj = self.backend.get(vectors_key(self.prefix, ns), max_bytes=self.limits.max_index_bytes)
        if obj is None:
            return None
        shard = VectorShard.from_bytes(obj.data, max_bytes=self.limits.max_index_bytes)
        return shard if shard.model == self.embedder.name else None

    def _mutate_vectors(self, ns: Namespace, fn: Callable[[VectorShard], bool]) -> None:
        embedder = self.embedder
        if embedder is None:
            return

        def load(data: bytes | None) -> VectorShard:
            if data is None:
                return VectorShard.empty(embedder.name)
            shard = VectorShard.from_bytes(data, max_bytes=self.limits.max_index_bytes)
            # A shard from another model is useless for this one: start over.
            return shard if shard.model == embedder.name else VectorShard.empty(embedder.name)

        self._cas(
            vectors_key(self.prefix, ns),
            load,
            lambda shard: shard.to_bytes(),
            lambda shard: (None, fn(shard)),
            max_bytes=self.limits.max_index_bytes,
            content_type="application/octet-stream",
        )

    def _embed(self, ns: Namespace, memories: Sequence[Memory]) -> int:
        """Embed and store vectors; a failing local model never loses the memory itself."""
        if self.embedder is None or not memories:
            return 0
        try:
            vectors = list(self.embedder.embed([m.content for m in memories]))
        except Exception as exc:
            log.warning(
                "embedding failed (%s); memories stay keyword-searchable", type(exc).__name__
            )
            return 0
        if len(vectors) != len(memories):
            raise ValidationError("embedder returned a different number of vectors")

        def apply(shard: VectorShard) -> bool:
            for memory, vector in zip(memories, vectors, strict=True):
                shard.set(memory.id, vector)
            return True

        self._mutate_vectors(ns, apply)
        return len(memories)

    def _unembed(self, ns: Namespace, ids: Iterable[str]) -> None:
        wanted = list(ids)
        self._mutate_vectors(ns, lambda shard: any([shard.remove(i) for i in wanted]))

    def _index_put(self, ns: Namespace, memories: Sequence[Memory], *, new: bool) -> None:
        def apply(idx: NamespaceIndex) -> tuple[None, bool]:
            changed = False
            for memory in memories:
                existing = idx.docs.get(memory.id)
                if new and existing is not None:
                    raise ConflictError(f"memory id {memory.id!r} already exists in this namespace")
                if existing is not None and existing.version > memory.version:
                    continue  # a newer version already landed: keep it
                if existing is None and len(idx.docs) >= self.limits.max_memories:
                    raise LimitExceededError(
                        f"namespace holds {self.limits.max_memories} memories (Limits.max_memories)"
                    )
                idx.add(memory)
                changed = True
            return None, changed

        self._mutate_index(ns, apply)

    def _index_remove(self, ns: Namespace, ids: Iterable[str]) -> None:
        wanted = list(ids)
        self._mutate_index(ns, lambda idx: (None, any([idx.remove(i) for i in wanted])))

    # ------------------------------------------------------------------ #
    # Reading objects
    # ------------------------------------------------------------------ #
    def _ns(self, namespace: NamespaceLike) -> Namespace:
        return validate_namespace(namespace)

    def _load(self, ns: Namespace, kind: str, memory_id: str) -> Memory | None:
        key = memory_key(self.prefix, ns, kind, memory_id)
        obj = self.backend.get(key, max_bytes=self.limits.max_object_bytes)
        if obj is None:
            return None
        memory = Memory.from_bytes(obj.data, self.limits, etag=obj.etag)
        if memory.namespace != ns or memory.id != memory_id or memory.kind != kind:
            # Someone wrote a memory under a key it does not belong to: never serve it here.
            raise ValidationError(f"{key}: object does not match its key")
        return memory

    def key_of(self, memory: Memory) -> str:
        """The object key a memory lives at (its provenance)."""
        return memory_key(self.prefix, memory.namespace, memory.kind, memory.id)

    def get(
        self, namespace: NamespaceLike, memory_id: str, *, include_expired: bool = False
    ) -> Memory | None:
        """One memory by id, or ``None`` (also for expired memories unless asked)."""
        ns = self._ns(namespace)
        validate_id(memory_id)
        hint = self._read_index(ns).docs.get(memory_id)
        kinds = [hint.kind, *(k for k in KINDS if k != hint.kind)] if hint else list(KINDS)
        for kind in kinds:
            memory = self._load(ns, kind, memory_id)
            if memory is not None:
                if memory.is_expired(self.clock()) and not include_expired:
                    return None
                return memory
        return None

    def iter_memories(
        self, namespace: NamespaceLike, *, kind: str | None = None
    ) -> Iterator[Memory]:
        """Every memory object in the namespace (straight from the bucket, not the index)."""
        ns = self._ns(namespace)
        for k in [validate_kind(kind)] if kind else KINDS:
            for key in self.backend.list(kind_dir(self.prefix, ns, k), recursive=False):
                name = key.rsplit("/", 1)[-1]
                if not name.endswith(".json"):
                    continue
                try:
                    memory = self._load(ns, k, validate_id(name[: -len(".json")]))
                except (InvalidKeyError, ValidationError, LimitExceededError) as exc:
                    log.warning("skipping unreadable memory object (%s)", type(exc).__name__)
                    continue
                if memory is not None:
                    yield memory

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #
    def remember(
        self,
        namespace: NamespaceLike,
        content: str,
        *,
        kind: Kind | str = "semantic",
        id: str | None = None,
        source: str | None = None,
        importance: float = 0.5,
        tags: Iterable[str] = (),
        ttl: float | timedelta | None = None,
        expires_at: datetime | None = None,
        data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Store a new memory (``ttl`` in seconds).

        Raises :class:`ConflictError` if ``id`` already exists in the namespace.
        """
        ns = self._ns(namespace)
        now = self.clock()
        memory = Memory(
            id=validate_id(id) if id is not None else new_id(),
            namespace=ns,
            kind=validate_kind(kind),
            content=content,
            created_at=now,
            updated_at=now,
            source=source,
            importance=importance,
            tags=tuple(dict.fromkeys(tags)),
            expires_at=_expiry(now, ttl, expires_at),
            data=data,
            metadata=dict(metadata or {}),
        ).validated(self.limits)
        # Reserve the id in the index first: ids are unique per namespace across kinds.
        self._index_put(ns, [memory], new=True)
        try:
            etag = self.backend.put(self.key_of(memory), memory.to_bytes(), if_none_match=True)
        except BaseException:
            self._index_remove(ns, [memory.id])
            raise
        self._embed(ns, [memory])
        return memory.with_etag(etag)

    def update(
        self,
        namespace: NamespaceLike,
        memory_id: str,
        *,
        content: str | _Unset = UNSET,
        importance: float | _Unset = UNSET,
        tags: Iterable[str] | _Unset = UNSET,
        source: str | None | _Unset = UNSET,
        data: dict[str, Any] | None | _Unset = UNSET,
        metadata: dict[str, Any] | _Unset = UNSET,
        ttl: float | timedelta | None | _Unset = UNSET,
        expected_version: int | None = None,
    ) -> Memory:
        """Change fields of a memory with ``If-Match`` on its ETag.

        Without ``expected_version`` a lost race re-reads and re-applies the
        change (last writer wins per field). With it, a concurrent change raises
        :class:`ConflictError` instead - optimistic concurrency for callers that
        computed the change from what they read.
        """
        ns = self._ns(namespace)
        validate_id(memory_id)
        for attempt in range(self.max_retries):
            current = self.get(ns, memory_id, include_expired=True)
            if current is None:
                raise NotFoundError(f"no memory {memory_id!r} in {'/'.join(ns)}")
            if expected_version is not None and current.version != expected_version:
                raise ConflictError(
                    f"memory {memory_id!r} is at version {current.version}, not {expected_version}"
                )
            now = self.clock()
            changes: dict[str, Any] = {"updated_at": now, "version": current.version + 1}
            if not isinstance(content, _Unset):
                changes["content"] = content
            if not isinstance(importance, _Unset):
                changes["importance"] = importance
            if not isinstance(tags, _Unset):
                changes["tags"] = tuple(dict.fromkeys(tags))
            if not isinstance(source, _Unset):
                changes["source"] = source
            if not isinstance(data, _Unset):
                changes["data"] = data
            if not isinstance(metadata, _Unset):
                changes["metadata"] = dict(metadata)
            if not isinstance(ttl, _Unset):
                changes["expires_at"] = None if ttl is None else _expiry(now, ttl, None)
            updated = replace(current, **changes).validated(self.limits)
            try:
                etag = self.backend.put(
                    self.key_of(updated), updated.to_bytes(), if_match=current.etag
                )
            except ConflictError:
                if expected_version is not None:
                    raise
                self._backoff(attempt)
                continue
            self._index_put(ns, [updated], new=False)
            if updated.content != current.content:
                self._embed(ns, [updated])
            return updated.with_etag(etag)
        raise ConflictError(f"memory {memory_id!r}: gave up after {self.max_retries} conflicts")

    def forget(
        self, namespace: NamespaceLike, memory_id: str, *, reason: str = "forgotten"
    ) -> bool:
        """Hard-delete a memory and leave a tombstone (id, kind, time, reason - no content).

        Tombstones stop :meth:`import_jsonl` from resurrecting forgotten memories.
        Returns ``False`` when there was nothing to forget.
        """
        ns = self._ns(namespace)
        validate_id(memory_id)
        if not isinstance(reason, str) or len(reason) > 256:
            raise ValidationError("reason must be a string of at most 256 characters")
        memory = self.get(ns, memory_id, include_expired=True)
        if memory is None:
            self._index_remove(ns, [memory_id])
            self._unembed(ns, [memory_id])
            return False
        tombstone = {
            "schema": 1,
            "id": memory.id,
            "kind": memory.kind,
            "namespace": list(ns),
            "deleted_at": iso(self.clock()),
            "reason": reason,
            "version": memory.version,
        }
        self.backend.put(
            tombstone_key(self.prefix, ns, memory.id),
            json.dumps(tombstone, separators=(",", ":")).encode(),
        )
        self.backend.delete(self.key_of(memory))
        self._index_remove(ns, [memory.id])
        self._unembed(ns, [memory.id])
        return True

    def tombstones(self, namespace: NamespaceLike) -> list[dict[str, Any]]:
        ns = self._ns(namespace)
        out = []
        for key in self.backend.list(tombstones_dir(self.prefix, ns), recursive=False):
            if not key.endswith(".json"):
                continue
            obj = self.backend.get(key, max_bytes=4096)
            if obj is None:
                continue
            try:
                raw = json.loads(obj.data)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(raw, dict) and raw.get("namespace") == list(ns):
                out.append(raw)
        return out

    def _tombstoned_ids(self, ns: Namespace) -> set[str]:
        ids = set()
        for key in self.backend.list(tombstones_dir(self.prefix, ns), recursive=False):
            name = key.rsplit("/", 1)[-1]
            if name.endswith(".json"):
                ids.add(name[: -len(".json")])
        return ids

    def prune_expired(self, namespace: NamespaceLike) -> list[str]:
        """Forget every expired memory (reason ``expired``); returns their ids."""
        ns = self._ns(namespace)
        now = self.clock()
        expired = [i for i, e in self._read_index(ns).docs.items() if e.is_expired(now)]
        return [i for i in expired if self.forget(ns, i, reason="expired")]

    # ------------------------------------------------------------------ #
    # Reading many
    # ------------------------------------------------------------------ #
    def _eligible(
        self,
        idx: NamespaceIndex,
        kind: str | None,
        tags: Iterable[str] | None,
        include_expired: bool,
    ) -> dict[str, Any]:
        now = self.clock()
        if kind is not None:
            validate_kind(kind)
        wanted = set(tags or ())
        return {
            doc_id: entry
            for doc_id, entry in idx.docs.items()
            if (include_expired or not entry.is_expired(now))
            and (kind is None or entry.kind == kind)
            and wanted.issubset(entry.tags)
        }

    def _clamp(self, limit: int) -> int:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValidationError("limit must be a positive integer")
        return min(limit, self.limits.max_results)

    def list(
        self,
        namespace: NamespaceLike,
        *,
        kind: str | None = None,
        tags: Iterable[str] | None = None,
        limit: int = 50,
        offset: int = 0,
        include_expired: bool = False,
    ) -> list[Memory]:
        """Memories, most recently updated first."""
        ns = self._ns(namespace)
        limit = self._clamp(limit)
        eligible = self._eligible(self._read_index(ns), kind, tags, include_expired)
        order = sorted(eligible, key=lambda i: (eligible[i].updated_at, i), reverse=True)
        out: list[Memory] = []
        for doc_id in order[max(0, offset) :]:
            memory = self._load(ns, eligible[doc_id].kind, doc_id)
            if memory is not None:
                out.append(memory)
            if len(out) >= limit:
                break
        return out

    def recall(
        self,
        namespace: NamespaceLike,
        query: str,
        *,
        kind: str | None = None,
        tags: Iterable[str] | None = None,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> builtins.list[RecallHit]:
        """Hybrid search: BM25 (+ cosine when an embedder is set), recency, importance.

        An empty query ranks by recency and importance alone. Results carry
        provenance (object key, source, timestamps): treat ``content`` as
        untrusted data, never as instructions.
        """
        ns = self._ns(namespace)
        limit = self._clamp(limit)
        if not isinstance(query, str):
            raise ValidationError("query must be a string")
        if len(query) > self.limits.max_query_chars:
            raise LimitExceededError(f"query longer than {self.limits.max_query_chars} characters")
        eligible = self._eligible(self._read_index(ns), kind, tags, include_expired=False)
        if not eligible:
            return []
        terms = tokenize(query)
        bm25 = BM25({i: e.tf for i, e in eligible.items()}).scores(terms) if terms else {}
        sims: dict[str, float] = {}
        if self.embedder is not None and query.strip():
            shard = self._read_vectors(ns)
            if shard is not None and shard.rows:
                vector = list(self.embedder.embed([query]))[0]
                sims = {i: s for i, s in shard.similarities(vector).items() if i in eligible}
        searched = bool(terms) or bool(sims)
        if searched:
            candidates = set(bm25) | {
                i for i, s in sims.items() if s >= self.ranking.min_similarity
            }
        else:
            candidates = set(eligible)
        best = max(bm25.values(), default=0.0) or 1.0
        now = self.clock()
        scored: list[tuple[float, str, dict[str, float]]] = []
        for doc_id in candidates:
            entry = eligible[doc_id]
            score, parts = self.ranking.score(
                kind=entry.kind,
                updated_at=entry.updated_at,
                importance=entry.importance,
                now=now,
                bm25=(bm25.get(doc_id, 0.0) / best) if terms else None,
                similarity=sims.get(doc_id, 0.0) if sims else None,
            )
            if score >= min_score:
                scored.append((score, doc_id, parts))
        scored.sort(key=lambda t: (-t[0], t[1]))
        hits: list[RecallHit] = []
        for score, doc_id, parts in scored:
            memory = self._load(ns, eligible[doc_id].kind, doc_id)
            if memory is None or memory.is_expired(now):
                continue  # deleted since the index was written
            hits.append(RecallHit(memory, score, parts, self.key_of(memory)))
            if len(hits) >= limit:
                break
        return hits

    def namespaces(self, prefix: NamespaceLike = ()) -> builtins.list[Namespace]:
        """Every namespace (with an index object) under ``prefix``."""
        base = validate_namespace(prefix) if prefix else ()
        list_prefix = (
            namespace_dir(self.prefix, base) if base else (self.prefix + "/" if self.prefix else "")
        )
        out = []
        for key in self.backend.list(list_prefix, recursive=True):
            ns = namespace_from_index_key(self.prefix, key)
            if ns is not None and ns[: len(base)] == base:
                out.append(ns)
        return sorted(set(out))

    def stats(self, namespace: NamespaceLike) -> dict[str, Any]:
        ns = self._ns(namespace)
        idx = self._read_index(ns)
        now = self.clock()
        by_kind: dict[str, int] = {k: 0 for k in KINDS}
        expired = 0
        for entry in idx.docs.values():
            by_kind[entry.kind] += 1
            expired += entry.is_expired(now)
        shard = self._read_vectors(ns)
        return {
            "namespace": "/".join(ns),
            "memories": len(idx.docs),
            "by_kind": by_kind,
            "expired": expired,
            "vectors": len(shard.rows) if shard else 0,
            "embedder": self.embedder.name if self.embedder else None,
            "tombstones": len(self._tombstoned_ids(ns)),
        }

    # ------------------------------------------------------------------ #
    # Maintenance
    # ------------------------------------------------------------------ #
    def reindex(self, namespace: NamespaceLike) -> ReindexReport:
        """Rebuild the index (and vector shard) from the memory objects themselves."""
        ns = self._ns(namespace)
        report = ReindexReport()
        memories = list(self.iter_memories(ns))
        report.indexed = len(memories)

        def rebuild(idx: NamespaceIndex) -> tuple[None, bool]:
            idx.docs.clear()
            for memory in memories:
                idx.add(memory)
            return None, True

        self._mutate_index(ns, rebuild)
        if self.embedder is not None:
            keep = {m.id for m in memories}
            self._mutate_vectors(
                ns, lambda shard: any([shard.remove(i) for i in list(shard.rows) if i not in keep])
            )
            report.vectors = self._embed(ns, memories)
        return report

    def consolidate(self, namespace: NamespaceLike, **kwargs: Any) -> Any:
        """Merge near-duplicate memories; see :func:`s3_agent_memory.consolidate.consolidate`."""
        from s3_agent_memory.consolidate import consolidate

        return consolidate(self, namespace, **kwargs)

    # ------------------------------------------------------------------ #
    # JSONL
    # ------------------------------------------------------------------ #
    def export_jsonl(
        self, namespace: NamespaceLike, fp: IO[str], *, include_expired: bool = False
    ) -> int:
        """Write one JSON memory per line (straight from the objects). Returns the count."""
        count = 0
        now = self.clock()
        memories = sorted(self.iter_memories(namespace), key=lambda m: (m.created_at, m.id))
        for memory in memories:
            if memory.is_expired(now) and not include_expired:
                continue
            fp.write(json.dumps(memory.to_dict(), ensure_ascii=False, separators=(",", ":")))
            fp.write("\n")
            count += 1
        return count

    def import_jsonl(
        self,
        namespace: NamespaceLike,
        fp: IO[str],
        *,
        overwrite: bool = False,
        resurrect: bool = False,
    ) -> ImportReport:
        """Load memories exported by :meth:`export_jsonl` into ``namespace``.

        The namespace in each line is ignored: everything lands in the target
        namespace, so an import file cannot write into somebody else's. Ids that
        were forgotten (tombstoned) are skipped unless ``resurrect=True``; ids
        that exist are skipped unless ``overwrite=True``.
        """
        ns = self._ns(namespace)
        report = ImportReport()
        tombstoned = self._tombstoned_ids(ns)
        existing = set(self._read_index(ns).docs)
        imported: list[Memory] = []
        line_no = 0
        while True:
            line = fp.readline(self.limits.max_jsonl_line_bytes + 1)
            if not line:
                break
            line_no += 1
            if len(line.encode()) > self.limits.max_jsonl_line_bytes and not line.endswith("\n"):
                report.errors.append((line_no, "line too long"))
                while line and not line.endswith("\n"):  # skip the rest of it
                    line = fp.readline(self.limits.max_jsonl_line_bytes + 1)
                continue
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValidationError("line is not a JSON object")
                raw["namespace"] = list(ns)
                memory = Memory.from_dict(raw, self.limits)
            except (
                json.JSONDecodeError,
                ValidationError,
                InvalidKeyError,
                LimitExceededError,
            ) as exc:
                report.errors.append((line_no, str(exc)[:200]))
                continue
            if memory.id in tombstoned and not resurrect:
                report.skipped_tombstoned += 1
                continue
            if memory.id in existing and not overwrite:
                report.skipped_existing += 1
                continue
            if memory.id not in existing and len(existing) >= self.limits.max_memories:
                report.errors.append((line_no, "namespace is full (Limits.max_memories)"))
                continue
            if memory.id in existing:
                old = self.get(ns, memory.id, include_expired=True)
                if old is not None and old.kind != memory.kind:
                    self.backend.delete(self.key_of(old))
                memory = replace(
                    memory, version=max(memory.version, (old.version + 1) if old else 1)
                )
            self.backend.put(self.key_of(memory), memory.to_bytes())
            if memory.id in tombstoned:
                self.backend.delete(tombstone_key(self.prefix, ns, memory.id))
            existing.add(memory.id)
            imported.append(memory)
        if imported:
            self._index_put(ns, imported, new=False)
            self._embed(ns, imported)
        report.imported = len(imported)
        return report
