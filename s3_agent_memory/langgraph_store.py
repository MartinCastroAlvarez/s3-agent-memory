"""LangGraph ``BaseStore`` on S3 (``pip install "s3-agent-memory[langgraph]"``).

Use it wherever LangGraph takes a store::

    from s3_agent_memory import MemoryStore
    from s3_agent_memory.langgraph_store import S3Store

    store = S3Store(MemoryStore.from_url("s3://my-bucket/agents"))
    graph = builder.compile(store=store)

Mapping: a LangGraph namespace is a memory namespace, an item is a memory
whose ``data`` is the item's ``value``, and the searchable text is the fields
named by ``index`` (or every string in the value). Keys that are not valid
memory ids are stored under a SHA-256-derived id; the original key is kept in
the memory's metadata. ``search(query=...)`` uses the same hybrid ranking as
:meth:`MemoryStore.recall`; ``filter`` supports equality and ``$eq``, ``$ne``,
``$gt``, ``$gte``, ``$lt``, ``$lte`` like LangGraph's in-memory store.
TTLs are in minutes, as in LangGraph.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Iterable
from collections.abc import Mapping
from typing import Any

from langgraph.store.base import NOT_PROVIDED
from langgraph.store.base import BaseStore
from langgraph.store.base import GetOp
from langgraph.store.base import Item
from langgraph.store.base import ListNamespacesOp
from langgraph.store.base import MatchCondition
from langgraph.store.base import Op
from langgraph.store.base import PutOp
from langgraph.store.base import Result
from langgraph.store.base import SearchItem
from langgraph.store.base import SearchOp
from langgraph.store.base import TTLConfig
from langgraph.store.base import get_text_at_path
from langgraph.store.base import tokenize_path

from s3_agent_memory.errors import ConflictError
from s3_agent_memory.errors import InvalidKeyError
from s3_agent_memory.keys import Kind
from s3_agent_memory.keys import Namespace
from s3_agent_memory.keys import validate_id
from s3_agent_memory.keys import validate_namespace
from s3_agent_memory.model import Memory
from s3_agent_memory.store import MemoryStore

_KEY_FIELD = "langgraph_key"
_TTL_FIELD = "langgraph_ttl_minutes"


def memory_id_for(key: str) -> str:
    """The memory id a LangGraph key is stored under."""
    try:
        return validate_id(key)
    except InvalidKeyError:
        return "k" + hashlib.sha256(key.encode()).hexdigest()[:40]


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list | tuple):
        for v in value:
            yield from _strings(v)


def _compare(item_value: Any, filter_value: Any) -> bool:
    if isinstance(filter_value, Mapping):
        if any(str(k).startswith("$") for k in filter_value):
            return all(_operator(item_value, op, v) for op, v in filter_value.items())
        return isinstance(item_value, Mapping) and all(
            _compare(item_value.get(k), v) for k, v in filter_value.items()
        )
    return bool(item_value == filter_value)


_ORDER_OPS = {
    "$gt": lambda a, b: a > b,
    "$gte": lambda a, b: a >= b,
    "$lt": lambda a, b: a < b,
    "$lte": lambda a, b: a <= b,
}


def _operator(value: Any, op: str, target: Any) -> bool:
    if op == "$eq":
        return bool(value == target)
    if op == "$ne":
        return bool(value != target)
    compare = _ORDER_OPS.get(op)
    if compare is None:
        raise ValueError(f"unsupported filter operator {op!r}")
    try:
        return bool(compare(float(value), float(target)))
    except (TypeError, ValueError):
        return False


def _matches(condition: MatchCondition, namespace: Namespace) -> bool:
    path = condition.path
    if len(namespace) < len(path):
        return False
    pairs = (
        zip(namespace, path, strict=False)
        if condition.match_type == "prefix"
        else zip(reversed(namespace), reversed(path), strict=False)
    )
    return all(p == "*" or n == p for n, p in pairs)


class S3Store(BaseStore):
    """A LangGraph store whose items are memory objects in S3 (or any backend)."""

    supports_ttl = True

    def __init__(
        self,
        memory: MemoryStore,
        *,
        kind: Kind = "semantic",
        index: list[str] | None = None,
        ttl: TTLConfig | None = None,
    ) -> None:
        self.memory = memory
        self.kind: Kind = kind
        self.index_fields = index
        self.ttl_config = ttl

    # -- helpers ------------------------------------------------------------ #
    def _text(self, value: Mapping[str, Any], index: Any) -> str:
        if index is False:
            return ""
        fields = index if isinstance(index, list) else self.index_fields
        if fields:
            parts: list[str] = []
            for path in fields:
                parts.extend(
                    t if isinstance(t, str) else json.dumps(t, ensure_ascii=False)
                    for t in get_text_at_path(dict(value), tokenize_path(path))
                )
        else:
            parts = list(_strings(value))
        text = "\n".join(p for p in parts if p).replace("\x00", "")
        limit = self.memory.limits.max_content_bytes
        return text.encode()[:limit].decode(errors="ignore")

    def _item(self, memory: Memory) -> Item:
        return Item(
            namespace=memory.namespace,
            key=str(memory.metadata.get(_KEY_FIELD, memory.id)),
            value=dict(memory.data or {}),
            created_at=memory.created_at,
            updated_at=memory.updated_at,
        )

    def _load(self, namespace: tuple[str, ...], key: str) -> Memory | None:
        memory = self.memory.get(validate_namespace(namespace), memory_id_for(key))
        if memory is None or memory.metadata.get(_KEY_FIELD, memory.id) != key:
            return None
        return memory

    def _ttl_minutes(self, ttl: Any) -> float | None:
        if ttl is NOT_PROVIDED:
            return (self.ttl_config or {}).get("default_ttl")
        return None if ttl is None else float(ttl)

    def _refresh(self, memory: Memory, refresh: bool | None) -> None:
        minutes = memory.metadata.get(_TTL_FIELD)
        if refresh and minutes:
            self.memory.update(memory.namespace, memory.id, ttl=float(minutes) * 60)

    # -- ops ---------------------------------------------------------------- #
    def _get(self, op: GetOp) -> Item | None:
        memory = self._load(op.namespace, op.key)
        if memory is None:
            return None
        refresh = op.refresh_ttl
        if refresh is None:
            refresh = (self.ttl_config or {}).get("refresh_on_read", True)
        self._refresh(memory, refresh)
        return self._item(memory)

    def _put(self, op: PutOp) -> None:
        ns = validate_namespace(op.namespace)
        memory_id = memory_id_for(op.key)
        if op.value is None:
            existing = self._load(ns, op.key)
            if existing is not None:
                self.memory.forget(ns, memory_id, reason="langgraph-delete")
            return
        value = dict(op.value)
        content = self._text(value, op.index)
        minutes = self._ttl_minutes(op.ttl)
        metadata: dict[str, Any] = {_KEY_FIELD: op.key}
        if minutes:
            metadata[_TTL_FIELD] = minutes
        ttl_seconds = minutes * 60 if minutes else None
        for _ in range(3):
            if self.memory.get(ns, memory_id, include_expired=True) is not None:
                self.memory.update(
                    ns, memory_id, content=content, data=value, metadata=metadata, ttl=ttl_seconds
                )
                return
            try:
                self.memory.remember(
                    ns,
                    content,
                    kind=self.kind,
                    id=memory_id,
                    data=value,
                    metadata=metadata,
                    ttl=ttl_seconds,
                    source="langgraph",
                )
                return
            except ConflictError:
                continue  # created concurrently: update it instead
        raise ConflictError(f"could not write {op.key!r}")

    def _search(self, op: SearchOp) -> list[SearchItem]:
        prefix = tuple(op.namespace_prefix)
        namespaces = self.memory.namespaces(prefix) if prefix else self.memory.namespaces()
        width = self.memory.limits.max_results
        found: list[tuple[float, Any, SearchItem]] = []
        for ns in namespaces:
            if op.query:
                for hit in self.memory.recall(ns, op.query, limit=width):
                    item = self._search_item(hit.memory, hit.score)
                    found.append((hit.score, hit.memory.updated_at, item))
            else:
                for memory in self.memory.list(ns, limit=width):
                    found.append((0.0, memory.updated_at, self._search_item(memory, None)))
        if op.filter:
            found = [
                f
                for f in found
                if all(_compare(f[2].value.get(k), v) for k, v in op.filter.items())
            ]
        found.sort(key=lambda f: (f[0], f[1]), reverse=True)
        return [f[2] for f in found[op.offset : op.offset + op.limit]]

    def _search_item(self, memory: Memory, score: float | None) -> SearchItem:
        key = str(memory.metadata.get(_KEY_FIELD, memory.id))
        return SearchItem(
            namespace=memory.namespace,
            key=key,
            value=dict(memory.data or {}),
            created_at=memory.created_at,
            updated_at=memory.updated_at,
            score=score,
        )

    def _list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        namespaces = self.memory.namespaces()
        if op.match_conditions:
            namespaces = [
                ns for ns in namespaces if all(_matches(c, ns) for c in op.match_conditions)
            ]
        if op.max_depth is not None:
            namespaces = sorted({ns[: op.max_depth] for ns in namespaces})
        return namespaces[op.offset : op.offset + op.limit]

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self._get(op))
            elif isinstance(op, PutOp):
                self._put(op)
                results.append(None)
            elif isinstance(op, SearchOp):
                results.append(self._search(op))
            elif isinstance(op, ListNamespacesOp):
                results.append(self._list_namespaces(op))
            else:  # pragma: no cover - LangGraph has no other op types
                raise TypeError(f"unsupported store operation {type(op).__name__}")
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await asyncio.to_thread(self.batch, list(ops))
