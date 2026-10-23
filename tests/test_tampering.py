"""A bucket is shared infrastructure: malformed or hostile objects are refused, never trusted."""

from __future__ import annotations

import json

import pytest

from s3_agent_memory import InMemoryBackend
from s3_agent_memory import Limits
from s3_agent_memory import Memory
from s3_agent_memory import MemoryStore
from s3_agent_memory import ValidationError
from s3_agent_memory.errors import InvalidKeyError
from s3_agent_memory.errors import LimitExceededError
from s3_agent_memory.errors import NotFoundError
from s3_agent_memory.index import NamespaceIndex
from s3_agent_memory.keys import index_key

NS = ("u",)
ENTRY = {"k": "semantic", "tf": {"a": 1}, "c": "2026-01-01T00:00:00Z", "u": "2026-01-01T00:00:00Z"}


def _index(**docs: object) -> bytes:
    return json.dumps({"schema": 1, "namespace": ["u"], "docs": docs}).encode()


@pytest.mark.parametrize(
    "payload",
    [
        b"not json",
        json.dumps({"schema": 2, "namespace": ["u"], "docs": {}}).encode(),
        json.dumps({"schema": 1, "namespace": ["other"], "docs": {}}).encode(),
        json.dumps({"schema": 1, "namespace": ["u"], "docs": []}).encode(),
        _index(**{"../x": ENTRY}),
        _index(a="nope"),
        _index(a={**ENTRY, "tf": {"a": -1}}),
        _index(a={**ENTRY, "tf": "a"}),
        _index(a={**ENTRY, "t": [1]}),
        _index(a={**ENTRY, "i": "high"}),
        _index(a={**ENTRY, "i": True}),
        _index(a={**ENTRY, "v": "1"}),
        _index(a={**ENTRY, "k": "admin"}),
        _index(a={**ENTRY, "c": "yesterday"}),
        _index(a={**ENTRY, "c": 5}),
    ],
)
def test_hostile_index_objects_are_refused(payload: bytes) -> None:
    with pytest.raises((ValidationError, InvalidKeyError)):
        NamespaceIndex.from_bytes(payload, NS)


def test_valid_index_roundtrip() -> None:
    idx = NamespaceIndex.from_bytes(_index(a={**ENTRY, "x": "2027-01-01T00:00:00", "t": ["x"]}), NS)
    assert idx.docs["a"].expires_at is not None and idx.docs["a"].tags == ("x",)
    assert NamespaceIndex.from_bytes(idx.to_bytes(), NS).docs.keys() == {"a"}


def test_oversized_objects_are_refused() -> None:
    store = MemoryStore(InMemoryBackend(), limits=Limits(max_object_bytes=200, max_index_bytes=300))
    m = store.remember("u", "hello")
    store.backend.put(store.key_of(m), m.to_bytes() + b" " * 500)
    with pytest.raises(LimitExceededError):
        store.get("u", m.id)
    store.backend.put(index_key("", NS), b"{" + b" " * 400 + b"}")
    with pytest.raises(LimitExceededError):
        store.recall("u", "hello")


BASE = {
    "id": "abc",
    "namespace": ["u"],
    "kind": "semantic",
    "content": "x",
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z",
}


@pytest.mark.parametrize(
    "change",
    [
        {"tags": "a,b"},
        {"namespace": 5},
        {"namespace": ["u", ".."]},
        {"metadata": [1]},
        {"metadata": {"x": "y" * 5000}},
        {"data": {"x": "y" * 40_000}},
        {"version": 0},
        {"content": 7},
        {"expires_at": "soon"},
        {"source": "s" * 300},
        {"tags": [5]},
    ],
)
def test_hostile_memory_objects_are_refused(change: dict[str, object]) -> None:
    with pytest.raises((ValidationError, InvalidKeyError, LimitExceededError)):
        Memory.from_dict({**BASE, **change}, Limits())


def test_memory_json_edge_cases() -> None:
    with pytest.raises(ValidationError):
        Memory.from_bytes(b"\xff\xfe", Limits())
    with pytest.raises(ValidationError):
        Memory.from_dict([], Limits())
    naive = Memory.from_dict({**BASE, "created_at": "2026-01-01T00:00:00"}, Limits())
    assert naive.created_at.tzinfo is not None
    assert str(NotFoundError("gone")) == "gone" and str(NotFoundError()) == "not found"
