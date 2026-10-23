from __future__ import annotations

import io
import json
from datetime import timedelta

import pytest

from s3_agent_memory import UNSET
from s3_agent_memory import ConflictError
from s3_agent_memory import InMemoryBackend
from s3_agent_memory import LimitExceededError
from s3_agent_memory import Limits
from s3_agent_memory import MemoryStore
from s3_agent_memory import NotFoundError
from s3_agent_memory import ValidationError
from s3_agent_memory.errors import InvalidKeyError
from s3_agent_memory.keys import index_key
from tests.conftest import Clock

NS = "users/alice"


def test_remember_get_and_layout(store: MemoryStore) -> None:
    m = store.remember(
        NS,
        "Alice prefers metric units",
        kind="procedural",
        tags=["prefs", "units", "prefs"],
        importance=0.8,
        source="chat:42",
        metadata={"lang": "en"},
    )
    assert m.tags == ("prefs", "units") and m.version == 1 and m.etag
    key = store.key_of(m)
    assert key == f"memories/users/alice/procedural/{m.id}.json"
    raw = store.backend.get(key, max_bytes=10_000)
    assert raw is not None and json.loads(raw.data)["content"] == "Alice prefers metric units"
    got = store.get(NS, m.id)
    assert got == m
    assert store.get(("users", "alice"), m.id) == m
    assert store.get(NS, "doesnotexist") is None
    assert store.backend.get(index_key("memories", ("users", "alice")), max_bytes=10**6)


def test_recall_ranks_by_bm25_then_recency_and_importance(store: MemoryStore) -> None:
    store.remember(NS, "Alice lives in Lisbon and works on data pipelines")
    units = store.remember(NS, "Alice prefers metric units", kind="procedural")
    store.remember(NS, "Talked about the Q3 roadmap", kind="episodic")
    hits = store.recall(NS, "which units does Alice prefer?")
    assert hits[0].memory.id == units.id
    assert hits[0].components["bm25"] == 1.0
    d = hits[0].to_dict()
    assert d["provenance"]["key"].endswith(f"/procedural/{units.id}.json")
    assert d["provenance"]["namespace"] == NS
    assert store.recall(NS, "roadmap", kind="episodic")[0].memory.kind == "episodic"
    assert store.recall(NS, "roadmap", kind="semantic") == []
    assert store.recall(NS, "zebra quantum") == []
    assert len(store.recall(NS, "", limit=10)) == 3  # empty query: recency + importance
    assert store.recall("users/nobody", "anything") == []


def test_recency_and_importance_break_ties(clock: Clock) -> None:
    store = MemoryStore(InMemoryBackend(), clock=clock)
    old = store.remember(NS, "meeting about budget", kind="episodic")
    clock.advance(days=90)
    new = store.remember(NS, "meeting about budget", kind="episodic")
    hits = store.recall(NS, "budget meeting")
    assert [h.memory.id for h in hits] == [new.id, old.id]
    assert hits[1].components["recency"] == pytest.approx(0.125)
    pref = store.remember(NS, "budget in euros please", kind="procedural", importance=1.0)
    clock.advance(days=3650)
    hit = store.recall(NS, "budget euros", kind="procedural")[0]
    assert hit.memory.id == pref.id and hit.components["recency"] == 1.0  # no decay


def test_tags_filter_and_list(store: MemoryStore, clock: Clock) -> None:
    store.clock = clock
    a = store.remember(NS, "likes tea", tags=["drinks"])
    clock.advance(seconds=1)
    b = store.remember(NS, "likes coffee", tags=["drinks", "morning"])
    clock.advance(seconds=1)
    c = store.remember(NS, "standup at nine", kind="episodic", tags=["morning"])
    assert [m.id for m in store.list(NS)] == [c.id, b.id, a.id]
    assert [m.id for m in store.list(NS, tags=["drinks"])] == [b.id, a.id]
    assert [m.id for m in store.list(NS, kind="episodic")] == [c.id]
    assert [m.id for m in store.list(NS, limit=1, offset=1)] == [b.id]
    assert [h.memory.id for h in store.recall(NS, "likes", tags=["morning"])] == [b.id]
    with pytest.raises(ValidationError):
        store.list(NS, limit=0)
    with pytest.raises(InvalidKeyError):
        store.list(NS, kind="secret")


def test_update_bumps_version_and_reindexes(store: MemoryStore) -> None:
    m = store.remember(NS, "Alice works at Initech")
    updated = store.update(NS, m.id, content="Alice works at Globex", importance=0.9, tags=["job"])
    assert updated.version == 2 and updated.importance == 0.9 and updated.tags == ("job",)
    assert store.recall(NS, "Initech") == []
    assert store.recall(NS, "Globex")[0].memory.version == 2
    again = store.update(
        NS, m.id, expected_version=2, source="hr", metadata={"x": 1}, data={"a": 1}
    )
    assert again.version == 3 and again.source == "hr" and again.data == {"a": 1}
    with pytest.raises(ConflictError):
        store.update(NS, m.id, content="stale", expected_version=2)
    with pytest.raises(NotFoundError):
        store.update(NS, "nope", content="x")
    assert store.update(NS, m.id, content=UNSET).content == "Alice works at Globex"


def test_ttl_expiry_and_prune(store: MemoryStore, clock: Clock) -> None:
    store.clock = clock
    short = store.remember(NS, "temporary code is 4471", ttl=60)
    keep = store.remember(NS, "permanent fact about codes")
    timed = store.remember(NS, "fixed expiry code", expires_at=clock.now + timedelta(hours=1))
    assert {h.memory.id for h in store.recall(NS, "code")} == {short.id, keep.id, timed.id}
    clock.advance(seconds=120)
    assert {h.memory.id for h in store.recall(NS, "code")} == {keep.id, timed.id}
    assert store.get(NS, short.id) is None
    assert store.get(NS, short.id, include_expired=True) is not None
    assert store.stats(NS)["expired"] == 1
    assert store.prune_expired(NS) == [short.id]
    assert store.tombstones(NS)[0]["reason"] == "expired"
    extended = store.update(NS, timed.id, ttl=timedelta(days=1))
    assert extended.expires_at == clock.now + timedelta(days=1)
    assert store.update(NS, timed.id, ttl=None).expires_at is None
    with pytest.raises(ValidationError):
        store.remember(NS, "x", ttl=-1)
    with pytest.raises(ValidationError):
        store.remember(NS, "x", ttl=5, expires_at=clock.now)


def test_forget_hard_deletes_and_tombstones(store: MemoryStore) -> None:
    m = store.remember(NS, "Alice's door code is 1234")
    key = store.key_of(m)
    assert store.forget(NS, m.id, reason="user asked")
    assert store.backend.get(key, max_bytes=10_000) is None
    assert store.get(NS, m.id) is None
    assert store.recall(NS, "door code") == []
    (stone,) = store.tombstones(NS)
    assert stone["id"] == m.id and stone["reason"] == "user asked"
    assert "1234" not in json.dumps(stone)  # tombstones keep no content
    assert not store.forget(NS, m.id)
    with pytest.raises(ValidationError):
        store.forget(NS, m.id, reason="x" * 300)


def test_ids_are_unique_per_namespace_across_kinds(store: MemoryStore) -> None:
    store.remember(NS, "one", id="fixed-id")
    with pytest.raises(ConflictError):
        store.remember(NS, "two", id="fixed-id")
    with pytest.raises(ConflictError):
        store.remember(NS, "two", id="fixed-id", kind="episodic")
    assert store.get(NS, "fixed-id").content == "one"  # type: ignore[union-attr]
    store.remember("users/bob", "bob's", id="fixed-id")  # other namespace: fine


def test_namespace_isolation(store: MemoryStore) -> None:
    secret = store.remember("users/alice", "alice secret diary entry")
    store.remember("users/bob", "bob shopping list")
    assert store.recall("users/bob", "secret diary") == []
    assert store.get("users/bob", secret.id) is None
    assert not store.forget("users/bob", secret.id)
    assert store.get("users/alice", secret.id) is not None
    assert store.list("users") == []  # a parent namespace sees nothing of its children
    assert store.namespaces() == [("users", "alice"), ("users", "bob")]
    assert store.namespaces("users") == [("users", "alice"), ("users", "bob")]
    assert store.namespaces("teams") == []
    for hostile in (
        "users/../users/alice",
        "../alice",
        "users/alice/semantic",
        "users/_index.json",
    ):
        with pytest.raises(InvalidKeyError):
            store.recall(hostile, "secret")


def test_object_under_wrong_key_is_never_served(store: MemoryStore) -> None:
    alice = store.remember("users/alice", "alice private note", id="note1")
    bob_key = store.key_of(alice).replace("users/alice", "users/bob")
    # A tampered bucket: alice's object copied under bob's prefix.
    store.backend.put(bob_key, alice.to_bytes())
    with pytest.raises(ValidationError):
        store.get("users/bob", "note1")
    assert list(store.iter_memories("users/bob")) == []


def test_limits() -> None:
    store = MemoryStore(
        InMemoryBackend(),
        limits=Limits(max_content_bytes=20, max_tags=2, max_memories=2, max_query_chars=10),
    )
    with pytest.raises(LimitExceededError):
        store.remember(NS, "x" * 21)
    with pytest.raises(LimitExceededError):
        store.remember(NS, "ok", tags=["a", "b", "c"])
    with pytest.raises(LimitExceededError):
        store.recall(NS, "q" * 11)
    store.remember(NS, "one")
    store.remember(NS, "two")
    with pytest.raises(LimitExceededError):
        store.remember(NS, "three")
    assert len(list(store.backend.list(""))) == 3  # two memories + index, no orphan object
    for bad in (
        {"importance": 2.0},
        {"importance": True},
        {"tags": ["bad\ntag"]},
        {"tags": ["  "]},
        {"source": "a\x00"},
        {"data": "not a dict"},
    ):
        with pytest.raises((ValidationError, LimitExceededError)):
            MemoryStore(InMemoryBackend()).remember(NS, "x", **bad)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        MemoryStore(InMemoryBackend()).remember(NS, "nul\x00byte")
    with pytest.raises(ValidationError):
        store.recall(NS, 5)  # type: ignore[arg-type]


def test_reindex_rebuilds_from_objects(store: MemoryStore) -> None:
    a = store.remember(NS, "apples are red")
    store.remember(NS, "bananas are yellow", kind="episodic")
    store.backend.delete(index_key(store.prefix, ("users", "alice")))
    assert store.recall(NS, "apples") == []
    assert store.reindex(NS).as_dict() == {"indexed": 2, "invalid": 0, "vectors": 0}
    assert store.recall(NS, "apples")[0].memory.id == a.id
    # A stale index entry whose object is gone is skipped by recall.
    store.backend.delete(store.key_of(a))
    assert store.recall(NS, "apples") == []


def test_export_import_roundtrip(store: MemoryStore) -> None:
    a = store.remember(NS, "fact one", tags=["t"])
    b = store.remember(NS, "event two", kind="episodic")
    gone = store.remember(NS, "forgotten three")
    buf = io.StringIO()
    assert store.export_jsonl(NS, buf) == 3
    store.forget(NS, gone.id)
    buf.seek(0)
    report = store.import_jsonl("users/carol", buf)
    assert report.imported == 3
    assert {m.content for m in store.list("users/carol")} == {
        "fact one",
        "event two",
        "forgotten three",
    }
    # Re-import into alice: existing are skipped, the forgotten one stays forgotten.
    buf.seek(0)
    report = store.import_jsonl(NS, buf)
    assert (report.imported, report.skipped_existing, report.skipped_tombstoned) == (0, 2, 1)
    buf.seek(0)
    report = store.import_jsonl(NS, buf, overwrite=True, resurrect=True)
    assert report.imported == 3
    assert store.get(NS, gone.id) is not None and store.tombstones(NS) == []
    assert store.get(NS, a.id).version == 2  # type: ignore[union-attr]
    assert store.recall(NS, "event")[0].memory.id == b.id


def test_import_ignores_foreign_namespace_and_reports_bad_lines(store: MemoryStore) -> None:
    good = {
        "id": "abc",
        "namespace": ["users", "bob"],
        "kind": "semantic",
        "content": "planted into alice?",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    lines = [
        json.dumps(good),
        "not json",
        "[1, 2]",
        json.dumps({**good, "id": "../../evil"}),
        json.dumps({**good, "id": "x2", "kind": "secret"}),
        json.dumps({**good, "id": "x3", "created_at": "yesterday"}),
        json.dumps({**good, "id": "x4", "schema": 99}),
        json.dumps({k: v for k, v in good.items() if k != "content"}),
        "",
    ]
    report = store.import_jsonl(NS, io.StringIO("\n".join(lines) + "\n"))
    assert report.imported == 1
    assert len(report.errors) == 7
    assert store.get(NS, "abc").namespace == ("users", "alice")  # type: ignore[union-attr]
    assert store.get("users/bob", "abc") is None
    assert report.as_dict()["errors"][0]["line"] == 2


def test_import_line_limit() -> None:
    store = MemoryStore(InMemoryBackend(), limits=Limits(max_jsonl_line_bytes=64))
    report = store.import_jsonl(NS, io.StringIO("x" * 500 + "\n" + "{}\n"))
    assert report.errors[0] == (1, "line too long") and report.imported == 0


def test_from_url_and_repr(tmp_path: object) -> None:
    store = MemoryStore.from_url("memory://")
    assert "InMemoryBackend" in repr(store)
    store.remember("a", "hello world")
    assert store.stats("a")["memories"] == 1
