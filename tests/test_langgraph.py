from __future__ import annotations

import asyncio
from typing import Any
from typing import TypedDict

import pytest
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.base import MatchCondition

from s3_agent_memory import MemoryStore
from s3_agent_memory.langgraph_store import S3Store
from s3_agent_memory.langgraph_store import memory_id_for


@pytest.fixture
def lg(store: MemoryStore) -> S3Store:
    return S3Store(store)


def test_put_get_update_delete(lg: S3Store) -> None:
    ns = ("users", "alice", "memories")
    lg.put(ns, "pref-1", {"text": "likes green tea", "score": 3})
    item = lg.get(ns, "pref-1")
    assert item is not None and item.value == {"text": "likes green tea", "score": 3}
    assert item.key == "pref-1" and item.namespace == ns
    lg.put(ns, "pref-1", {"text": "likes black tea", "score": 4})
    updated = lg.get(ns, "pref-1")
    assert updated is not None and updated.value["text"] == "likes black tea"
    assert updated.created_at == item.created_at and updated.updated_at >= item.updated_at
    lg.delete(ns, "pref-1")
    assert lg.get(ns, "pref-1") is None
    lg.delete(ns, "never-existed")
    assert lg.get(ns, "missing") is None


def test_unsafe_keys_are_hashed_not_trusted(lg: S3Store, store: MemoryStore) -> None:
    ns = ("users", "alice")
    key = "../../users/bob/semantic/steal me"
    lg.put(ns, key, {"text": "harmless"})
    item = lg.get(ns, key)
    assert item is not None and item.key == key
    mid = memory_id_for(key)
    assert mid.startswith("k") and len(mid) == 41
    assert store.get("users/alice", mid) is not None
    assert store.namespaces() == [("users", "alice")]
    assert lg.get(ns, mid) is None  # the hashed id is not an alias for the key


def test_search_query_filter_and_prefix(lg: S3Store) -> None:
    lg.put(("users", "alice"), "a", {"text": "alice loves hiking in the alps", "level": 3})
    lg.put(("users", "alice"), "b", {"text": "alice owns a road bike", "level": 1})
    lg.put(("users", "bob"), "c", {"text": "bob loves hiking too", "level": 5})
    hits = lg.search(("users", "alice"), query="hiking")
    assert [h.key for h in hits] == ["a"] and hits[0].score is not None
    assert {h.key for h in lg.search(("users",), query="hiking")} == {"a", "c"}
    assert {h.key for h in lg.search(("users",))} == {"a", "b", "c"}
    filtered = lg.search(("users",), filter={"level": {"$gte": 3}}, query="hiking")
    assert {h.key for h in filtered} == {"a", "c"}
    assert {h.key for h in lg.search(("users",), filter={"level": {"$gt": 2}})} == {"a", "c"}
    assert {h.key for h in lg.search(("users",), filter={"level": 1})} == {"b"}
    assert {h.key for h in lg.search(("users",), filter={"level": {"$ne": 1, "$lte": 3}})} == {"a"}
    assert {h.key for h in lg.search(("users",), filter={"level": {"$lt": 2}})} == {"b"}
    assert {h.key for h in lg.search(("users",), filter={"level": {"$eq": 5}})} == {"c"}
    assert lg.search(("users",), filter={"text": {"nested": 1}}) == []
    assert lg.search(("users",), filter={"missing": {"$gt": 1}}) == []
    assert lg.search(("users",), filter={"text": {"$gt": 1}}) == []
    assert len(lg.search(("users",), limit=1, offset=1)) == 1
    with pytest.raises(ValueError):
        lg.search(("users",), filter={"level": {"$regex": "x"}})


def test_index_fields_and_disabled_index(store: MemoryStore) -> None:
    lg = S3Store(store, index=["title"])
    lg.put(("docs",), "1", {"title": "quarterly report", "body": "secret numbers"})
    assert [h.key for h in lg.search(("docs",), query="quarterly")] == ["1"]
    assert lg.search(("docs",), query="secret") == []
    lg.put(("docs",), "2", {"title": "x", "tags": ["alpha", "beta"]}, index=["tags[*]"])
    assert [h.key for h in lg.search(("docs",), query="beta")] == ["2"]
    lg.put(("docs",), "3", {"title": "hidden thing"}, index=False)
    assert lg.search(("docs",), query="hidden") == []
    assert lg.get(("docs",), "3") is not None
    lg.put(("docs",), "4", {"title": {"nested": "deep"}}, index=["title"])
    assert [h.key for h in lg.search(("docs",), query="nested")] == ["4"]


def test_list_namespaces(lg: S3Store) -> None:
    for ns in [("users", "alice", "prefs"), ("users", "bob", "prefs"), ("teams", "core", "notes")]:
        lg.put(ns, "k", {"text": "x"})
    assert lg.list_namespaces() == [
        ("teams", "core", "notes"),
        ("users", "alice", "prefs"),
        ("users", "bob", "prefs"),
    ]
    assert lg.list_namespaces(prefix=("users",)) == [
        ("users", "alice", "prefs"),
        ("users", "bob", "prefs"),
    ]
    assert lg.list_namespaces(suffix=("notes",)) == [("teams", "core", "notes")]
    assert lg.list_namespaces(prefix=("users", "*", "prefs")) == [
        ("users", "alice", "prefs"),
        ("users", "bob", "prefs"),
    ]
    assert lg.list_namespaces(max_depth=1) == [("teams",), ("users",)]
    assert lg.list_namespaces(limit=1, offset=1) == [("users", "alice", "prefs")]
    assert lg.batch([lg_op()])[0] == [("teams", "core", "notes")]


def lg_op() -> Any:
    from langgraph.store.base import ListNamespacesOp

    return ListNamespacesOp(match_conditions=(MatchCondition("prefix", ("teams",)),), limit=5)


def test_ttl_minutes_and_refresh(store: MemoryStore, clock: Any) -> None:
    store.clock = clock
    lg = S3Store(store, ttl={"default_ttl": 10, "refresh_on_read": True})
    lg.put(("s",), "a", {"text": "session scratch"})
    lg.put(("s",), "b", {"text": "forever"}, ttl=None)
    clock.advance(minutes=8)
    assert lg.get(("s",), "a") is not None  # refreshes to +10 minutes
    clock.advance(minutes=8)
    assert lg.get(("s",), "a", refresh_ttl=False) is not None
    clock.advance(minutes=3)
    assert lg.get(("s",), "a") is None
    assert lg.get(("s",), "b") is not None
    lg.put(("s",), "c", {"text": "custom"}, ttl=1)
    clock.advance(minutes=2)
    assert lg.get(("s",), "c") is None


def test_async_api(lg: S3Store) -> None:
    async def scenario() -> None:
        await lg.aput(("a",), "k", {"text": "async works"})
        item = await lg.aget(("a",), "k")
        assert item is not None and item.value["text"] == "async works"
        assert [h.key for h in await lg.asearch(("a",), query="async")] == ["k"]
        assert await lg.alist_namespaces() == [("a",)]
        await lg.adelete(("a",), "k")
        assert await lg.aget(("a",), "k") is None

    asyncio.run(scenario())


def test_inside_a_langgraph_graph(lg: S3Store) -> None:
    class State(TypedDict):
        user: str
        message: str
        recalled: list[str]

    def remember(state: State, *, store: BaseStore) -> dict[str, Any]:
        store.put(("users", state["user"]), "last", {"text": state["message"]})
        return {}

    def recall(state: State, *, store: BaseStore) -> dict[str, Any]:
        hits = store.search(("users", state["user"]), query="pizza")
        return {"recalled": [h.value["text"] for h in hits]}

    builder = StateGraph(State)
    builder.add_node("remember", remember)
    builder.add_node("recall", recall)
    builder.add_edge(START, "remember")
    builder.add_edge("remember", "recall")
    builder.add_edge("recall", END)
    graph = builder.compile(store=lg)
    out = graph.invoke({"user": "alice", "message": "I love pizza margherita", "recalled": []})
    assert out["recalled"] == ["I love pizza margherita"]
    other = graph.invoke({"user": "bob", "message": "salad please", "recalled": []})
    assert other["recalled"] == []
