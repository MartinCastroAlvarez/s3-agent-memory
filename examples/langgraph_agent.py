"""A LangGraph graph whose long-term memory is S3Store (runs offline, no LLM).

    pip install "s3-agent-memory[langgraph]"
    python examples/langgraph_agent.py

Swap the URL for s3://bucket/prefix (and S3_AGENT_MEMORY_ENDPOINT_URL for a
local SeaweedFS/MinIO) to keep the same memories in object storage.
"""

from __future__ import annotations

import hashlib
import tempfile
from typing import Any
from typing import TypedDict

from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.store.base import BaseStore

from s3_agent_memory import MemoryStore
from s3_agent_memory.langgraph_store import S3Store


class State(TypedDict):
    user: str
    message: str
    context: list[str]


def remember(state: State, *, store: BaseStore) -> dict[str, Any]:
    if state["message"].lower().startswith("remember"):
        key = hashlib.sha256(state["message"].encode()).hexdigest()[:16]
        store.put(("users", state["user"], "facts"), key, {"text": state["message"][9:]})
    return {}


def recall(state: State, *, store: BaseStore) -> dict[str, Any]:
    hits = store.search(("users", state["user"], "facts"), query=state["message"], limit=3)
    return {"context": [h.value["text"] for h in hits]}


def build(store: BaseStore) -> Any:
    builder = StateGraph(State)
    builder.add_node("remember", remember)
    builder.add_node("recall", recall)
    builder.add_edge(START, "remember")
    builder.add_edge("remember", "recall")
    builder.add_edge("recall", END)
    return builder.compile(store=store)


def main() -> None:
    with tempfile.TemporaryDirectory() as root:
        graph = build(S3Store(MemoryStore.from_url(f"file://{root}")))
        graph.invoke(
            {"user": "alice", "message": "remember I am allergic to peanuts", "context": []}
        )
        out = graph.invoke({"user": "alice", "message": "any peanuts in satay?", "context": []})
        print("alice's context:", out["context"])
        out = graph.invoke({"user": "bob", "message": "any peanuts in satay?", "context": []})
        print("bob's context:  ", out["context"])


if __name__ == "__main__":
    main()
