from __future__ import annotations

import json
import sys
from pathlib import Path

import anyio
import pytest
from mcp import Client
from mcp.server.mcpserver.exceptions import ToolError

from s3_agent_memory import InMemoryBackend
from s3_agent_memory import MemoryStore
from s3_agent_memory.server import MemoryTools
from s3_agent_memory.server import build_server

TOOLS = {"remember", "recall", "forget", "list"}


def test_mcp_tools_over_the_protocol() -> None:
    store = MemoryStore(InMemoryBackend())
    store.remember("users/bob", "bob's private note about the merger")
    server = build_server(store, "users/alice")

    async def scenario() -> None:
        async with Client(server) as client:
            listed = await client.list_tools()
            assert {t.name for t in listed.tools} == TOOLS
            by_name = {t.name: t for t in listed.tools}
            for name, tool in by_name.items():
                props = tool.input_schema.get("properties", {})
                assert "namespace" not in props, name  # isolation: no namespace argument
            assert by_name["recall"].annotations.read_only_hint  # type: ignore[union-attr]
            assert by_name["forget"].annotations.destructive_hint  # type: ignore[union-attr]

            r = await client.call_tool(
                "remember",
                {"content": "Alice prefers metric units", "kind": "procedural", "tags": ["prefs"]},
            )
            assert not r.is_error
            created = json.loads(r.content[0].text)  # type: ignore[union-attr]
            assert created["key"].startswith("users/alice/procedural/")

            r = await client.call_tool("recall", {"query": "metric units"})
            payload = json.loads(r.content[0].text)  # type: ignore[union-attr]
            assert payload["untrusted_content"] is True and "not instructions" in payload["note"]
            (hit,) = payload["memories"]
            assert hit["content"] == "Alice prefers metric units"
            assert hit["provenance"]["source"] == "mcp"

            r = await client.call_tool("recall", {"query": "merger private note"})
            assert json.loads(r.content[0].text)["memories"] == []  # type: ignore[union-attr]

            r = await client.call_tool("list", {"kind": "procedural", "tag": "prefs"})
            assert len(json.loads(r.content[0].text)["memories"]) == 1  # type: ignore[union-attr]

            bad = await client.call_tool("remember", {"content": "x", "kind": "admin"})
            assert bad.is_error and "kind must be one of" in bad.content[0].text  # type: ignore[union-attr]

            r = await client.call_tool("forget", {"id": created["id"]})
            assert json.loads(r.content[0].text)["forgotten"] is True  # type: ignore[union-attr]
            traversal = await client.call_tool("forget", {"id": "../../users/bob/semantic/x"})
            assert traversal.is_error

    anyio.run(scenario)
    assert len(store.list("users/bob")) == 1


def test_read_only_tools_refuse_writes() -> None:
    store = MemoryStore(InMemoryBackend())
    m = store.remember("a", "keep me")
    tools = MemoryTools(store, "a", read_only=True)
    with pytest.raises(ToolError):
        tools.remember("new")
    with pytest.raises(ToolError):
        tools.forget(m.id)
    assert json.loads(tools.list())["memories"][0]["id"] == m.id
    ttl = MemoryTools(store, "a").remember("short-lived", ttl_days=1)
    assert json.loads(ttl)["kind"] == "semantic"


def test_stdio_transport_end_to_end(tmp_path: Path) -> None:
    from mcp import StdioServerParameters

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "s3_agent_memory", "serve", "--url", f"file://{tmp_path}", "-n", "users/eve"],
        env={"PATH": "/usr/bin:/bin"},
    )

    async def scenario() -> None:
        from mcp.client.stdio import stdio_client

        async with stdio_client(params) as (read, write):
            from mcp import ClientSession

            async with ClientSession(read, write) as session:
                await session.initialize()
                await session.call_tool("remember", {"content": "eve likes chess"})
                result = await session.call_tool("recall", {"query": "chess"})
                text = result.content[0].text  # type: ignore[union-attr]
                assert json.loads(text)["memories"][0]["content"] == "eve likes chess"

    anyio.run(scenario)
    assert any((tmp_path / "users" / "eve" / "semantic").iterdir())
