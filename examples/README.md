# Examples

| File | What |
| --- | --- |
| [`quickstart.py`](quickstart.py) | filesystem backend: remember, recall, consolidate, look at the objects (no server, no keys) |
| [`s3_demo.py`](s3_demo.py) | local S3 (`docker compose up -d`): typed memories, 8 concurrent writers, a refused stale update, tombstones |
| [`langgraph_agent.py`](langgraph_agent.py) | a LangGraph graph with `S3Store` as its long-term store (offline, no LLM) |
| [`claude-code.mcp.json`](claude-code.mcp.json) | `.mcp.json` for Claude Code against local S3 (or `claude mcp add memory -- s3-agent-memory serve -n users/me`) |
| [`cursor.mcp.json`](cursor.mcp.json) | `.cursor/mcp.json` with the filesystem backend |

Codex: `codex mcp add memory -- s3-agent-memory serve -n users/me`.
