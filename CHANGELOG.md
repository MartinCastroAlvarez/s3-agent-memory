# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### To do before the first PyPI release
- Register the PyPI pending publisher for `publish.yml` (environment `pypi`),
  then tag `v0.1.0`.

## [0.1.0] — unreleased

### Added
- `MemoryStore`: typed memories (semantic, episodic, procedural) as JSON
  objects at `<prefix>/<namespace>/<kind>/<id>.json` with source, importance,
  tags, TTL/expiry, structured `data`, metadata and a version.
- Operations: remember, get, update (optimistic concurrency with
  `expected_version`), forget (hard delete + tombstone), list, recall,
  prune_expired, reindex, consolidate, stats, namespaces, export/import JSONL.
- Backends: S3 (boto3; AWS S3, R2, SeaweedFS, MinIO) with `If-None-Match` /
  `If-Match` conditional writes and SSE, local filesystem (flock + atomic
  rename), in-memory.
- Per-namespace BM25 index updated by compare-and-swap; optional local
  embeddings (fastembed, sentence-transformers) in a float16 `.npz` shard;
  hybrid ranking with per-kind recency half-lives and importance.
- Consolidation of near-duplicates (Jaccard / cosine) with a local ollama
  summariser or a no-model fallback.
- MCP server (`mcp` 2.x `MCPServer`, stdio) with `remember`, `recall`,
  `forget`, `list`, bound to one namespace; `--read-only`.
- LangGraph `BaseStore` adapter `S3Store` (get/put/delete/search/list_namespaces,
  filters, TTL in minutes, async).
- CLI `s3-agent-memory`; docker-compose (SeaweedFS, optional MinIO) and demo.
- Security: key allow-list, filesystem confinement, size limits, tamper
  checks, pickle-free shard loading with zip-bomb guard, no content or
  credentials in logs.
