# CLAUDE.md

Contract between this repository and any AI agent contributing to it.

## Required reading

1. This file. 2. [`README.md`](README.md). 3. [`ARCHITECTURE.md`](ARCHITECTURE.md).
4. [`SECURITY.md`](SECURITY.md).

## Mission

`s3-agent-memory` is long-term agent memory on plain objects: no database, no
hosted services, safe under concurrent writers, and strict about namespaces.

## The rules

1. **Keys only from `keys.py`.** Never build an object key by string
   concatenation elsewhere; every new path segment gets a hostile test case.
2. **Shared objects change through `MemoryStore._cas`** (conditional writes).
   No unconditional overwrite of an index or shard.
3. **Memory objects are the source of truth.** Anything else must be
   rebuildable by `reindex`.
4. **Untrusted content.** Memory text is data: never log it, never put it in
   tombstones, always return it with provenance.
5. **No network in tests**: moto, the filesystem backend, fakes for embedders
   and summarisers. Never AWS, never a hosted model.
6. **Every interface** (Python, CLI, MCP, LangGraph) gets tests for a new
   operation, plus README and CHANGELOG entries.
7. Never add `# noqa`, `# nosec` or `# type: ignore` to the package, skip a
   test, or loosen lint/type/coverage config. Fix the code.

## Done means

`poetry run pytest` (coverage >= 85%), `ruff check`, `ruff format --check`,
`mypy` (strict), `bandit`, `pip-audit` all green; docs and CHANGELOG updated.

## Git

Small commits, imperative subjects, never force-push `main`, never commit
secrets, `.env` or `.agent-memory/`.
