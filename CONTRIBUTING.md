# Contributing to s3-agent-memory

## TL;DR

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install poetry
poetry install -E s3 -E embeddings -E langgraph
poetry run pytest
poetry run ruff check s3_agent_memory tests examples && poetry run ruff format --check s3_agent_memory tests examples
poetry run mypy s3_agent_memory
poetry run bandit -c pyproject.toml -q -r s3_agent_memory
poetry run pip-audit
```

Optional, against a real local S3 (`docker compose up -d`):

```bash
S3_AGENT_MEMORY_LIVE_ENDPOINT=http://127.0.0.1:8333 AWS_ACCESS_KEY_ID=local-dev \
  AWS_SECRET_ACCESS_KEY=local-dev-only poetry run pytest -k s3-live --no-cov
```

The live fixture only accepts `localhost` / `127.0.0.1` endpoints.

Open a PR against `main` from a `feat/…`, `fix/…`, `docs/…` or `chore/…` branch.

## What this package will not accept

- A required database, vector service or hosted model.
- Unconditional writes to shared objects (index, shard).
- Keys built outside `keys.py`, or relaxed key validation.
- Logging memory content or credentials.
- Tests that need AWS credentials or network access.

## What it does accept

- Backends for other stores with the same conditional-write semantics.
- Embedder and summariser adapters for local open-source models.
- Better ranking (with a test that shows the ordering it fixes).
- New interfaces (with the namespace bound by the caller, not the model).

## House rules

1. Tests alongside features; security-relevant changes get a hostile case.
2. A memory or index format change bumps its `schema` and handles the old one.
3. No secrets in commits; `.env.example` holds local placeholders only.

## Releasing

1. Bump `version` in `pyproject.toml` and `__version__` in
   `s3_agent_memory/__init__.py`; move `Unreleased` in `CHANGELOG.md` under the new version.
2. Merge, tag the merge commit `vX.Y.Z`, push the tag. `publish.yml` checks
   tag == version, runs the tests, scans the artefacts and publishes through
   PyPI Trusted Publishing (environment `pypi`), then creates the GitHub Release.
   The PyPI pending publisher must be registered first
   (owner `MartinCastroAlvarez`, repository `s3-agent-memory`, workflow `publish.yml`, environment `pypi`).
