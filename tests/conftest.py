from __future__ import annotations

import hashlib
import os
import secrets
from collections.abc import Iterator
from collections.abc import Sequence
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import pytest

from s3_agent_memory import MemoryStore
from s3_agent_memory.backends import FilesystemBackend
from s3_agent_memory.backends import InMemoryBackend
from s3_agent_memory.backends import ObjectStore
from s3_agent_memory.text import tokenize

BUCKET = "agent-memory-test"
# Fake values only: moto never talks to AWS. They also let tests prove they never leak.
FAKE_KEY_ID = "AKIAFAKEFAKEFAKE0000"
FAKE_SECRET = "fake-secret-never-logged-1234567890"


@pytest.fixture
def aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", FAKE_KEY_ID)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", FAKE_SECRET)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    for name in ("AWS_PROFILE", "AWS_SESSION_TOKEN", "AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def s3_backend(aws_env: None) -> Iterator[ObjectStore]:
    import boto3
    from moto import mock_aws

    from s3_agent_memory.backends.s3 import S3Backend

    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield S3Backend(BUCKET, client=client)


LIVE_ENDPOINT = os.environ.get("S3_AGENT_MEMORY_LIVE_ENDPOINT")
"""Opt-in: run every backend test against a LOCAL S3-compatible server too (never AWS)."""


def _live_backend() -> ObjectStore:
    import boto3

    from s3_agent_memory.backends.s3 import S3Backend

    assert LIVE_ENDPOINT and LIVE_ENDPOINT.startswith(("http://localhost", "http://127.0.0.1"))
    client = boto3.client("s3", endpoint_url=LIVE_ENDPOINT, region_name="us-east-1")
    bucket = f"it-{secrets.token_hex(6)}"
    client.create_bucket(Bucket=bucket)
    return S3Backend(bucket, client=client)


BACKENDS = ["memory", "filesystem", "s3"] + (["s3-live"] if LIVE_ENDPOINT else [])


@pytest.fixture(params=BACKENDS)
def backend(request: pytest.FixtureRequest, tmp_path: Path) -> ObjectStore:
    if request.param == "memory":
        return InMemoryBackend()
    if request.param == "filesystem":
        return FilesystemBackend(tmp_path / "store")
    if request.param == "s3-live":
        return _live_backend()
    result: ObjectStore = request.getfixturevalue("s3_backend")
    return result


THREADED_BACKENDS = ["memory", "filesystem"] + (["s3-live"] if LIVE_ENDPOINT else [])
"""Backends for tests that start real threads. moto's in-process mock is not safe under
concurrent requests (its own ValueErrors), so real-S3 races run against the local
SeaweedFS server in CI (s3-live); moto still covers the 412 path deterministically."""


@pytest.fixture(params=THREADED_BACKENDS)
def threaded_backend(request: pytest.FixtureRequest, tmp_path: Path) -> ObjectStore:
    if request.param == "memory":
        return InMemoryBackend()
    if request.param == "filesystem":
        return FilesystemBackend(tmp_path / "store")
    return _live_backend()


@pytest.fixture
def threaded_store(threaded_backend: ObjectStore) -> MemoryStore:
    return MemoryStore(threaded_backend, prefix="memories")


@pytest.fixture
def store(backend: ObjectStore) -> MemoryStore:
    return MemoryStore(backend, prefix="memories")


class Clock:
    def __init__(self, start: datetime | None = None) -> None:
        self.now = start or datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now += timedelta(**kwargs)


@pytest.fixture
def clock() -> Clock:
    return Clock()


SYNONYMS = {"automobile": "car", "vehicle": "car", "dog": "pet", "puppy": "pet", "cat": "pet"}


class FakeEmbedder:
    """Deterministic bag-of-words hashing with a synonym table: 'semantic' enough for tests."""

    name = "fake:bow-64"

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        out = []
        for text in texts:
            vec = [0.0] * 64
            for token in tokenize(text):
                token = SYNONYMS.get(token, token)
                vec[int(hashlib.sha256(token.encode()).hexdigest(), 16) % 64] += 1.0
            out.append(vec)
        return out


class FakeSummariser:
    def __init__(self) -> None:
        self.seen: list[list[str]] = []

    def merge(self, texts: Sequence[str]) -> str:
        self.seen.append(list(texts))
        return "MERGED: " + " | ".join(texts)
