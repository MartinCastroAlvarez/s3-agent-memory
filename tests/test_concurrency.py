"""Optimistic concurrency: conditional writes never lose an update."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from s3_agent_memory import ConflictError
from s3_agent_memory import MemoryStore
from s3_agent_memory.backends import FilesystemBackend
from s3_agent_memory.backends import ObjectStore
from s3_agent_memory.backends.base import StoredObject

NS = "team/shared"


def _parallel(n: int, fn: Callable[[int], object]) -> list[BaseException]:
    errors: list[BaseException] = []
    barrier = threading.Barrier(n)

    def run(i: int) -> None:
        barrier.wait()
        try:
            fn(i)
        except BaseException as exc:  # collected and asserted on below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def test_concurrent_remember_keeps_every_index_entry(threaded_store: MemoryStore) -> None:
    threaded_store.max_retries = 200
    errors = _parallel(12, lambda i: threaded_store.remember(NS, f"fact number {i} about topic{i}"))
    assert errors == []
    assert threaded_store.stats(NS)["memories"] == 12
    for i in range(12):
        assert threaded_store.recall(NS, f"topic{i}")[0].memory.content.startswith(
            f"fact number {i} "
        )


def test_concurrent_updates_with_retry_all_land(threaded_store: MemoryStore) -> None:
    threaded_store.max_retries = 200
    m = threaded_store.remember(NS, "counter", tags=[])

    def bump(i: int) -> None:
        threaded_store.update(NS, m.id, metadata={"writer": i})

    assert _parallel(8, bump) == []
    final = threaded_store.get(NS, m.id)
    assert final is not None and final.version == 9


def test_expected_version_makes_exactly_one_winner(threaded_store: MemoryStore) -> None:
    m = threaded_store.remember(NS, "balance is 10")
    wins: list[int] = []

    def write(i: int) -> None:
        threaded_store.update(NS, m.id, content=f"balance is {i}", expected_version=1)
        wins.append(i)

    errors = _parallel(6, write)
    assert len(wins) == 1
    assert len(errors) == 5 and all(isinstance(e, ConflictError) for e in errors)
    final = threaded_store.get(NS, m.id)
    assert final is not None and final.content == f"balance is {wins[0]}" and final.version == 2


def test_two_processes_style_stores_share_a_directory(tmp_path: Path) -> None:
    # Two independent store objects (as two processes would have) on one directory.
    a = MemoryStore(FilesystemBackend(tmp_path), max_retries=200)
    b = MemoryStore(FilesystemBackend(tmp_path), max_retries=200)
    errors = _parallel(10, lambda i: (a if i % 2 else b).remember(NS, f"note {i}"))
    assert errors == []
    assert a.stats(NS)["memories"] == 10 == b.stats(NS)["memories"]


class _Racer(ObjectStore):
    """Wraps a backend and lets another writer sneak in before our next conditional put."""

    def __init__(self, inner: ObjectStore, sneak: Callable[[], None], times: int = 1) -> None:
        self.inner = inner
        self.sneak = sneak
        self.times = times
        self.conflicts = 0

    def get(self, key: str, *, max_bytes: int) -> StoredObject | None:
        return self.inner.get(key, max_bytes=max_bytes)

    def put(self, key: str, data: bytes, **kwargs: object) -> str:
        if self.times and key.endswith("_index.json"):
            self.times -= 1
            self.sneak()
        try:
            return self.inner.put(key, data, **kwargs)  # type: ignore[arg-type]
        except ConflictError:
            self.conflicts += 1
            raise

    def delete(self, key: str) -> None:
        self.inner.delete(key)

    def list(self, prefix: str, *, recursive: bool = True):  # type: ignore[no-untyped-def]
        return self.inner.list(prefix, recursive=recursive)


def test_lost_race_on_the_index_is_retried_not_lost(backend: ObjectStore) -> None:
    other = MemoryStore(backend)
    racer = _Racer(backend, lambda: other.remember(NS, "the other writer's memory"))
    mine = MemoryStore(racer)
    mine.remember(NS, "my memory")
    assert racer.conflicts == 1  # the S3 412 really happened ...
    assert {m.content for m in mine.list(NS)} == {"my memory", "the other writer's memory"}


def test_gives_up_after_max_retries(backend: ObjectStore) -> None:
    other = MemoryStore(backend)
    counter = iter(range(1000))
    racer = _Racer(backend, lambda: other.remember(NS, f"spam {next(counter)}"), times=10**6)
    mine = MemoryStore(racer, max_retries=3)
    with pytest.raises(ConflictError):
        mine.remember(NS, "never indexed")
    assert all(m.content != "never indexed" for m in other.list(NS, limit=100))
