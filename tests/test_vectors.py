from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest

from s3_agent_memory import InMemoryBackend
from s3_agent_memory import MemoryStore
from s3_agent_memory.errors import AgentMemoryError
from s3_agent_memory.errors import LimitExceededError
from s3_agent_memory.errors import ValidationError
from s3_agent_memory.keys import vectors_key
from s3_agent_memory.vectors import Embedder
from s3_agent_memory.vectors import FastEmbedEmbedder
from s3_agent_memory.vectors import SentenceTransformerEmbedder
from s3_agent_memory.vectors import VectorShard
from s3_agent_memory.vectors import embedder_from_name
from tests.conftest import FakeEmbedder

NS = "users/alice"


@pytest.fixture
def vstore(store: MemoryStore) -> MemoryStore:
    store.embedder = FakeEmbedder()
    return store


def test_hybrid_recall_finds_synonyms_bm25_cannot(vstore: MemoryStore) -> None:
    car = vstore.remember(NS, "Alice drives an automobile to work")
    vstore.remember(NS, "Alice enjoys jazz concerts")
    plain = MemoryStore(vstore.backend, prefix=vstore.prefix)
    assert plain.recall(NS, "car") == []  # no keyword overlap
    hits = vstore.recall(NS, "car")
    assert hits and hits[0].memory.id == car.id
    assert hits[0].components["similarity"] > 0.35
    assert vstore.stats(NS)["vectors"] == 2
    key = vectors_key(vstore.prefix, ("users", "alice"))
    assert vstore.backend.get(key, max_bytes=10**6) is not None


def test_vectors_follow_update_forget_and_reindex(vstore: MemoryStore) -> None:
    m = vstore.remember(NS, "has a puppy named Rex")
    assert vstore.recall(NS, "dog")[0].memory.id == m.id
    vstore.update(NS, m.id, content="drives a vehicle")
    assert vstore.recall(NS, "car")[0].memory.id == m.id
    assert all(
        h.memory.id != m.id or h.components["similarity"] < 0.35 for h in vstore.recall(NS, "dog")
    )
    vstore.forget(NS, m.id)
    assert vstore.stats(NS)["vectors"] == 0
    n = vstore.remember(NS, "owns a cat")
    vstore.backend.delete(vectors_key(vstore.prefix, ("users", "alice")))
    assert vstore.reindex(NS).vectors == 1
    assert vstore.recall(NS, "pet")[0].memory.id == n.id


def test_shard_from_another_model_is_ignored_and_replaced(vstore: MemoryStore) -> None:
    vstore.remember(NS, "one")

    class Other(FakeEmbedder):
        name = "fake:other"

    other = MemoryStore(vstore.backend, prefix=vstore.prefix, embedder=Other())
    assert other.stats(NS)["vectors"] == 0
    other.remember(NS, "two")
    assert other.stats(NS)["vectors"] == 1  # only the new model's vector
    assert vstore.stats(NS)["vectors"] == 0


def test_embedder_failure_keeps_the_memory(caplog: pytest.LogCaptureFixture) -> None:
    class Broken:
        name = "broken"

        def embed(self, texts: object) -> list[list[float]]:
            raise RuntimeError("model crashed")

    store = MemoryStore(InMemoryBackend(), embedder=Broken())
    m = store.remember(NS, "still stored")
    assert store.get(NS, m.id) is not None
    assert "embedding failed" in caplog.text and "still stored" not in caplog.text

    class Wrong:
        name = "wrong"

        def embed(self, texts: object) -> list[list[float]]:
            return []

    with pytest.raises(ValidationError):
        MemoryStore(InMemoryBackend(), embedder=Wrong()).remember(NS, "x")


def test_shard_roundtrip_and_validation() -> None:
    shard = VectorShard.empty("m")
    shard.set("a", [3.0, 4.0])
    shard.set("b", [0.0, 0.0])
    data = shard.to_bytes()
    back = VectorShard.from_bytes(data, max_bytes=10_000)
    assert back.model == "m" and set(back.rows) == {"a", "b"}
    assert back.similarities([3.0, 4.0])["a"] == pytest.approx(1.0, abs=1e-3)
    assert VectorShard.empty("m").similarities([1.0]) == {}
    with pytest.raises(ValidationError):
        shard.set("c", [1.0, 2.0, 3.0])
    with pytest.raises(ValidationError):
        shard.set("c", [float("nan"), 1.0])
    with pytest.raises(ValidationError):
        back.similarities([1.0, 2.0, 3.0])
    assert VectorShard.from_bytes(VectorShard.empty("m").to_bytes(), max_bytes=10_000).rows == {}


def test_hostile_shards_are_refused() -> None:
    with pytest.raises(ValidationError):
        VectorShard.from_bytes(b"not a zip", max_bytes=10_000)
    # An object array would need pickle to load: refused.
    buf = io.BytesIO()
    np.savez(
        buf,
        ids=np.array([{"evil": 1}], dtype=object),
        vectors=np.zeros((1, 2)),
        model=np.array("m"),
    )
    with pytest.raises(ValidationError):
        VectorShard.from_bytes(buf.getvalue(), max_bytes=10**6)
    # Unexpected members.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("payload.py", "print('hi')")
    with pytest.raises(ValidationError):
        VectorShard.from_bytes(buf.getvalue(), max_bytes=10**6)
    # Zip bomb: tiny compressed, huge expanded.
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        ids=np.array(["a"], dtype="<U128"),
        vectors=np.zeros((1, 2_000_000), dtype=np.float16),
        model=np.array("m"),
    )
    assert len(buf.getvalue()) < 100_000
    with pytest.raises(LimitExceededError):
        VectorShard.from_bytes(buf.getvalue(), max_bytes=1_000_000)
    # ids/vectors mismatch.
    buf = io.BytesIO()
    np.savez(
        buf, ids=np.array(["a", "b"], dtype="<U8"), vectors=np.zeros((1, 2)), model=np.array("m")
    )
    with pytest.raises(ValidationError):
        VectorShard.from_bytes(buf.getvalue(), max_bytes=10**6)


def test_embedder_factory() -> None:
    assert embedder_from_name(None) is None and embedder_from_name("none") is None
    fe = embedder_from_name("fastembed")
    assert isinstance(fe, FastEmbedEmbedder) and fe.name == "fastembed:BAAI/bge-small-en-v1.5"
    st = embedder_from_name("sentence-transformers", "all-mpnet-base-v2")
    assert isinstance(st, SentenceTransformerEmbedder) and st.name == "st:all-mpnet-base-v2"
    assert isinstance(embedder_from_name("fastembed", "x"), FastEmbedEmbedder)
    assert isinstance(embedder_from_name("st"), SentenceTransformerEmbedder)
    assert isinstance(FakeEmbedder(), Embedder)
    with pytest.raises(AgentMemoryError):
        embedder_from_name("openai")
