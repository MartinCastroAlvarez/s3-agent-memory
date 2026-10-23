"""Optional local embeddings (``pip install "s3-agent-memory[fastembed]"``, or
``[embeddings]`` plus an embedder of your own).

Vectors live in one compact shard per namespace, ``_vectors.npz``: an
``ids`` array of fixed-width unicode strings and a ``vectors`` float16 matrix
of L2-normalised rows. It is loaded with ``allow_pickle=False`` and its
uncompressed size is checked before loading, so a hostile shard cannot run
code or inflate a zip bomb.

Embedders run locally and open-source models only:
:class:`FastEmbedEmbedder` (ONNX, default ``BAAI/bge-small-en-v1.5``) or
:class:`SentenceTransformerEmbedder` (``pip install sentence-transformers``).
Anything with ``name`` and ``embed(texts) -> list[list[float]]`` works.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol
from typing import runtime_checkable

from s3_agent_memory.errors import AgentMemoryError
from s3_agent_memory.errors import LimitExceededError
from s3_agent_memory.errors import ValidationError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    Matrix = npt.NDArray[np.float32]


@runtime_checkable
class Embedder(Protocol):
    """Turns texts into vectors. Must be deterministic for the same model."""

    name: str

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


def _numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise AgentMemoryError(
            'embeddings need numpy: pip install "s3-agent-memory[embeddings]"'
        ) from exc
    return numpy


class FastEmbedEmbedder:
    """Local ONNX embeddings via `fastembed` (model downloaded once from Hugging Face)."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", **kwargs: Any) -> None:
        self.name = f"fastembed:{model}"
        self._model_name = model
        self._kwargs = kwargs
        self._model: Any = None

    def embed(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover - model I/O
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise AgentMemoryError('pip install "s3-agent-memory[fastembed]"') from exc

            self._model = TextEmbedding(self._model_name, **self._kwargs)
        return [list(map(float, v)) for v in self._model.embed(list(texts))]


class SentenceTransformerEmbedder:
    """Local embeddings via `sentence-transformers` (installed separately)."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.name = f"st:{model}"
        self._model_name = model
        self._model: Any = None

    def embed(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover - model I/O
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise AgentMemoryError("pip install sentence-transformers") from exc

            self._model = SentenceTransformer(self._model_name)
        return [list(map(float, v)) for v in self._model.encode(list(texts))]


def embedder_from_name(name: str | None, model: str | None = None) -> Embedder | None:
    """``none`` / ``fastembed`` / ``sentence-transformers`` -> an embedder (or ``None``)."""
    if not name or name == "none":
        return None
    if name == "fastembed":
        return FastEmbedEmbedder(model) if model else FastEmbedEmbedder()
    if name in {"sentence-transformers", "st"}:
        return SentenceTransformerEmbedder(model) if model else SentenceTransformerEmbedder()
    raise AgentMemoryError(f"unknown embedder {name!r} (none, fastembed, sentence-transformers)")


@dataclass
class VectorShard:
    """Normalised vectors keyed by memory id, for one namespace and one model."""

    model: str
    rows: dict[str, Matrix]

    @classmethod
    def empty(cls, model: str) -> VectorShard:
        return cls(model, {})

    def set(self, memory_id: str, vector: Sequence[float]) -> None:
        np = _numpy()
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            raise ValidationError("embedding must be a non-empty finite vector")
        if self.rows:
            dim = next(iter(self.rows.values())).shape[0]
            if arr.shape[0] != dim:
                raise ValidationError(
                    f"embedding dimension {arr.shape[0]} != shard dimension {dim}"
                )
        norm = float(np.linalg.norm(arr))
        self.rows[memory_id] = arr / norm if norm else arr

    def remove(self, memory_id: str) -> bool:
        return self.rows.pop(memory_id, None) is not None

    def similarities(self, query: Sequence[float]) -> dict[str, float]:
        """Cosine similarity of ``query`` to every row."""
        if not self.rows:
            return {}
        np = _numpy()
        ids = list(self.rows)
        matrix = np.stack([self.rows[i] for i in ids])
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != matrix.shape[1]:
            raise ValidationError("query embedding dimension does not match the shard")
        norm = float(np.linalg.norm(q))
        sims = matrix @ (q / norm if norm else q)
        return {i: float(s) for i, s in zip(ids, sims, strict=True)}

    def to_bytes(self) -> bytes:
        np = _numpy()
        ids = sorted(self.rows)
        dim = next(iter(self.rows.values())).shape[0] if ids else 0
        matrix = (
            np.stack([self.rows[i] for i in ids]).astype(np.float16)
            if ids
            else np.zeros((0, dim), dtype=np.float16)
        )
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            ids=np.asarray(ids, dtype="<U128"),
            vectors=matrix,
            model=np.asarray(self.model, dtype="<U256"),
        )
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes, *, max_bytes: int) -> VectorShard:
        np = _numpy()
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                infos = zf.infolist()
        except (zipfile.BadZipFile, OSError) as exc:
            raise ValidationError("vector shard is not a valid .npz") from exc
        if {info.filename for info in infos} != {"ids.npy", "vectors.npy", "model.npy"}:
            raise ValidationError("vector shard has unexpected members")
        if sum(info.file_size for info in infos) > max_bytes:
            raise LimitExceededError("vector shard expands beyond the size limit")
        try:
            with np.load(io.BytesIO(data), allow_pickle=False) as npz:
                ids = [str(i) for i in npz["ids"].tolist()]
                vectors = npz["vectors"].astype(np.float32)
                model = str(npz["model"].item())
        except (zipfile.BadZipFile, ValueError, OSError, KeyError) as exc:
            raise ValidationError(f"vector shard is unreadable: {type(exc).__name__}") from exc
        if vectors.ndim != 2 or vectors.shape[0] != len(ids):
            raise ValidationError("vector shard ids and vectors disagree")
        return cls(model, {i: vectors[n] for n, i in enumerate(ids)})
