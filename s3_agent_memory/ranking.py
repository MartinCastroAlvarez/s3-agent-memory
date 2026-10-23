"""Hybrid ranking: lexical/semantic relevance blended with recency and importance.

``score = w_rel * relevance + w_rec * recency + w_imp * importance`` where

* ``relevance`` is BM25 normalised to the best hit, mixed with cosine
  similarity when an embedder is configured (``vector_share``);
* ``recency`` is ``0.5 ** (age / half_life)`` with a half-life per kind -
  episodic memories fade in weeks, semantic facts in months and procedural
  instructions/preferences not at all;
* ``importance`` is the caller-supplied prior in ``[0, 1]``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime


def _default_half_lives() -> dict[str, float | None]:
    return {"episodic": 30.0, "semantic": 180.0, "procedural": None}


@dataclass(frozen=True)
class Ranking:
    relevance: float = 0.7
    recency: float = 0.15
    importance: float = 0.15
    vector_share: float = 0.5
    """Share of relevance taken from cosine similarity when vectors exist."""
    min_similarity: float = 0.35
    """Cosine similarity a memory needs to be a candidate without any keyword match."""
    half_life_days: Mapping[str, float | None] = field(default_factory=_default_half_lives)

    def recency_of(self, kind: str, updated_at: datetime, now: datetime) -> float:
        half_life = self.half_life_days.get(kind)
        if not half_life:
            return 1.0
        age_days = max(0.0, (now - updated_at).total_seconds() / 86_400)
        return float(0.5 ** (age_days / half_life))

    def score(
        self,
        *,
        kind: str,
        updated_at: datetime,
        importance: float,
        now: datetime,
        bm25: float | None,
        similarity: float | None,
    ) -> tuple[float, dict[str, float]]:
        """Return the final score and its components. ``None`` means "signal not used"."""
        parts: list[tuple[float, float]] = []
        if bm25 is not None:
            parts.append((bm25, 1.0 - self.vector_share if similarity is not None else 1.0))
        if similarity is not None:
            parts.append((max(0.0, similarity), self.vector_share if bm25 is not None else 1.0))
        relevance = sum(v * w for v, w in parts)
        recency = self.recency_of(kind, updated_at, now)
        components = {"relevance": relevance, "recency": recency, "importance": importance}
        if bm25 is not None:
            components["bm25"] = bm25
        if similarity is not None:
            components["similarity"] = similarity
        if not parts:  # no query: rank by recency and importance alone
            total = self.recency + self.importance or 1.0
            return (self.recency * recency + self.importance * importance) / total, components
        score = self.relevance * relevance + self.recency * recency + self.importance * importance
        return score, components
