"""Tokeniser and Okapi BM25, dependency-free."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from collections.abc import Mapping

_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
_STOPWORDS = """
a an and are as at be been but by can could did do does for from had has have he her his
i if in into is it its me my of on or our she so than that the their them then there these
they this to too us was we were what when where which who why will with would you your
"""
STOPWORDS = frozenset(_STOPWORDS.split())
MAX_TOKEN_CHARS = 40


def _stem(token: str) -> str:
    """A deliberately tiny English suffix stripper: plurals and possessives only."""
    for suffix in ("'s", "’s"):
        if token.endswith(suffix):
            token = token[: -len(suffix)]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens with stopwords removed and plurals folded."""
    out = []
    for match in _WORD.finditer(text.lower()):
        token = _stem(match.group(0))[:MAX_TOKEN_CHARS]
        if (len(token) > 1 or token.isdigit()) and token not in STOPWORDS:
            out.append(token)
    return out


def term_frequencies(text: str) -> dict[str, int]:
    return dict(Counter(tokenize(text)))


class BM25:
    """Okapi BM25 over pre-computed term frequencies (``{doc_id: {term: tf}}``)."""

    def __init__(
        self, docs: Mapping[str, Mapping[str, int]], *, k1: float = 1.2, b: float = 0.75
    ) -> None:
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.lengths = {doc_id: sum(tf.values()) for doc_id, tf in docs.items()}
        self.avg_len = (sum(self.lengths.values()) / len(docs)) if docs else 0.0
        df: Counter[str] = Counter()
        for tf in docs.values():
            df.update(tf.keys())
        n = len(docs)
        self.idf = {term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()}

    def scores(
        self, query: Iterable[str], candidates: Iterable[str] | None = None
    ) -> dict[str, float]:
        """BM25 score per document with a non-zero score."""
        terms = [t for t in dict.fromkeys(query) if t in self.idf]
        out: dict[str, float] = {}
        if not terms:
            return out
        ids = self.docs.keys() if candidates is None else candidates
        for doc_id in ids:
            tf = self.docs.get(doc_id)
            if not tf:
                continue
            length = self.lengths[doc_id]
            norm = self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1.0))
            score = 0.0
            for term in terms:
                freq = tf.get(term, 0)
                if freq:
                    score += self.idf[term] * freq * (self.k1 + 1) / (freq + norm)
            if score > 0:
                out[doc_id] = score
        return out
