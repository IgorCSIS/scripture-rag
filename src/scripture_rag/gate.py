"""Answer gate: reranker score per language plus a lexical support check (FR-24).

An answer is allowed only when the best reranked passage clears the
threshold for the question's language *and* the passage lexically supports
the question: an IDF-weighted share of the question's content words (after
translation and archaic normalization, matched on 5-letter stems) must appear
in the passage. This blocks answers built from unrelated passages that a
cross-encoder scores highly because of topical overlap (for example a note
that mentions a commentator by name but not the asked content).

Cost: O(|q| + |p|) per passage with hashed term sets.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

from .text_utils import tokenize

STEM = 5


def stems(text: str) -> set[str]:
    """Content-word stems (first five letters)."""
    return {t[:STEM] for t in tokenize(text) if len(t) > 2}


class IdfTable:
    """Document frequencies over passage stems for support weighting."""

    def __init__(self, texts: Iterable[str]) -> None:
        """Count stems once per passage. O(total tokens)."""
        df: dict[str, int] = {}
        n = 0
        for t in texts:
            n += 1
            for s in stems(t):
                df[s] = df.get(s, 0) + 1
        self.n = max(n, 1)
        self.df = df

    def idf(self, s: str) -> float:
        """Smoothed IDF; unseen stems get the maximum weight."""
        return math.log((self.n + 1) / (self.df.get(s, 0) + 0.5))


def support(question: str, passage: str, idf: IdfTable) -> float:
    """IDF-weighted share of question stems found in the passage, in [0, 1]."""
    q = stems(question)
    if not q:
        return 0.0
    p = stems(passage)
    total = sum(idf.idf(s) for s in q)
    return sum(idf.idf(s) for s in q if s in p) / total if total else 0.0


@dataclass(frozen=True)
class GatePolicy:
    """Thresholds tuned on the tuning split.

    Parameters
    ----------
    thresholds : dict[str, float]
        Minimum reranker score per language (``en``, ``pt``, ``es``).
    default : float
        Threshold for other languages.
    min_support : float
        Minimum lexical support of the best passage.
    """

    thresholds: dict[str, float] = field(default_factory=dict)
    default: float = 0.5
    min_support: float = 0.0

    def threshold(self, lang: str) -> float:
        """Threshold for one language."""
        return self.thresholds.get(lang, self.default)

    def allow(self, lang: str, score: float, sup: float) -> bool:
        """True when both checks pass."""
        return score >= self.threshold(lang) and sup >= self.min_support
