"""Lexical BM25 index."""

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from .index import top_k
from .text_utils import tokenize


class BM25Index:
    """BM25 Okapi over chunk texts.

    Notes
    -----
    ``rank_bm25`` rebuilds statistics on every ``add``; the corpus is small
    enough that this is acceptable for the MVP.
    """

    def __init__(self) -> None:
        """Create an empty index."""
        self._ids: list[str] = []
        self._corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def __len__(self) -> int:
        """Return the number of documents."""
        return len(self._ids)

    def __str__(self) -> str:
        """Return a short summary."""
        return f"BM25Index(n={len(self)})"

    def add(self, ids: list[str], texts: list[str]) -> None:
        """Index texts.

        Parameters
        ----------
        ids : list[str]
            Chunk ids.
        texts : list[str]
            Chunk texts.
        """
        self._ids.extend(ids)
        self._corpus.extend(tokenize(t) or ["_empty_"] for t in texts)
        self._bm25 = BM25Okapi(self._corpus)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Top-k chunks by BM25 score; zero-score hits are dropped.

        Parameters
        ----------
        query : str
            Question text.
        k : int
            Number of results.

        Returns
        -------
        list[tuple[str, float]]
            (chunk_id, score) sorted descending.
        """
        terms = tokenize(query)
        if self._bm25 is None or not terms:
            return []
        scores = np.asarray(self._bm25.get_scores(terms))
        return [(self._ids[i], float(scores[i])) for i in top_k(scores, k) if scores[i] > 0]
