"""Hybrid retrieval with Reciprocal Rank Fusion (FR-04)."""

from __future__ import annotations

from typing import Final

from .bm25 import BM25Index
from .embed import Embedder
from .index import VectorIndex

RRF_K: Final[int] = 60


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Fuse ranked lists with Reciprocal Rank Fusion.

    score(d) = sum over lists of 1 / (k + rank(d)), rank starting at 1.

    Parameters
    ----------
    rankings : list[list[str]]
        Ranked id lists, best first.
    k : int
        Smoothing constant.

    Returns
    -------
    list[tuple[str, float]]
        Ids with fused scores, best first; ties keep first-seen order.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class HybridRetriever:
    """Combine BM25 and vector search.

    Parameters
    ----------
    embedder : Embedder
        Query embedder.
    vectors : VectorIndex
        Dense index.
    bm25 : BM25Index
        Lexical index.
    """

    def __init__(self, embedder: Embedder, vectors: VectorIndex, bm25: BM25Index) -> None:
        """Store components."""
        self._embedder = embedder
        self._vectors = vectors
        self._bm25 = bm25

    def __str__(self) -> str:
        """Return a short summary."""
        return f"HybridRetriever({self._vectors}, {self._bm25})"

    def vector_only(self, query: str, k: int) -> list[str]:
        """Dense ranking.

        Parameters
        ----------
        query : str
            Question.
        k : int
            Depth.

        Returns
        -------
        list[str]
            Chunk ids.
        """
        qv = self._embedder.encode([query], is_query=True)[0]
        return [cid for cid, _ in self._vectors.search(qv, k)]

    def bm25_only(self, query: str, k: int) -> list[str]:
        """Lexical ranking.

        Parameters
        ----------
        query : str
            Question.
        k : int
            Depth.

        Returns
        -------
        list[str]
            Chunk ids.
        """
        return [cid for cid, _ in self._bm25.search(query, k)]

    def retrieve(self, query: str, k: int = 50) -> list[str]:
        """Hybrid ranking fused by RRF.

        Parameters
        ----------
        query : str
            Question.
        k : int
            Depth of each list and of the result.

        Returns
        -------
        list[str]
            Fused chunk ids, best first.
        """
        fused = rrf([self.bm25_only(query, k), self.vector_only(query, k)])
        return [cid for cid, _ in fused[:k]]
