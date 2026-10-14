"""Cross-book similar passages (FR-10)."""

from __future__ import annotations

import numpy as np

from .chunker import Chunk
from .index import top_k


def similar_passages(
    chunk_id: str,
    ids: list[str],
    matrix: np.ndarray,
    chunks: dict[str, Chunk],
    k: int = 5,
    other_books_only: bool = True,
) -> list[tuple[Chunk, float]]:
    """Nearest neighbors of one passage, optionally from other books.

    Parameters
    ----------
    chunk_id : str
        Query passage id.
    ids : list[str]
        Row ids of ``matrix``.
    matrix : np.ndarray
        Unit vectors (n, d).
    chunks : dict[str, Chunk]
        Lookup from id to chunk.
    k : int
        Neighbors to return.
    other_books_only : bool
        Exclude passages from the same book.

    Returns
    -------
    list[tuple[Chunk, float]]
        Neighbors with cosine scores, descending.

    Raises
    ------
    KeyError
        If ``chunk_id`` is unknown.
    """
    row = ids.index(chunk_id)
    scores = matrix @ matrix[row]
    scores[row] = -np.inf
    if other_books_only:
        book = chunks[chunk_id].book_id
        mask = np.array([chunks[i].book_id == book for i in ids])
        scores[mask] = -np.inf
    return [(chunks[ids[i]], float(scores[i])) for i in top_k(scores, k) if np.isfinite(scores[i])]
