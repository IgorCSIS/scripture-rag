"""Vector indexes behind one interface: brute force, FAISS HNSW, LanceDB."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class VectorIndex(ABC):
    """Interface for inner product search over unit vectors."""

    @abstractmethod
    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        """Add vectors.

        Parameters
        ----------
        ids : list[str]
            Chunk ids, one per row.
        vectors : np.ndarray
            Unit vectors of shape (n, d).
        """

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Return the k most similar ids.

        Parameters
        ----------
        query : np.ndarray
            Unit vector of shape (d,).
        k : int
            Number of results.

        Returns
        -------
        list[tuple[str, float]]
            (chunk_id, cosine score) sorted by score descending.
        """

    def __len__(self) -> int:
        """Return the number of stored vectors."""
        return len(getattr(self, "_ids", []))

    def __str__(self) -> str:
        """Return the class name and size."""
        return f"{type(self).__name__}(n={len(self)})"


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores, sorted descending.

    Uses ``argpartition`` (linear selection) then sorts only k items,
    which gives O(n + k log k).

    Parameters
    ----------
    scores : np.ndarray
        Score vector.
    k : int
        Count.

    Returns
    -------
    np.ndarray
        Index array of length min(k, n).
    """
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part], kind="stable")]


class BruteForceIndex(VectorIndex):
    """Exact search with one matrix-vector product."""

    def __init__(self) -> None:
        """Create an empty index."""
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        """Append vectors (see ``VectorIndex.add``)."""
        vectors = np.asarray(vectors, dtype=np.float32)
        self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])
        self._ids.extend(ids)

    @property
    def matrix(self) -> np.ndarray:
        """Stored vectors, shape (n, d)."""
        return self._matrix if self._matrix is not None else np.zeros((0, 0), np.float32)

    @property
    def ids(self) -> list[str]:
        """Stored chunk ids in row order."""
        return list(self._ids)

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Exact top-k by inner product (see ``VectorIndex.search``)."""
        if self._matrix is None:
            return []
        scores = self._matrix @ np.asarray(query, dtype=np.float32)
        return [(self._ids[i], float(scores[i])) for i in top_k(scores, k)]


class FaissHNSWIndex(VectorIndex):
    """Approximate search with a FAISS HNSW graph.

    Parameters
    ----------
    dim : int
        Vector dimension.
    m : int
        Graph neighbors per node.
    ef_search : int
        Search beam width; higher means better recall and slower search.
    """

    def __init__(self, dim: int, m: int = 32, ef_search: int = 64) -> None:
        """Create an empty HNSW index."""
        import faiss

        self._index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efSearch = ef_search
        self._ids: list[str] = []

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        """Insert vectors into the graph (see ``VectorIndex.add``)."""
        self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self._ids.extend(ids)

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Approximate top-k (see ``VectorIndex.search``)."""
        if not self._ids:
            return []
        q = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        scores, idx = self._index.search(q, min(k, len(self._ids)))
        return [(self._ids[i], float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]


class LanceDBIndex(VectorIndex):
    """On-disk vector table using LanceDB.

    Parameters
    ----------
    path : str or Path
        Database directory.
    table : str
        Table name.
    """

    def __init__(self, path: str | Path, table: str = "chunks") -> None:
        """Open or create the database directory."""
        import lancedb

        self._db = lancedb.connect(str(path))
        self._table_name = table
        self._table = None
        self._ids: list[str] = []

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        """Write rows to the table (see ``VectorIndex.add``)."""
        rows = [{"chunk_id": i, "vector": v.astype(np.float32).tolist()} for i, v in zip(ids, vectors)]
        if self._table is None:
            self._table = self._db.create_table(self._table_name, data=rows, mode="overwrite")
        else:
            self._table.add(rows)
        self._ids.extend(ids)

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Cosine top-k from LanceDB (see ``VectorIndex.search``)."""
        if self._table is None:
            return []
        hits = self._table.search(np.asarray(query, np.float32).tolist()).metric("cosine").limit(k).to_list()
        return [(h["chunk_id"], 1.0 - float(h["_distance"])) for h in hits]
