"""Unsupervised discovery tied to search (FR-19).

* ``cluster_hits``: group Internet Archive results by embeddings of title,
  description and subject, labeled with TF-IDF keywords. O(r * d) to embed r
  results, O(K * r * k * d * i) for the k-means sweep (K values of k, i
  iterations), small because r <= 50.
* ``related_query`` and ``rank_related``: recommend books near the centroid of
  the loaded library. Centroid O(n * d); ranking O(r * d + r log r).
* ``book_tags``: tag each loaded book with the topic clusters most of its
  passages fall in. O(n) after clustering.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .embed import Embedder
from .ia_search import SearchHit
from .text_utils import tokenize
from .topics import TopicModeler, TopicResult


def hit_text(h: SearchHit) -> str:
    """Text used to embed a search result."""
    return f"{h.title}. {h.subject}. {h.description}"


@dataclass(frozen=True)
class HitGroup:
    """One labeled group of search results."""

    label: str
    hits: list[SearchHit]


def cluster_hits(hits: list[SearchHit], embedder: Embedder, k: int | None = None) -> list[HitGroup]:
    """Cluster search results into labeled groups, largest first.

    Parameters
    ----------
    hits : list[SearchHit]
        Search results.
    embedder : Embedder
        Passage embedder.
    k : int or None
        Fixed number of groups; None picks k by silhouette.

    Returns
    -------
    list[HitGroup]
        Groups with ``keyword / keyword / keyword`` labels.
    """
    if not hits:
        return []
    texts = [hit_text(h) for h in hits]
    if len(hits) < 4:
        return [HitGroup(" / ".join(_top_terms(texts, 3)) or "results", list(hits))]
    res = TopicModeler(k).fit(embedder.encode(texts), texts)
    groups = []
    for c in sorted(set(res.labels.tolist()), key=lambda c: -int((res.labels == c).sum())):
        members = [h for h, lab in zip(hits, res.labels) if lab == c]
        groups.append(HitGroup(" / ".join(res.keywords.get(c, [])[:3]) or f"group {c}", members))
    return groups


def _top_terms(texts: list[str], n: int) -> list[str]:
    """Most frequent content words."""
    return [w for w, _ in Counter(t for x in texts for t in tokenize(x) if len(t) > 3).most_common(n)]


def library_centroid(vectors: np.ndarray) -> np.ndarray:
    """Unit mean of passage vectors (n, d) -> (d,)."""
    c = vectors.mean(axis=0)
    return c / (np.linalg.norm(c) or 1.0)


def related_query(texts: list[str], n_terms: int = 4) -> str:
    """Keywords that describe the loaded library, for a related-books search."""
    return " ".join(_top_terms(texts, n_terms))


def rank_related(hits: list[SearchHit], embedder: Embedder, centroid: np.ndarray,
                 exclude: set[str] | None = None, top_n: int = 10) -> list[tuple[SearchHit, float]]:
    """Rank results by cosine similarity to the library centroid.

    Parameters
    ----------
    hits : list[SearchHit]
        Candidates.
    embedder : Embedder
        Same embedder as the library.
    centroid : np.ndarray
        Library centroid.
    exclude : set[str] or None
        Identifiers already loaded.
    top_n : int
        Number to return.

    Returns
    -------
    list[tuple[SearchHit, float]]
        Best matches with cosine scores.
    """
    pool = [h for h in hits if h.identifier not in (exclude or set())]
    if not pool:
        return []
    sims = embedder.encode([hit_text(h) for h in pool]) @ centroid
    order = np.argsort(-sims)[:top_n]
    return [(pool[i], float(sims[i])) for i in order]


def book_tags(book_ids: list[str], result: TopicResult, top_n: int = 2) -> dict[str, list[str]]:
    """Tag each book with the labels of the clusters most of its passages fall in.

    Parameters
    ----------
    book_ids : list[str]
        Book id per passage, aligned with ``result.labels``.
    result : TopicResult
        Clustering of the library.
    top_n : int
        Tags per book.

    Returns
    -------
    dict[str, list[str]]
        Book id to tag labels.
    """
    per: dict[str, Counter[int]] = {}
    for b, lab in zip(book_ids, result.labels.tolist()):
        per.setdefault(b, Counter())[lab] += 1
    return {b: [" / ".join(result.keywords.get(c, [])[:2]) for c, _ in cnt.most_common(top_n) if c >= 0]
            for b, cnt in per.items()}
