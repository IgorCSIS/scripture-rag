"""Unsupervised theme discovery (FR-09): k-means baseline, optional BERTopic."""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from .text_utils import STOPWORDS


@dataclass
class TopicResult:
    """Cluster assignment and labels.

    Parameters
    ----------
    labels : np.ndarray
        Cluster id per passage (-1 means outlier).
    keywords : dict[int, list[str]]
        Top TF-IDF words per cluster.
    coords : np.ndarray
        2D projection (n, 2) for plotting.
    silhouette : float
        Silhouette score (nan when undefined).
    method : str
        ``kmeans`` or ``bertopic``.
    """

    labels: np.ndarray
    keywords: dict[int, list[str]] = field(default_factory=dict)
    coords: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    silhouette: float = float("nan")
    method: str = "kmeans"

    def __str__(self) -> str:
        """Return a short summary."""
        return f"TopicResult({self.method}, k={len(self.keywords)}, silhouette={self.silhouette:.3f})"


def cluster_keywords(texts: list[str], labels: np.ndarray, top_n: int = 5) -> dict[int, list[str]]:
    """Label clusters with class-based TF-IDF keywords.

    All texts in a cluster are joined into one document, then TF-IDF picks
    words that are frequent in that cluster but rare in the others.

    Parameters
    ----------
    texts : list[str]
        Passage texts.
    labels : np.ndarray
        Cluster ids.
    top_n : int
        Words per cluster.

    Returns
    -------
    dict[int, list[str]]
        Keywords per cluster id.
    """
    ids = sorted(int(c) for c in set(labels.tolist()) if c >= 0)
    if not ids:
        return {}
    docs = [" ".join(t for t, lab in zip(texts, labels) if lab == c) for c in ids]
    vec = TfidfVectorizer(stop_words=sorted(STOPWORDS), token_pattern=r"(?u)\b[^\W\d_]{3,}\b")
    try:
        mat = vec.fit_transform(docs).toarray()
    except ValueError:  # empty vocabulary
        return {c: [] for c in ids}
    vocab = np.array(vec.get_feature_names_out())
    return {c: vocab[np.argsort(-mat[i])[:top_n]].tolist() for i, c in enumerate(ids)}


class TopicModeler:
    """Cluster passage embeddings and project them to 2D.

    Parameters
    ----------
    n_clusters : int or None
        Fixed k; when None, k is chosen by silhouette over a small range.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, n_clusters: int | None = None, seed: int = 42) -> None:
        """Store parameters."""
        self._k = n_clusters
        self._seed = seed

    def __str__(self) -> str:
        """Return a short summary."""
        return f"TopicModeler(k={self._k}, seed={self._seed})"

    def fit(self, vectors: np.ndarray, texts: list[str]) -> TopicResult:
        """Run k-means and label clusters.

        Parameters
        ----------
        vectors : np.ndarray
            Unit embeddings (n, d).
        texts : list[str]
            Passage texts, same order.

        Returns
        -------
        TopicResult
            Labels, keywords, 2D coordinates and silhouette.
        """
        n = vectors.shape[0]
        coords = self.project(vectors)
        if n < 3:
            labels = np.zeros(n, dtype=int)
            return TopicResult(labels, cluster_keywords(texts, labels), coords)
        candidates = [self._k] if self._k else list(range(2, min(10, n - 1) + 1))
        best: tuple[float, np.ndarray] | None = None
        for k in candidates:
            k = max(2, min(k, n - 1))
            labels = KMeans(n_clusters=k, n_init=10, random_state=self._seed).fit_predict(vectors)
            sil = float(silhouette_score(vectors, labels)) if len(set(labels)) > 1 else float("nan")
            if best is None or sil > best[0]:
                best = (sil, labels)
        assert best is not None
        sil, labels = best
        return TopicResult(labels, cluster_keywords(texts, labels), coords, sil, "kmeans")

    def project(self, vectors: np.ndarray) -> np.ndarray:
        """Project to 2D with PCA.

        Parameters
        ----------
        vectors : np.ndarray
            Embeddings (n, d).

        Returns
        -------
        np.ndarray
            Coordinates (n, 2).
        """
        n = vectors.shape[0]
        if n < 2:
            return np.zeros((n, 2))
        out = PCA(n_components=min(2, n, vectors.shape[1]), random_state=self._seed).fit_transform(vectors)
        if out.shape[1] == 1:
            out = np.hstack([out, np.zeros((n, 1))])
        return out

    def fit_bertopic(self, vectors: np.ndarray, texts: list[str]) -> TopicResult:  # pragma: no cover
        """Run BERTopic (UMAP + HDBSCAN) when it is installed.

        Parameters
        ----------
        vectors : np.ndarray
            Precomputed embeddings.
        texts : list[str]
            Passage texts.

        Returns
        -------
        TopicResult
            Topic ids (-1 are outliers) and keywords.
        """
        from bertopic import BERTopic

        model = BERTopic(calculate_probabilities=False)
        labels = np.asarray(model.fit_transform(texts, embeddings=vectors)[0])
        kw = {int(t): [w for w, _ in model.get_topic(t)][:5] for t in set(labels.tolist()) if t >= 0}
        mask = labels >= 0
        sil = float(silhouette_score(vectors[mask], labels[mask])) if len(set(labels[mask])) > 1 else float("nan")
        return TopicResult(labels, kw, self.project(vectors), sil, "bertopic")


def export_theme_map(result: TopicResult, books: list[str], texts: list[str], path: str | Path) -> Path:
    """Write an interactive Plotly theme map as standalone HTML.

    Parameters
    ----------
    result : TopicResult
        Clustering output.
    books : list[str]
        Book id per passage (marker symbol and legend).
    texts : list[str]
        Passage texts for hover (escaped, truncated).
    path : str or Path
        Output HTML path.

    Returns
    -------
    Path
        Written file.
    """
    import plotly.graph_objects as go

    symbols = ["circle", "square", "diamond", "triangle-up", "cross", "x", "star", "hexagon"]
    fig = go.Figure()
    for b_i, book in enumerate(sorted(set(books))):
        idx = [i for i, b in enumerate(books) if b == book]
        topic_names = [
            f"T{result.labels[i]}: {', '.join(result.keywords.get(int(result.labels[i]), []))}" for i in idx
        ]
        fig.add_trace(
            go.Scatter(
                x=result.coords[idx, 0], y=result.coords[idx, 1], mode="markers", name=html.escape(book),
                marker={"symbol": symbols[b_i % len(symbols)], "size": 9,
                        "color": result.labels[idx], "colorscale": "Viridis", "line": {"width": 0.5}},
                text=[f"{t}<br>{html.escape(texts[i][:160])}" for t, i in zip(topic_names, idx)],
                hoverinfo="text",
            )
        )
    fig.update_layout(title="Theme map (clusters are statistical similarity, not doctrinal claims)",
                      legend_title="Book", template="plotly_white")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn")
    return out
