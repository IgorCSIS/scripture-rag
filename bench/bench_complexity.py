"""Empirical time complexity benchmarks (FR-12).

Times chunking, brute-force search, FAISS HNSW search, BM25 query, k-means
and PCA for n in {1e3, 2e3, 5e3, 1e4, 2e4, 5e4} on seeded synthetic data,
5 repeats, median. Writes PNGs to docs/benchmarks/ and site/assets/benchmarks/
and a JSON summary with fitted log-log slopes.

Run: ``python bench/bench_complexity.py`` (add ``--quick`` for a smoke run).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from scripture_rag.bm25 import BM25Index  # noqa: E402
from scripture_rag.chunker import Chunker  # noqa: E402
from scripture_rag.index import BruteForceIndex, FaissHNSWIndex  # noqa: E402
from scripture_rag.ingest import Document  # noqa: E402

SIZES = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000]
OUT_DIRS = [Path("docs/benchmarks"), Path("site/assets/benchmarks")]
DIM = 384
VOCAB = [f"w{i}" for i in range(5000)]


def median_time(fn: Callable[[], object], repeats: int) -> float:
    """Median wall time of a callable.

    Parameters
    ----------
    fn : Callable
        Work to time.
    repeats : int
        Number of runs.

    Returns
    -------
    float
        Median seconds.
    """
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def unit_vectors(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    """Random unit vectors.

    Parameters
    ----------
    rng : np.random.Generator
        Seeded generator.
    n : int
        Count.
    d : int
        Dimension.

    Returns
    -------
    np.ndarray
        Array (n, d), float32.
    """
    x = rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def run(sizes: list[int], repeats: int) -> dict[str, list[float]]:
    """Run all benchmarks.

    Parameters
    ----------
    sizes : list[int]
        Values of n.
    repeats : int
        Repeats per point.

    Returns
    -------
    dict[str, list[float]]
        Median seconds per operation, aligned with ``sizes``.
    """
    rng = np.random.default_rng(42)
    res: dict[str, list[float]] = {k: [] for k in ("chunking", "brute_force_search", "hnsw_search",
                                                   "bm25_query", "kmeans", "pca_2d")}
    for n in sizes:
        print(f"n={n}", flush=True)
        words = rng.choice(VOCAB, size=n * 10)  # n = tokens / 10 for chunking
        doc = Document("bench", "bench", [" ".join(words)])
        res["chunking"].append(median_time(lambda: Chunker(400, 60).split(doc), repeats))
        vecs, queries = unit_vectors(rng, n, DIM), unit_vectors(rng, 20, DIM)
        ids = [str(i) for i in range(n)]
        bf = BruteForceIndex()
        bf.add(ids, vecs)
        res["brute_force_search"].append(median_time(lambda: [bf.search(q, 10) for q in queries], repeats) / 20)
        hnsw = FaissHNSWIndex(DIM)
        hnsw.add(ids, vecs)
        res["hnsw_search"].append(median_time(lambda: [hnsw.search(q, 10) for q in queries], repeats) / 20)
        bm = BM25Index()
        texts = [" ".join(rng.choice(VOCAB, size=30)) for _ in range(n)]
        bm.add(ids, texts)
        res["bm25_query"].append(median_time(lambda: bm.search("w1 w2 w3", 10), repeats))
        small = vecs[:, :64]
        res["kmeans"].append(median_time(
            lambda: KMeans(n_clusters=8, n_init=1, max_iter=20, tol=0, random_state=0).fit(small), repeats))
        res["pca_2d"].append(median_time(lambda: PCA(n_components=2, random_state=0).fit_transform(vecs), repeats))
    return res


PREDICTED = {
    "chunking": ("O(N)", lambda n: n),
    "brute_force_search": ("O(n d)", lambda n: n),
    "hnsw_search": ("O(d log n)", lambda n: np.log2(n)),
    "bm25_query": ("O(postings), about O(n)", lambda n: n),
    "kmeans": ("O(n K i d)", lambda n: n),
    "pca_2d": ("O(n d^2)", lambda n: n),
}


def plot(sizes: list[int], res: dict[str, list[float]]) -> dict[str, float]:
    """Save one PNG per operation plus an overview, and return log-log slopes.

    Parameters
    ----------
    sizes : list[int]
        Values of n.
    res : dict[str, list[float]]
        Timings.

    Returns
    -------
    dict[str, float]
        Fitted slope of log(time) vs log(n) per operation.
    """
    n = np.asarray(sizes, dtype=float)
    slopes = {}
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    for name, times in res.items():
        t = np.asarray(times)
        label, f = PREDICTED[name]
        ref = f(n)
        scale = float(np.dot(ref, t) / np.dot(ref, ref))  # least squares fit of c * f(n)
        slopes[name] = float(np.polyfit(np.log(n), np.log(t), 1)[0])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, logscale in zip(axes, (False, True)):
            ax.plot(n, t * 1000, "o-", color="#3b5bdb", label="measured (median)")
            ax.plot(n, scale * ref * 1000, "--", color="#e8590c", label=f"fit c*{label}")
            ax.set_xlabel("n")
            ax.set_ylabel("ms")
            if logscale:
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(f"log-log, slope {slopes[name]:.2f}")
            else:
                ax.set_title(name.replace("_", " "))
            ax.grid(alpha=0.3)
            ax.legend()
        fig.tight_layout()
        for d in OUT_DIRS:
            fig.savefig(d / f"{name}.png", dpi=110)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, times in res.items():
        ax.loglog(n, np.asarray(times) * 1000, "o-", label=name.replace("_", " "))
    ax.set_xlabel("n")
    ax.set_ylabel("ms (median)")
    ax.set_title("Runtime vs n (log-log)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    for d in OUT_DIRS:
        fig.savefig(d / "overview.png", dpi=110)
    plt.close(fig)
    return slopes


def main() -> None:
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small sizes, 1 repeat")
    args = ap.parse_args()
    sizes = SIZES[:3] if args.quick else SIZES
    res = run(sizes, 1 if args.quick else 5)
    slopes = plot(sizes, res)
    summary = {"sizes": sizes, "seconds": res, "loglog_slopes": slopes}
    Path("docs/benchmarks/results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for k, v in slopes.items():
        print(f"{k:20s} slope {v:.2f}")


if __name__ == "__main__":
    main()
