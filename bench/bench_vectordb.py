"""Vector index benchmark: brute-force NumPy vs FAISS Flat/HNSW/IVF vs LanceDB vs Chroma.

Synthetic unit vectors (d=384, the e5-small width) with a planted cluster
structure, n from 1e3 to 1e6. For each backend we record build time, resident
memory growth, p50/p95 single-query latency and recall@10 against exact search.

Usage::

    python bench/bench_vectordb.py [--max-n 1000000] [--queries 200]

Writes ``docs/benchmarks/vectordb.json`` and ``vectordb_*.png``.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import psutil

OUT = Path("docs/benchmarks")
D, K = 384, 10


def data(n: int, q: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Clustered unit vectors plus queries near random points."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(max(8, n // 500), D)).astype(np.float32)
    x = centers[rng.integers(0, len(centers), n)] + 0.6 * rng.normal(size=(n, D)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    qs = x[rng.integers(0, n, q)] + 0.1 * rng.normal(size=(q, D)).astype(np.float32)
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    return x, qs.astype(np.float32)


def exact(x: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """Ground truth top-k ids."""
    out = []
    for i in range(0, len(qs), 50):
        s = qs[i : i + 50] @ x.T
        out.append(np.argpartition(-s, K, axis=1)[:, :K])
    return np.vstack(out)


def rss() -> float:
    """Resident memory in MB."""
    return psutil.Process().memory_info().rss / 1e6


def numpy_backend(x: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    mat = x.copy()

    def search(q: np.ndarray) -> np.ndarray:
        s = mat @ q
        return np.argpartition(-s, K)[:K]
    return search


def faiss_backend(kind: str) -> Callable[[np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    def build(x: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        import faiss

        if kind == "flat":
            idx = faiss.IndexFlatIP(D)
        elif kind == "hnsw":
            idx = faiss.IndexHNSWFlat(D, 32, faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efConstruction, idx.hnsw.efSearch = 80, 64
        else:
            nlist = max(16, int(4 * np.sqrt(len(x))))
            idx = faiss.IndexIVFFlat(faiss.IndexFlatIP(D), D, nlist, faiss.METRIC_INNER_PRODUCT)
            idx.train(x[np.random.default_rng(1).choice(len(x), min(len(x), 40 * nlist), replace=False)])
            idx.nprobe = max(4, nlist // 32)
        idx.add(x)

        def search(q: np.ndarray) -> np.ndarray:
            return idx.search(q[None, :], K)[1][0]
        return search
    return build


def lancedb_backend(x: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    import lancedb
    import pyarrow as pa

    tmp = tempfile.mkdtemp()
    db = lancedb.connect(tmp)
    tbl = db.create_table("v", pa.table({"id": np.arange(len(x)), "vector": pa.FixedSizeListArray.from_arrays(
        pa.array(x.ravel()), D)}))
    if len(x) >= 10_000:
        tbl.create_index(metric="dot", num_partitions=max(16, int(np.sqrt(len(x)))), num_sub_vectors=48)

    def search(q: np.ndarray) -> np.ndarray:
        r = tbl.search(q).metric("dot").limit(K).nprobes(20).refine_factor(10).select(["id", "_distance"]).to_arrow()
        return r["id"].to_numpy()
    search.cleanup = lambda: shutil.rmtree(tmp, ignore_errors=True)  # type: ignore[attr-defined]
    return search


def chroma_backend(x: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    import chromadb

    client = chromadb.EphemeralClient()
    col = client.create_collection(f"v{time.time_ns()}", metadata={"hnsw:space": "ip"})
    for i in range(0, len(x), 5000):
        col.add(ids=[str(j) for j in range(i, min(i + 5000, len(x)))], embeddings=x[i : i + 5000])

    def search(q: np.ndarray) -> np.ndarray:
        return np.array([int(s) for s in col.query(query_embeddings=[q], n_results=K)["ids"][0]])
    search.cleanup = lambda: client.delete_collection(col.name)  # type: ignore[attr-defined]
    return search


BACKENDS: dict[str, tuple[Callable, int]] = {  # name: (builder, max n)
    "numpy brute force": (numpy_backend, 10**6),
    "faiss Flat": (faiss_backend("flat"), 10**6),
    "faiss HNSW": (faiss_backend("hnsw"), 10**6),
    "faiss IVF": (faiss_backend("ivf"), 10**6),
    "LanceDB (IVF-PQ)": (lancedb_backend, 10**6),
    "Chroma (HNSW)": (chroma_backend, 10**5),
}


def run(max_n: int, nq: int) -> list[dict]:
    """Benchmark every backend at every size."""
    rows = []
    sizes = [n for n in (1_000, 10_000, 100_000, 1_000_000) if n <= max_n]
    for n in sizes:
        x, qs = data(n, nq)
        truth = exact(x, qs)
        for name, (builder, cap) in BACKENDS.items():
            if n > cap:
                continue
            gc.collect()
            m0, t0 = rss(), time.perf_counter()
            try:
                search = builder(x)
            except Exception as exc:  # report and continue
                rows.append({"backend": name, "n": n, "error": str(exc)[:200]})
                continue
            build_s, mem = time.perf_counter() - t0, max(0.0, rss() - m0)
            lat, hits = [], 0
            for i, q in enumerate(qs):
                t = time.perf_counter()
                ids = search(q)
                lat.append(1000 * (time.perf_counter() - t))
                hits += len(set(ids.tolist()) & set(truth[i].tolist()))
            rows.append({"backend": name, "n": n, "build_s": round(build_s, 3), "mem_mb": round(mem, 1),
                         "p50_ms": round(float(np.percentile(lat, 50)), 3),
                         "p95_ms": round(float(np.percentile(lat, 95)), 3), "recall@10": round(hits / (K * nq), 4)})
            print(json.dumps(rows[-1]), flush=True)
            if hasattr(search, "cleanup"):
                search.cleanup()
            del search
        del x
    return rows


def plot(rows: list[dict]) -> None:
    """Latency, build time, memory and recall versus n (log-log)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric, label in (("p50_ms", "p50 query latency (ms)"), ("build_s", "build time (s)"),
                          ("mem_mb", "memory growth (MB)"), ("recall@10", "recall@10")):
        fig, ax = plt.subplots(figsize=(6, 4))
        for name in BACKENDS:
            pts = [(r["n"], r[metric]) for r in rows if r["backend"] == name and metric in r]
            if pts:
                ax.plot(*zip(*pts), marker="o", label=name)
        ax.set_xscale("log")
        if metric != "recall@10":
            ax.set_yscale("log")
        ax.set_xlabel("n vectors (d=384)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / f"vectordb_{metric.replace('@', '_at_')}.png", dpi=120)
        plt.close(fig)


def main() -> None:
    """Run and save."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=1_000_000)
    ap.add_argument("--queries", type=int, default=200)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = run(args.max_n, args.queries)
    (OUT / "vectordb.json").write_text(json.dumps(rows, indent=2))
    plot(rows)


if __name__ == "__main__":
    main()
