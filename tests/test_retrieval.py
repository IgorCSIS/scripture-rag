"""FR-04 retrieval tests: BM25, brute force, HNSW recall, RRF math, LanceDB."""

import numpy as np
import pytest

from scripture_rag.bm25 import BM25Index
from scripture_rag.hybrid import rrf
from scripture_rag.index import BruteForceIndex, FaissHNSWIndex, LanceDBIndex, top_k


def _unit(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    x = rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_top_k_sorted() -> None:
    assert top_k(np.array([0.1, 0.9, 0.5, 0.7]), 2).tolist() == [1, 3]
    assert top_k(np.array([1.0]), 0).size == 0


def test_bm25_toy() -> None:
    idx = BM25Index()
    idx.add(["a", "b", "c"], ["the shepherd and sheep", "still waters run deep", "light and darkness"])
    hits = idx.search("shepherd", 3)
    assert hits[0][0] == "a" and len(hits) == 1
    assert idx.search("the and", 3) == []


def test_brute_force_exact() -> None:
    vecs = np.eye(4, dtype=np.float32)
    idx = BruteForceIndex()
    idx.add(["a", "b", "c", "d"], vecs)
    assert idx.search(vecs[2], 1) == [("c", 1.0)]
    assert len(idx) == 4 and BruteForceIndex().search(vecs[0], 1) == []


def test_rrf_math() -> None:
    fused = dict(rrf([["x", "y"], ["y", "z"]], k=60))
    assert fused["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["x"] == pytest.approx(1 / 61)
    assert max(fused, key=fused.get) == "y"


def test_hnsw_recall_at_10() -> None:
    rng = np.random.default_rng(0)
    data, queries = _unit(rng, 2000, 32), _unit(rng, 50, 32)
    ids = [str(i) for i in range(len(data))]
    exact, approx = BruteForceIndex(), FaissHNSWIndex(32)
    exact.add(ids, data)
    approx.add(ids, data)
    recalls = []
    for q in queries:
        truth = {i for i, _ in exact.search(q, 10)}
        recalls.append(len(truth & {i for i, _ in approx.search(q, 10)}) / 10)
    assert np.mean(recalls) >= 0.9


def test_lancedb_roundtrip(tmp_path) -> None:
    vecs = np.eye(3, dtype=np.float32)
    idx = LanceDBIndex(tmp_path / "db")
    assert idx.search(vecs[0], 1) == []
    idx.add(["a", "b"], vecs[:2])
    idx.add(["c"], vecs[2:])
    assert idx.search(vecs[2], 1)[0][0] == "c"


def test_hybrid_not_worse_than_vector(pipeline) -> None:
    import json

    from scripture_rag.eval import evaluate
    from tests.conftest import FIXTURES

    gold = [json.loads(x) for x in (FIXTURES / "gold_fixture.jsonl").read_text().splitlines()]
    m = evaluate(pipeline, gold)
    assert m["recall@10_hybrid"] >= m["recall@10_vector"]
