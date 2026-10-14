"""Evaluation harness (FR-11): ``python -m scripture_rag.eval``.

Gold file format, one JSON object per line::

    {"question": "...", "answerable": true, "book_id": "psalms", "page": 1}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .config import Settings
from .corpus import load_folder
from .pipeline import RagPipeline


def _hit(pipeline: RagPipeline, ids: list[str], book: str, page: int) -> int | None:
    """Return the 1-based rank of the first chunk covering book/page, or None."""
    for rank, cid in enumerate(ids, start=1):
        c = pipeline.chunks[cid]
        if c.book_id == book and c.page_start <= page <= c.page_end:
            return rank
    return None


def evaluate(pipeline: RagPipeline, gold: list[dict[str, Any]], k: int = 10) -> dict[str, Any]:
    """Compute retrieval, rerank ablation, refusal and latency metrics.

    Parameters
    ----------
    pipeline : RagPipeline
        Loaded pipeline.
    gold : list[dict]
        Gold items.
    k : int
        Recall cutoff.

    Returns
    -------
    dict
        Metric name to value.
    """
    r = pipeline.retriever
    systems = {
        "vector": lambda q: r.vector_only(q, k),
        "bm25": lambda q: r.bm25_only(q, k),
        "hybrid": lambda q: r.retrieve(q, k),
        "hybrid+rerank": lambda q: [
            c.chunk_id for c, _ in pipeline.reranker.rerank(q, [pipeline.chunks[i] for i in r.retrieve(q, 50)], k)
        ],
    }
    answerable = [g for g in gold if g["answerable"]]
    unanswerable = [g for g in gold if not g["answerable"]]
    out: dict[str, Any] = {"n_answerable": len(answerable), "n_unanswerable": len(unanswerable), "k": k}
    for name, fn in systems.items():
        ranks = [_hit(pipeline, fn(g["question"]), g["book_id"], g["page"]) for g in answerable]
        n = max(len(ranks), 1)
        out[f"recall@{k}_{name}"] = sum(rk is not None for rk in ranks) / n
        out[f"mrr_{name}"] = sum(1 / rk for rk in ranks if rk) / n
    latencies, answered, refused_ok, cited_ok, cited_total = [], 0, 0, 0, 0
    for g in gold:
        t0 = time.perf_counter()
        ans = pipeline.ask(g["question"])
        latencies.append(time.perf_counter() - t0)
        if g["answerable"] and not ans.refused:
            answered += 1
            cited_total += 1
            cited_ok += any(
                c.book_id == g["book_id"] and c.page_start <= g["page"] <= c.page_end for c in ans.citations
            )
        if not g["answerable"] and ans.refused:
            refused_ok += 1
    latencies.sort()
    out["answer_rate"] = answered / max(len(answerable), 1)
    out["refusal_recall"] = refused_ok / max(len(unanswerable), 1)
    out["citation_page_accuracy"] = cited_ok / max(cited_total, 1)
    out["latency_p50_ms"] = 1000 * latencies[len(latencies) // 2] if latencies else 0.0
    out["latency_p95_ms"] = 1000 * latencies[int(0.95 * (len(latencies) - 1))] if latencies else 0.0
    return out


def to_markdown(metrics: dict[str, Any]) -> str:
    """Render metrics as a Markdown table.

    Parameters
    ----------
    metrics : dict
        Output of ``evaluate``.

    Returns
    -------
    str
        Markdown report.
    """
    rows = "\n".join(f"| {k} | {v:.3f} |" if isinstance(v, float) else f"| {k} | {v} |" for k, v in metrics.items())
    return "# Evaluation report\n\n| Metric | Value |\n|---|---|\n" + rows + "\n"


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Command line entry point.

    Parameters
    ----------
    argv : list[str] or None
        Arguments (defaults to ``sys.argv``).

    Returns
    -------
    dict
        Metrics written to the reports folder.
    """
    ap = argparse.ArgumentParser(description="Run the Scripture RAG evaluation")
    ap.add_argument("--corpus", default="data/demo")
    ap.add_argument("--gold", default="data/gold/gold.jsonl")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args(argv)
    pipeline = RagPipeline(Settings.from_env())
    if not load_folder(pipeline, args.corpus):
        raise SystemExit(f"No .pdf or .txt books in {args.corpus}")
    gold = [json.loads(line) for line in Path(args.gold).read_text(encoding="utf-8").splitlines() if line.strip()]
    metrics = evaluate(pipeline, gold)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "eval.md").write_text(to_markdown(metrics), encoding="utf-8")
    print(to_markdown(metrics))
    return metrics


__all__ = ["evaluate", "main", "to_markdown"]

if __name__ == "__main__":
    main()
