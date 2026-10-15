"""Export the Python pipeline's inputs and decisions so the browser port can be checked against them.

Writes ``data/cache/parity/`` (not committed; about 15 MB): document pages, the Python chunks and
index texts, passage vectors, and per eval item the query vectors, translation, reranker scores of
every candidate and the Python decision under the shipped policy. ``node scripts/parity_check.mjs``
replays the browser code (site/js/rag.js) on the same inputs and writes ``docs/eval/parity.json``.

Usage::

    python scripts/export_parity.py [--set docs/eval/eval_set.jsonl] [--name eval]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from elite_eval import EVAL, Config, Models, Policy, evidence_snippets, grade, load, run_config  # noqa: E402

from scripture_rag.sanitize import clean_query  # noqa: E402
from scripture_rag.translate import detect_language  # noqa: E402

OUT = Path("data/cache/parity")


def final_config() -> tuple[Config, Policy]:
    """Shipped configuration and gate from ablation.json."""
    ab = json.loads((EVAL / "ablation.json").read_text(encoding="utf-8"))
    fc = dict(ab["final_config"])
    fc["rerankers"] = tuple(fc.get("rerankers") or ())
    pol = ab["final_policy"]
    return Config(**fc), Policy(pol["t"], pol["s_min"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=str(EVAL / "eval_set.jsonl"))
    ap.add_argument("--name", default="eval")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    items = [json.loads(x) for x in Path(args.set).read_text(encoding="utf-8").splitlines() if x.strip()]
    corpus = load()
    cfg, pol = final_config()
    models, cache = Models(), {}
    runs = run_config(cfg, corpus, items, models, cache)
    chunks, index_texts, _, _ = cache[(cfg.chunking, cfg.normalize, cfg.bm25_k1, cfg.bm25_b)]
    mat = cache[((cfg.chunking, cfg.normalize), cfg.embedder)]
    rows = grade(items, runs, chunks, pol, evidence_snippets(corpus))
    if not (OUT / "chunks.json").exists() or args.name == "eval":
        (OUT / "pages.json").write_text(json.dumps({d.book_id: d.pages for d in corpus.docs}), encoding="utf-8")
        (OUT / "chunks.json").write_text(json.dumps({
            "chunks": [[c.chunk_id, c.book_id, c.text, c.page_start, c.page_end, c.ref] for c in chunks],
            "index_texts": index_texts, "dim": int(mat.shape[1])}), encoding="utf-8")
        np.asarray(mat, dtype="<f4").tofile(OUT / "pvec.f32")
    out = []
    for it, r, row in zip(items, runs, rows):
        q = clean_query(it["question"])
        lang = detect_language(q)
        qvecs = {qq: [float(x) for x in models.query(cfg.embedder, qq)] for qq in dict.fromkeys([q, r["q_en"]])}
        # Scores for every candidate the Python run reranked (cached by elite_eval).
        cand_texts = {chunks[i].text for i, _, _ in r["ranked"]}
        cand_texts |= set(r.get("cand_texts", []))
        scores = dict(zip(sorted(cand_texts), models.rerank(cfg.reranker, r["q_en"], sorted(cand_texts))))
        out.append({"id": it["id"], "split": it.get("split", ""), "question": it["question"], "q": q, "lang": lang,
                    "q_en": r["q_en"], "qvecs": qvecs, "rerank": scores,
                    "py": {"ranked": [[chunks[i].chunk_id, s, sp] for i, s, sp in r["ranked"]],
                           "answered": row["answered"],
                           "context": [chunks[i].chunk_id for i, s, sp in r["ranked"]
                                       if s >= pol.t.get(lang, pol.t.get("en", 0.5)) and sp >= pol.s_min][:3]
                           if row["answered"] else []}})
    models.save()
    (OUT / f"items_{args.name}.json").write_text(json.dumps({"policy": {"t": pol.t, "s_min": pol.s_min},
                                                             "candidates": cfg.candidates, "rrf_k": cfg.rrf_k,
                                                             "bm25": [cfg.bm25_k1, cfg.bm25_b], "items": out}),
                                                 encoding="utf-8")
    print(f"exported {len(out)} items to {OUT}")


if __name__ == "__main__":
    main()
