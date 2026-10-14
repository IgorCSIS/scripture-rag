# Evaluation

Command: `python -m scripture_rag.eval --corpus <books> --gold <gold.jsonl>` writes `reports/eval.json` and `reports/eval.md` (FR-11). `make eval` runs it on the test fixtures.

## Smoke results (fixture corpus, 12 items)

Corpus: three tiny public-domain excerpts in `tests/fixtures/books` (7 pages total). Gold: 7 answerable, 5 unanswerable (`tests/fixtures/gold_fixture.jsonl`). Default config: hashing embedder, lexical reranker, threshold 0.5, 400/60 chunks.

| System | Recall@10 | MRR |
|---|---|---|
| Vector only | 1.000 | 1.000 |
| BM25 only | 1.000 | 1.000 |
| Hybrid (RRF) | 1.000 | 1.000 |
| Hybrid + rerank | 1.000 | 1.000 |

| Metric | Value |
|---|---|
| Answer rate (answerable) | 1.000 |
| Refusal recall (unanswerable) | 1.000 |
| Citation page accuracy | 1.000 |
| Latency p50 | about 0.3 ms |

These numbers only prove the harness and the gates work: the corpus is so small that every system saturates, so the ablation does not yet separate systems (FR-05 table is a placeholder until the real gold set exists).

## Next
1. Choose 2 to 3 demo books and confirm license evidence (`docs/DATA_SOURCES.md`).
2. Write the 50 item gold set (30 answerable with page, 15 unanswerable, 5 adversarial) plus 5 PT and 5 ES items; Igor reviews.
3. Run with `EMBED_MODEL=intfloat/multilingual-e5-small` and `RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`, calibrate `REFUSAL_THRESHOLD` on the answerable/unanswerable split, and store it in `.env`.
4. Optional RAGAS faithfulness with Ollama as evaluator (not implemented).
