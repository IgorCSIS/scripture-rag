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

## Real-book evaluation
The real eval set (tune / held-out split plus a synthetic dev set), the shipped configuration and its held-out results are in [eval/SCORECARD.md](eval/SCORECARD.md). Stress and grounding runs of the previous configuration are in [eval/stress_results.md](eval/stress_results.md) and [eval/loaded_phase_results.md](eval/loaded_phase_results.md).
