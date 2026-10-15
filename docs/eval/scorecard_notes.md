## Experiments in this pass (tune split, 109 items)

All numbers below are tune split only; they were used for selection. Retrieval recall R@k = share of answerable tune items with a gold passage in the top k after reranking.

| Variant | R@1 | R@5 | Tune acc (in-sample thresholds) | Unsupported |
|---|---|---|---|---|
| 30 candidates (previous) | 0.655 | 0.845 | 0.707 | 7 |
| 50 candidates | 0.638 | 0.897 | 0.776 | 6 |
| 80 candidates + BM25 on original query | 0.621 | 0.879 | 0.759 | 7 |
| 120 candidates + BM25 on original query | 0.638 | 0.879 | 0.638 | 6 |
| 30 candidates + BM25 on original query | 0.672 | 0.862 | 0.724 | 7 |
| 30 candidates, rerank max(original, translated) | 0.603 | 0.845 | 0.707 | 7 |
| 50 + BM25 orig, reranker ms-marco-MiniLM-L-6-v2 (English) | 0.741 | 0.862 | 0.793 | 10 |
| 50 + BM25 orig, reranker ms-marco-MiniLM-L-12-v2 (English) | 0.724 | 0.879 | 0.741 | 11 |
| 50 + BM25 orig, mean of mMiniLM and ms-marco L-6 | | | 0.810 | 7 |

Leave-one-out on tune (thresholds re-tuned without the held-out tune item), 50 candidates + BM25 on original query:

| Gate | mMiniLM acc / unsupported / adversarial answered | Ensemble acc / unsupported / adversarial answered |
|---|---|---|
| Per-language thresholds + support (current gate) | 0.724 / 7 / 1 | 0.759 / 8 / 1 |
| Learned logistic confidence over top-10 scores, support, language (p >= 0.5) | 0.741 / 5 / 0 | 0.776 / 6 / 1 |

Decisions:

- Kept: 50 reranker candidates, BM25 list on the original PT/ES query, query cleaning that strips role tags and instruction preambles (fixes I02). No new model, no extra download.
- Not shipped: the reranker ensemble won on tune, but ms-marco-MiniLM-L-6-v2 is English only. Users can load any Internet Archive book, including PT/ES texts, and no out-of-corpus check was run, so it is reported as A12 and not selected.
- Not shipped: the learned confidence gate. Its LOO gain on 109 items is small (one or two items) and it was not checked on out-of-corpus books.
- Not done: embedder fine-tuning on synthetic queries, NLI entailment verifier, doc2query expansion, BM25 k1/b and RRF k grids. PT stays at 0.529 held-out; PT tune has only 7 answerable items, so PT choices are close to unconstrained by data.
