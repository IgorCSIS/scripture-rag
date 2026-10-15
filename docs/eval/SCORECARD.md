# Scorecard

Eval set [eval_set.jsonl](eval_set.jsonl), strict split: every choice is made on the **tune** split (109 items) and reported once on the **held-out** split (136 items). Dropped (gold not in the PDF OCR): B12, B43, B46. Language detection accuracy: 1.000.

Reproduce: `python scripts/elite_eval.py --quick` then `python scripts/make_scorecard.py`. Raw data: [ablation.json](ablation.json); previous run: [ablation_prev.json](ablation_prev.json).

Unsupported answers = answers to adversarial items plus answers whose cited passages do not hold the gold answer (wrong evidence). Citations are extractive quotes, so citation validity is 1.0 by construction.

## Held-out: previous final vs current final

| Metric | Previous | Current | Target |
|---|---|---|---|
| Answer accuracy | 0.758 | 0.800 | >= 0.90 |
| EN | 0.815 | 0.862 | >= 0.85 |
| PT | 0.529 | 0.529 | >= 0.85 |
| ES | 0.769 | 0.846 | >= 0.85 |
| Refusal recall | 0.933 | 0.900 |  |
| Refusal precision | 0.696 | 0.725 |  |
| Adversarial answered | 2 | 3 | 0 |
| Unsupported answers (incl. adversarial) | 11 | 12 | 0 |
| Injection pass | 0.833 | 1.000 | 1.0 |

Previous: `verse|norm=1|mt=1|exp=1|multilingual-e5-small|w=0.5|mmarco-mMiniLMv2-L12-H384-v1`, policy `{"t": {"en": 0.36, "pt": 0.72, "es": 0.36}, "s_min": 0.3}`.
Current: `verse|norm=1|mt=1|exp=1|multilingual-e5-small|w=0.5|mmarco-mMiniLMv2-L12-H384-v1|k=50|bmo=1`, policy `{"t": {"en": 0.12, "pt": 0.72, "es": 0.16}, "s_min": 0.3}`.

## All steps, tune split (selection data)

| Run | n | Answer acc | EN | PT | ES | Refusal recall | Refusal precision | Adversarial answered | Unsupported answers | Injection pass |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 baseline (window 200, raw, e5-small, mmarco, global threshold) | 109 | 0.569 | 0.700 | 0.429 | 0.182 | 1.000 | 0.681 | 0 | 2 | 0.000 |
| A1 + verse-aware chunker | 109 | 0.466 | 0.550 | 0.571 | 0.091 | 1.000 | 0.640 | 0 | 4 | 0.000 |
| A2 + archaic normalization | 109 | 0.466 | 0.550 | 0.571 | 0.091 | 1.000 | 0.658 | 0 | 6 | 0.000 |
| A3 + PT/ES query translation | 109 | 0.569 | 0.625 | 0.857 | 0.182 | 1.000 | 0.716 | 0 | 6 | 0.000 |
| A4 + archaic query expansion | 109 | 0.569 | 0.625 | 0.857 | 0.182 | 1.000 | 0.716 | 0 | 6 | 0.000 |
| A5 embedder multilingual-e5-small | 109 | 0.569 | 0.625 | 0.857 | 0.182 | 1.000 | 0.716 | 0 | 6 | 0.000 |
| A5 embedder multilingual-e5-base | 109 | 0.569 | 0.625 | 0.857 | 0.182 | 1.000 | 0.716 | 0 | 6 | 0.000 |
| A6 dense weight 0.0 | 109 | 0.569 | 0.625 | 0.857 | 0.182 | 1.000 | 0.716 | 0 | 7 | 0.000 |
| A6 dense weight 0.25 | 109 | 0.569 | 0.625 | 0.857 | 0.182 | 1.000 | 0.716 | 0 | 7 | 0.000 |
| A6 dense weight 0.5 | 109 | 0.569 | 0.625 | 0.857 | 0.182 | 1.000 | 0.716 | 0 | 6 | 0.000 |
| A6 dense weight 0.75 | 109 | 0.534 | 0.575 | 0.857 | 0.182 | 1.000 | 0.676 | 0 | 4 | 0.000 |
| A6 dense weight 1.0 | 109 | 0.534 | 0.575 | 0.857 | 0.182 | 1.000 | 0.676 | 0 | 4 | 0.000 |
| A7 reranker mmarco-mMiniLMv2-L12-H384-v1 | 109 | 0.569 | 0.625 | 0.857 | 0.182 | 1.000 | 0.716 | 0 | 6 | 0.000 |
| A8 per-language thresholds | 109 | 0.586 | 0.625 | 1.000 | 0.182 | 1.000 | 0.727 | 0 | 6 | 0.000 |
| A9 per-language thresholds + lexical support gate | 109 | 0.707 | 0.750 | 1.000 | 0.364 | 1.000 | 0.810 | 0 | 7 | 0.000 |
| A10 + 50 reranker candidates | 109 | 0.776 | 0.825 | 1.000 | 0.455 | 1.000 | 0.868 | 0 | 6 | 0.000 |
| A11 + BM25 on the original PT/ES query | 109 | 0.776 | 0.825 | 1.000 | 0.455 | 1.000 | 0.868 | 0 | 6 | 0.000 |
| A12 reranker ensemble mMiniLM + ms-marco-MiniLM-L-6 (English only; not shipped, see SCORECARD) | 109 | 0.810 | 0.850 | 1.000 | 0.545 | 1.000 | 0.939 | 0 | 7 | 0.000 |

## All steps, held-out split (reported, never used for selection)

| Run | n | Answer acc | EN | PT | ES | Refusal recall | Refusal precision | Adversarial answered | Unsupported answers | Injection pass |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 baseline (window 200, raw, e5-small, mmarco, global threshold) | 136 | 0.663 | 0.769 | 0.353 | 0.538 | 1.000 | 0.589 | 0 | 9 | 1.000 |
| A1 + verse-aware chunker | 136 | 0.589 | 0.708 | 0.353 | 0.308 | 1.000 | 0.507 | 0 | 5 | 1.000 |
| A2 + archaic normalization | 136 | 0.589 | 0.692 | 0.353 | 0.385 | 1.000 | 0.500 | 0 | 4 | 1.000 |
| A3 + PT/ES query translation | 136 | 0.663 | 0.692 | 0.588 | 0.615 | 1.000 | 0.541 | 0 | 4 | 1.000 |
| A4 + archaic query expansion | 136 | 0.663 | 0.692 | 0.588 | 0.615 | 1.000 | 0.541 | 0 | 4 | 1.000 |
| A5 embedder multilingual-e5-small | 136 | 0.663 | 0.692 | 0.588 | 0.615 | 1.000 | 0.541 | 0 | 4 | 1.000 |
| A5 embedder multilingual-e5-base | 136 | 0.684 | 0.723 | 0.588 | 0.615 | 1.000 | 0.569 | 0 | 5 | 1.000 |
| A6 dense weight 0.0 | 136 | 0.663 | 0.708 | 0.529 | 0.615 | 1.000 | 0.559 | 0 | 6 | 1.000 |
| A6 dense weight 0.25 | 136 | 0.663 | 0.708 | 0.529 | 0.615 | 1.000 | 0.559 | 0 | 6 | 1.000 |
| A6 dense weight 0.5 | 136 | 0.663 | 0.692 | 0.588 | 0.615 | 1.000 | 0.541 | 0 | 4 | 1.000 |
| A6 dense weight 0.75 | 136 | 0.653 | 0.677 | 0.588 | 0.615 | 1.000 | 0.540 | 0 | 4 | 1.000 |
| A6 dense weight 1.0 | 136 | 0.653 | 0.677 | 0.588 | 0.615 | 1.000 | 0.531 | 0 | 3 | 1.000 |
| A7 reranker mmarco-mMiniLMv2-L12-H384-v1 | 136 | 0.663 | 0.692 | 0.588 | 0.615 | 1.000 | 0.541 | 0 | 4 | 1.000 |
| A8 per-language thresholds | 136 | 0.674 | 0.692 | 0.647 | 0.615 | 1.000 | 0.559 | 0 | 5 | 1.000 |
| A9 per-language thresholds + lexical support gate | 136 | 0.758 | 0.815 | 0.529 | 0.769 | 0.933 | 0.689 | 2 | 11 | 1.000 |
| A10 + 50 reranker candidates | 136 | 0.800 | 0.862 | 0.529 | 0.846 | 0.900 | 0.725 | 3 | 12 | 1.000 |
| A11 + BM25 on the original PT/ES query | 136 | 0.800 | 0.862 | 0.529 | 0.846 | 0.900 | 0.725 | 3 | 12 | 1.000 |
| A12 reranker ensemble mMiniLM + ms-marco-MiniLM-L-6 (English only; not shipped, see SCORECARD) | 136 | 0.811 | 0.877 | 0.588 | 0.769 | 1.000 | 0.846 | 0 | 12 | 1.000 |

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

