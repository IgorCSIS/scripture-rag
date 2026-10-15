# Scorecard

Full 248 item eval set ([eval_set.jsonl](eval_set.jsonl)), strict split: every threshold, weight, embedder and reranker choice is made on the **tune** split only (109 items) and reported unchanged on the **held-out** split (136 items). 3 items (B12, B43, B46) are dropped because their gold text cannot be found in the PDF OCR, so n = 245. Language detection accuracy: 1.000.

Reproduce: `python scripts/elite_eval.py --quick` (cached, about 1 minute) or without `--quick` to include bge-m3 and bge-reranker-v2-m3. Raw data: [ablation.json](ablation.json).

## Before vs after

| Metric | Before (first stress run, 81 items, mixed split) | After, held-out (136 items) | Target | Met |
|---|---|---|---|---|
| Overall pass rate | 0.765 | 0.809 | | |
| Answer accuracy | 0.682 | 0.758 | >= 0.90 | No |
| Accuracy EN / PT / ES | 0.75 / 0.667 / 0.333 | 0.815 / 0.529 / 0.769 | >= 0.85 each | No |
| Refusal recall | 0.929 | 0.933 | | |
| Adversarial answered | 2 | 2 | 0 | No |
| Wrong-evidence (hallucinated) answers | not measured | 11 | 0 | No |
| Citation validity | 1.0 | 1.0 (extractive quotes; validator rejects any citation not in the supplied chunks) | 1.0 | Yes |
| Injection pass | 0.667 | 0.833 (5/6; I02 refused instead of answering, nothing obeyed) | 1.0 | No |

A wrong-evidence answer quotes a real passage from the loaded book, correctly cited, that does not contain the gold answer. No answer contains invented text, but these count as hallucinated answers against the target.

## Held-out by configuration

| Run | n | Answer acc | EN | PT | ES | Refusal recall | Refusal precision | Adversarial answered | Wrong-evidence answers | Injection pass |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 baseline (window chunks, e5-small, global threshold) | 136 | 0.663 | 0.769 | 0.353 | 0.538 | 1.000 | 0.596 | 0 | 9 | 0.833 |
| A8 + verse chunker, normalization, PT/ES translation, expansion, per-language thresholds | 136 | 0.674 | 0.692 | 0.647 | 0.615 | 1.000 | 0.567 | 0 | 5 | 0.833 |
| A9 = A8 + lexical support gate (final) | 136 | 0.758 | 0.815 | 0.529 | 0.769 | 0.933 | 0.696 | 2 | 11 | 0.833 |

## Tune split (for reference, not a result)

| Run | n | Answer acc | EN | PT | ES | Refusal recall | Refusal precision | Adversarial answered | Wrong-evidence answers | Injection pass |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 109 | 0.569 | 0.700 | 0.429 | 0.182 | 1.000 | 0.681 | 0 | 2 | 0.000 |
| A8 | 109 | 0.586 | 0.625 | 1.000 | 0.182 | 1.000 | 0.727 | 0 | 6 | 0.000 |
| A9 final | 109 | 0.707 | 0.750 | 1.000 | 0.364 | 1.000 | 0.810 | 0 | 7 | 0.000 |

Tune counts are small per language (PT 7, ES 11 answerable), so per-language thresholds are noisy.

## Final configuration

`verse|norm=1|mt=1|exp=1|multilingual-e5-small|w=0.5|mmarco-mMiniLMv2-L12-H384-v1`, policy `{"t": {"en": 0.36, "pt": 0.72, "es": 0.36}, "s_min": 0.3}`.

## Improvements tried

| Run | n | Answer acc | EN | PT | ES | Refusal recall | Refusal precision | Adversarial answered | Wrong-evidence answers | Injection pass |
|---|---|---|---|---|---|---|---|---|---|---|
| A5 embedder multilingual-e5-small | 136 | 0.663 | 0.692 | 0.588 | 0.615 | 1.000 | 0.548 | 0 | 4 | 0.833 |
| A5 embedder multilingual-e5-base | 136 | 0.684 | 0.723 | 0.588 | 0.615 | 1.000 | 0.576 | 0 | 5 | 0.833 |
| A6 dense weight 0.0 | 136 | 0.663 | 0.708 | 0.529 | 0.615 | 1.000 | 0.567 | 0 | 6 | 0.833 |
| A6 dense weight 0.25 | 136 | 0.663 | 0.708 | 0.529 | 0.615 | 1.000 | 0.567 | 0 | 6 | 0.833 |
| A6 dense weight 0.5 | 136 | 0.663 | 0.692 | 0.588 | 0.615 | 1.000 | 0.548 | 0 | 4 | 0.833 |
| A6 dense weight 0.75 | 136 | 0.653 | 0.677 | 0.588 | 0.615 | 1.000 | 0.547 | 0 | 4 | 0.833 |
| A6 dense weight 1.0 | 136 | 0.653 | 0.677 | 0.588 | 0.615 | 1.000 | 0.538 | 0 | 3 | 0.833 |
| A7 reranker mmarco-mMiniLMv2-L12-H384-v1 | 136 | 0.663 | 0.692 | 0.588 | 0.615 | 1.000 | 0.548 | 0 | 4 | 0.833 |

- Bigger embedders (multilingual-e5-base, bge-m3): +0.021 held-out accuracy but more wrong-evidence answers; not selected because the choice is made on tune, where they did not win.
- bge-reranker-v2-m3: a CPU run (no GPU on this box) was still scoring candidates after more than 2.5 hours, so it is not in these numbers. Run the full script on a GPU to measure it.
- Embedder fine-tuning on the tune split: not done in this pass.

## Known gaps

- Held-out accuracy is below 0.90 and PT is the weakest language; most PT/ES misses are refusals after translation.
- The support gate trades 2 adversarial answers (A10, A41: Umayyad caliph, a fact outside the texts) for +0.084 accuracy. A8 answers 0 adversarial questions at 0.674 accuracy. Choose A8 if zero answers to unanswerable questions matters more than coverage.
