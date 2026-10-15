## Round 2 (selection on tune + synthetic dev, held-out scored once on the final configuration)

Raw data: [round2.json](round2.json) (every variant below re-tunes the per-language thresholds and support on tune + dev), PT misses: [round2_pt_misses.json](round2_pt_misses.json).

Base for round 2 was the round 1 final retrieval with two parity fixes that the browser needs: every book is normalized (as in RagPipeline), and ties in top-k and in the verse aligner break by index. On tune + dev that base scored tune 0.729, dev 0.641 (dev PT 0.545), 14 unsupported, objective 1.0323 (from the round 2 run log; round2.json holds the rows on top of the final BM25/RRF).

| Experiment | Tune acc | Dev acc | Dev PT | Unsupported (tune + dev) | Adversarial answered | Objective | Shipped |
|---|---|---|---|---|---|---|---|
| Gate tuned on tune only (round 1 practice) | 0.729 | 0.603 | 0.523 | 16 | 1 | (not comparable) | no |
| Gate re-tuned on tune + dev | 0.729 | 0.641 | 0.545 | 14 | 0 | 1.0323 | yes |
| + BM25 k1=1.2 (grid k1 0.9 to 2.0, b 0.3 to 0.9) | 0.729 | 0.695 | 0.682 | 18 | 0 | 1.0343 | yes |
| + RRF k=20 (grid 10, 20, 30, 60, 100) | 0.729 | 0.702 | 0.682 | 17 | 0 | 1.0667 | yes |
| + key name/number filter on ranked passages | 0.712 | 0.634 | 0.636 | 20 | 0 | 0.9043 | no |
| + NLI filter, mDeBERTa-v3-base-xnli (MIT), top 3, p >= 0.2 | 0.729 | 0.573 | 0.455 | 11 | 0 | 0.9990 | no |
| + NLI filter, p >= 0.5 | 0.695 | 0.565 | 0.455 | 9 | 0 | 1.0090 | no |
| + support also on the untranslated query | 0.729 | 0.702 | 0.682 | 17 | 0 | 1.0667 | no (no change) |

Notes:

- NLI removes wrong-evidence answers (17 to 9) but costs more correct answers than it saves, so it was not shipped and its browser latency was not measured.
- Dense search on the untranslated query was already part of the pipeline (both the original and translated query are embedded). PT diagnosis: in 2 of 20 dev PT misses MarianMT returned the Portuguese question unchanged; most other misses have the right verse ranked first but lexical support just under 0.3 after paraphrasing translation. No better Apache/MIT PT to EN Marian checkpoint was found on the Hub, so the translator is unchanged.
- No learned or fine-tuned component was shipped this round (embedder fine-tuning skipped), so the out-of-corpus guard was not triggered.
- Held-out went from 0.800 to 0.789 and unsupported answers from 12 to 17: the dev-informed gate (PT threshold 0.04, i.e. the support check alone) did not transfer to held-out. Reported as measured; held-out was not used to change anything.

## Shipped: round 1 gate rollback

After the round 2 held-out result above, the shipped gate was rolled back to the round 1 final policy: thresholds en 0.12, pt 0.72, es 0.16, s_min 0.3, 50 candidates, BM25 on the original query, default BM25 (k1 1.5, b 0.75) and RRF (k 60). The round 2 parity fixes (every book normalized, deterministic tie breaks) are kept. This rollback was chosen after seeing the held-out scores, so the held-out numbers for it are not an untouched estimate. Against the round 1 run the only changes are the parity fixes; on held-out they move PT from 0.529 to 0.471 and refusal precision from 0.725 to 0.707, other held-out metrics are unchanged. No new test set was added. Round 2 raw data is kept in [ablation_round2.json](ablation_round2.json); the round 1 run is [ablation_prev.json](ablation_prev.json).

## Browser parity

The browser pipeline ([site/js/rag.js](../../site/js/rag.js)) is a port of the Python one with the same defaults. [parity.json](parity.json): same chunks (6538/6538), same index texts, and identical answer/refuse decisions and citations on all 248 eval items and 175 dev items when fed the Python model outputs. [parity_models.json](parity_models.json): with the browser models themselves (transformers.js ONNX, run in Node) decisions agree on the eval set as listed there; differences come from beam-search translation and float noise. CI runs a committed replay fixture (site/tests/parity.test.mjs).
