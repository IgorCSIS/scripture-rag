# Retrieval Tuning Research: Browser-Side Extractive Citation RAG

Scope: verse-level chunks, BM25 + multilingual-e5-small weighted RRF, MarianMT PT/ES to EN query translation, mMiniLM cross-encoder reranker, per-language thresholds, abstaining support gate. Corpus: archaic English (KJV, Rodwell Quran, Rodkinson Talmud, Sale commentary). Known weak points: PT accuracy 0.53, wrong-evidence answers.

Convention: every fact has a link. Items marked **[UNVERIFIED]** are reasoning or expectations not directly measured in a cited source. Expected gains for this corpus are estimates unless stated.

## Ranked action list

| # | Action | Expected gain | Cost | Browser-feasible |
|---|---|---|---|---|
| 1 | Build a labeled dev set (PT/ES/EN, answerable and unanswerable, with gold verse IDs) and tune everything on it | Prerequisite for all others; prevents overfitting thresholds | Low to medium (human labeling of ~200 to 500 queries) | N/A (offline) |
| 2 | Replace single-score thresholds with a learned confidence function over the sorted reranker score vector (linear regression), per language | Better abstention curve; sourced +~10 nAUC points vs best heuristic | Very low (a few dozen weights in JS) | Yes |
| 3 | Domain-adapt multilingual-e5-small with synthetic queries (GPL style: generate queries, mine hard negatives, cross-encoder pseudo-labels, MarginMSE), including PT/ES queries targeting EN verses | Sourced +2.7 to +9.3 nDCG@10 on BEIR domains; biggest expected lever for PT | Medium (offline GPU hours preferred; CPU possible but slow) | Yes at inference (same model size, re-export ONNX) |
| 4 | Run the dense retriever on the native PT/ES query in addition to the translated query, and fuse both into RRF | Removes MT error as single point of failure; expected PT gain **[UNVERIFIED]** | Low | Yes |
| 5 | Add an answer-containment / entailment verifier step for wrong-evidence cases (multilingual NLI on query-as-claim vs verse, plus lexical span checks) | Targets wrong-evidence failure mode directly | Medium (NLI model ~280M params; latency) | Yes, with quantized ONNX |
| 6 | Tune fusion: RRF k, per-retriever weights, BM25 k1/b, archaic-English normalization (thee/thou, -eth/-est) | Small to moderate; RRF k is not critical per source | Low | Yes |
| 7 | Fine-tune or swap the reranker; candidates measured below | Moderate; mMiniLM trails larger multilingual rerankers on BEIR | Medium to high (size/latency) | Partly (278M to 568M is heavy) |
| 8 | doc2query style index expansion of verses (offline generated questions appended to BM25 field), with filtering | Moderate for lexical gap with archaic text **[UNVERIFIED for this corpus]** | Medium offline, zero runtime cost | Yes (index only) |
| 9 | Context windows: keep verse-level units but score with neighbor context (verse plus +/-1) for the reranker / verifier | Small to moderate on multi-verse answers **[UNVERIFIED]** | Low | Yes |
| 10 | Evaluate a larger embedder (EmbeddingGemma-300m or gte-multilingual-base) only if 3 is insufficient | Possibly large on MMTEB, but 2.5x params | Medium (download size, memory) | Yes, borderline |

## 1. Embedder fine-tuning on synthetic queries

Evidence
- GPL (generate queries with T5, mine negatives with a dense retriever, pseudo-label with a cross-encoder, train with MarginMSE) beats the zero-shot MS MARCO model by up to 9.3 nDCG@10 and QGen by up to 4.5, avg +2.7 over QGen on six BEIR domains; TSDAE + GPL reaches 52.9 avg. https://arxiv.org/abs/2112.07577 . Code: https://github.com/UKPLab/gpl
- GPL injection improved TAS-B on 16/18 BEIR datasets (avg 0.415 to 0.459 per search summary of the paper). https://arxiv.org/pdf/2205.11498
- Promptagator: few-shot (0 to 8 examples) LLM query generation trains a 110M dual encoder to 47.8 avg BEIR nDCG@10; with reranking (Promptagator++) 52.8. https://arxiv.org/abs/2209.11755
- InPars: few-shot (3 examples) LLM query generation for IR training data. https://dl.acm.org/doi/10.1145/3477495.3531863
- Sentence-Transformers provides `mine_hard_negatives` (margin, num_negatives, sampling strategy) producing triplets for MultipleNegativesRankingLoss or CachedMultipleNegativesRankingLoss (cached variant allows larger effective batches). https://www.sbert.net/docs/package_reference/util/hard_negatives.html
- A domain fine-tune of EmbeddingGemma on medical retrieval raised NDCG@10 from 0.834 to 0.886 in the HF blog example; the same table lists multilingual-e5-small at 0.685. https://huggingface.co/blog/embeddinggemma

Actionable recipe
1. For each verse, generate 2 to 4 questions in EN, PT and ES (any local or API LLM offline; generation is offline so browser constraints do not apply). Include paraphrased, modern-vocabulary questions so the model learns archaic to modern mapping.
2. Filter: keep a query only if a cross-encoder ranks its source verse in the top-k (doc2query-- style filtering, see section 5). Hold out whole books to avoid leakage.
3. Mine hard negatives with the current e5-small using a margin so near-duplicates (parallel verses, Sale commentary quoting the Quran) are not labeled negative.
4. Train with MNRL (or GPL MarginMSE if a good cross-encoder teacher exists). Keep the `query: ` / `passage: ` prefixes e5 expects.
5. Export to ONNX and quantize with Optimum; same runtime footprint as today.

CPU feasibility **[UNVERIFIED estimate]**: a 118M encoder with MNRL at batch 32 and seq len 64 on a modern 8-core CPU is roughly hours for ~50k pairs, not days; GPL's default of many steps is better done on a single rented GPU. Query generation with an LLM is the dominant cost.

Risk: synthetic queries mirror the generator's style; mix in the real labeled dev queries only for evaluation, never training.

## 2. Small multilingual embedders and rerankers for the browser

Embedders

| Model | Params | Dims | Ctx | Langs | Notes | Source |
|---|---|---|---|---|---|---|
| intfloat/multilingual-e5-small (current) | 118M | 384 | 512 | 94 | ONNX at Xenova/multilingual-e5-small | https://huggingface.co/Xenova/multilingual-e5-small |
| Alibaba-NLP/gte-multilingual-base | 305M | 768 (summary table lists 1536 in one source; verify) | 8192 | 75 | ONNX at onnx-community | https://huggingface.co/onnx-community/gte-multilingual-base |
| google/embeddinggemma-300m | 308M | 768, MRL to 512/256/128 | 2048 | 100+ | MMTEB multilingual v2 mean 61.15; under 200 MB RAM quantized; Gemma terms of use, not Apache | https://huggingface.co/blog/embeddinggemma |
| Snowflake/snowflake-arctic-embed-m-v2.0 | 305M | 768 | 8192 | multilingual | 0.847 NDCG@10 on the HF medical example vs 0.685 for e5-small | https://huggingface.co/blog/embeddinggemma |

Licenses: e5 MIT, gte Apache-2.0, Arctic Apache-2.0 **[UNVERIFIED here, check model cards]**; EmbeddingGemma uses Gemma terms (HF blog).

Rerankers (numbers from the Jina model card)

| Model | Params | MKQA nDCG@10 (26 langs) | BEIR nDCG@10 | License |
|---|---|---|---|---|
| mmarco-mMiniLMv2-L12-H384-v1 (current) | 118M | 53.37 | 45.40 | Apache-2.0 |
| jina-reranker-v2-base-multilingual | 278M | 54.83 | 53.17 | CC-BY-NC-4.0 (non-commercial) |
| bge-reranker-v2-m3 | 568M | 54.17 | 53.65 | Apache-2.0 |

Sources: https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual , https://huggingface.co/BAAI/bge-reranker-v2-m3
- mMiniLM ONNX: int8 is ~118 MB vs fp32 ~470 MB; int8 reported ~equal ranking quality and ~1.9x faster; ~30 ms per 10 pairs on desktop CPU (Node). https://huggingface.co/SugoLabs/mmarco-mMiniLMv2-L12-H384-v1 , official ONNX variants: https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1/tree/main/onnx
- Takeaway: on multilingual (MKQA) the small reranker is within ~1.5 points of the big ones; the large BEIR gap suggests English domain transfer is its weakness. Fine-tuning mMiniLM on the synthetic pairs from section 1 is cheaper than switching to a 568M model. bge-reranker-v2-m3 is Apache but heavy for a browser **[UNVERIFIED latency]**.

## 3. Abstention, answerability and calibration

Evidence
- Gisserot-Boukhlef et al. (TMLR): black-box abstention on reranker scores. Heuristics: max score, std, top1-top2 gap. A ridge linear regression over the sorted top-10 score vector (u_lin) beats the best heuristic by almost 10 nAUC points (mAP) on average; it outperforms std with only ~38 labeled reference queries on average; overhead ~1.2% of scoring time. Reference-based calibration transfers poorly across domains, so calibrate in-domain and per language. https://arxiv.org/abs/2402.12997 , code MIT: https://github.com/artefactory/abstention-reranker
- Calibration of reading systems at scale (Platt / temperature scaling of verifier scores). https://arxiv.org/abs/2203.10623
- HALT-RAG: calibrated NLI ensembles with abstention for hallucination detection. https://arxiv.org/abs/2509.07475
- NLI-based factuality/attribution metrics are imperfect and must be validated on held-out labeled data. https://aclanthology.org/2025.findings-acl.1175.pdf
- Browser-ready multilingual NLI: Xenova/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 (ONNX, quantized variants incl. q4f16). https://huggingface.co/Xenova/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
- SQuAD 2.0 framed unanswerable questions as a first-class task. https://arxiv.org/abs/1806.03822

Actionable
1. Replace per-language scalar thresholds with u_lin over sorted reranker scores plus features (BM25 rank of top hit, dense cosine, agreement between native and translated query). Fit per language on the dev set; pick tau for a target precision.
2. Wrong-evidence gate **[UNVERIFIED design]**: convert the question into a declarative hypothesis template where possible and run NLI with verse as premise; require entailment probability above a calibrated cut. Run only on the top 1 to 3 candidates to bound latency.
3. Cheap span checks: require overlap of the question's key entity/number (after archaic normalization) with the cited verse for factoid questions.
4. Report risk-coverage curves, not single accuracy.

## 4. Multilingual retrieval with vs without translation

Evidence
- ColBERT-X: multilingual dense retrieval outperformed BM25 over query translation in general on CLEF/HC4. https://arxiv.org/abs/2201.08471
- Machine-translated queries roughly halved MRR@10 vs native queries in a financial CLIR study (0.0755 vs 0.1535). https://research.nii.ac.jp/ntcir/workshop/OnlineProceedings18/pdf/ai_cup/03-AICUP-AICUP-ChiuY.pdf
- Query vs document translation tradeoffs; translating into English is generally easier than into morphologically rich languages. https://arxiv.org/abs/2111.05988
- MIRACL BM25+mDPR hybrid (avg 0.581) beats either alone (BM25 alone not shown here; mDPR 0.421). https://castorini.github.io/pyserini/2cr/miracl.html

Actionable: keep MarianMT for BM25 (BM25 needs English tokens) but query the dense model with the original PT/ES text as well as the translation, and the reranker with both; fuse. Train PT/ES synthetic queries into the embedder (section 1). Expected to address PT 0.53 **[UNVERIFIED magnitude]**.

## 5. Hybrid fusion tuning and query/document expansion

Evidence
- RRF score = sum 1/(k + rank); k=60 near-optimal in pilots but "not critical"; RRF beat Condorcet and CombMNZ by 4 to 5% on average. https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf
- BM25 k1 controls TF saturation, b controls length normalization; tune on a dev set. https://arxiv.org/pdf/0911.5046 . Verse-level units are short and uniform, so b likely matters less **[UNVERIFIED]**.
- Doc2Query--: filtering generated expansion queries with a relevance scorer improves effectiveness and shrinks index; BM25 params and filter percentage were tuned on dev. https://arxiv.org/abs/2301.03266
- Dense pseudo-relevance feedback (Rocchio/average of top-k doc embeddings into the query vector) is an LLM-free expansion. https://arxiv.org/abs/2112.06400
- HyDE-without-LLM: no direct source found. Closest evidence-backed substitutes are doc2query (index-time) and dense PRF (query-time) **[UNVERIFIED as HyDE equivalent]**.

Actionable: grid search k in {10, 20, 60}, dense weight in {0.5..2}, BM25 k1 in {0.9, 1.2, 1.5}, b in {0.3, 0.75}, on the dev set; add archaic-form normalization (spelling map for thee/thou/ye, -eth/-est stemming) to BM25 analyzers **[UNVERIFIED gain]**; add dense PRF (top-3, alpha 0.7) as an ablation.

## 6. Chunking evidence

- Semantic chunking does not consistently justify its compute cost vs simpler fixed or sentence-based splitting. https://aclanthology.org/2025.findings-naacl.114.pdf
- Document segmentation choices materially affect RAG retrieval. https://aclanthology.org/2025.findings-acl.422.pdf
- Context window utilization as a hyperparameter; chunk size should be tuned per task. https://arxiv.org/abs/2407.19794
- Verse units are natural sentence-like boundaries, consistent with the evidence that sentence-boundary chunking is a strong default. Recommendation: keep verse retrieval units (citations stay precise) and add an optional neighbor-window representation (verse +/-1) for the reranker and verifier; evaluate both **[UNVERIFIED gain]**.

## Unverified / open items
- Exact license strings for e5, gte, Arctic (check model cards before shipping).
- Browser latency of 278M to 568M rerankers and NLI under WASM/WebGPU.
- Any quantitative effect of archaic English on e5 (no source found).
