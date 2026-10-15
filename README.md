<div align="center">

<img src="site/assets/logo.svg" alt="Scripture RAG logo: an open book with light rising evenly in every direction" width="120" height="120">

# Scripture RAG

**Answers only from the texts you load. With citations, or an honest refusal.**

![Python 3.11](https://img.shields.io/badge/python-3.11-3b5bdb)
![License MIT](https://img.shields.io/badge/license-MIT-0ca678)
![Tests](https://img.shields.io/badge/tests-pytest%20%2B%20node-e8590c)
![Coverage](https://img.shields.io/badge/coverage-89%25-0ca678)
![Offline first](https://img.shields.io/badge/offline-first-495057)

[Quick start](#quick-start) | [Diagrams](#diagrams) | [How it works](#how-it-works) | [Measured results](#measured-results) | [Theme map](#unsupervised-theme-map) | [Complexity](#big-o-summary) | [Docs](#documentation)

</div>

---

Load public-domain books (PDF or TXT, including Internet Archive downloads), ask in English or Spanish, and get the exact verses behind each answer as `[book p.N #chunk_id]`. Answers are extractive: the app quotes your texts and never generates prose, and there is no LLM anywhere in the pipeline. When the text does not support an answer the app says `I could not find this in the loaded texts.`

<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/screenshots/desktop-dark-answer.png">
  <img src="docs/screenshots/desktop-light-answer.png" alt="Browser app answering 'Who is my shepherd?' with the quoted text of Psalm 23 and the citation psalm23_kjv p.1-3 #0000" width="820">
</picture>
</p>

## Features

| | | |
|---|---|---|
| **Hybrid retrieval**<br>BM25 on the translated and original query plus `multilingual-e5-small`, fused with weighted Reciprocal Rank Fusion | **Reranking**<br>`mmarco-mMiniLMv2-L12` cross-encoder reranks the top 50 (browser default; Python default is an offline lexical reranker) | **Citations everywhere**<br>Every quote carries `[book p.N #chunk_id]`; a validator rejects anything else |
| **Two refusal gates**<br>Per-language score threshold plus lexical support >= 0.3 | **Extractive answers**<br>Answers are quoted verses; there is no LLM or generation step | **Theme map**<br>k-means, silhouette, PCA, TF-IDF topic labels, optional BERTopic |
| **English and Spanish**<br>ES questions are translated to EN with MarianMT before search; Portuguese is not supported yet | **Verse-aware chunks**<br>About 80 words, 1 verse overlap; 200/40 token windows when no verses are found | **Archaic normalization**<br>`thee`, `saith`, `-eth` forms normalized; queries expanded with synonyms |
| **Similar passages**<br>Nearest neighbors from other books | **Browser demo**<br>Runs fully client side, history in IndexedDB | **Safe uploads**<br>Magic bytes, size and page caps, parse timeout, escaped output |
| **Compare mode**<br>Same question answered per tradition, side by side | **Outlier flags**<br>Passages far from every cluster are marked on the theme map | **Library manifest**<br>Books, roles, traditions and licenses, exportable as CSV or JSON |

### Feature notes

- **Internet Archive panel.** Search archive.org with client-side year, author, collection and public-domain filters; each hit shows a license label. Hits are clustered (k-means, K by silhouette, up to 5) with the active embedder and ranked by similarity to the centroid of the loaded library, so related books surface first. For any item, `licenseurl`, `rights`, `possible-copyright-status` and dates are shown before anything downloads. PDFs are parsed page by page (pypdf in Python, pdf.js in the browser) so every chunk keeps `page_start`/`page_end`.
- **Book selector.** Choose which loaded books an answer may search; retrieval then runs on that subset only (subset BM25 plus filtered vectors).
- **Query preparation.** `clean_query` tidies the raw question, the language is detected (English and Spanish are supported; a Portuguese question gets a "not supported yet, ask in English or Spanish" notice), Spanish questions are translated to English with MarianMT, then archaic normalization and query expansion are applied.
- **Hybrid retrieval.** BM25 (exact words, names, verse references; run on both the translated and the original query) and `multilingual-e5-small` embeddings (meaning, cross-language) each return candidates; weighted Reciprocal Rank Fusion merges the rankings without score calibration. The `mmarco-mMiniLMv2-L12-H384-v1` cross-encoder reranks the top 50 fused candidates, and the top 3 passages that pass both gates are quoted.
- **Two gates.** The best reranked passage must clear a per-language threshold (en 0.12, es 0.16) and have lexical support of at least 0.3 against the question; otherwise the app refuses. Values and how they were chosen: [docs/eval/SCORECARD.md](docs/eval/SCORECARD.md).
- **Browser and Python parity.** The browser pipeline (`site/js/rag.js`) and the Python one make identical answer/refuse decisions and citations on the eval set when fed the same model outputs ([docs/eval/parity.json](docs/eval/parity.json)); with the browser models themselves 248 of 248 decisions agree ([parity_models.json](docs/eval/parity_models.json)). Above 20,000 passages the browser switches from brute-force search to an HNSW index.
- **History.** Every question, answer and citation is stored in IndexedDB (`scripture-rag` database) with search, re-open, export/import JSON and clear all. Nothing leaves the browser.
- **Theme map.** k-means (K by silhouette) on passage embeddings, PCA to 2D, class-based TF-IDF labels. Re-clusters when books change; click a point for nearest passages from other books.
- **Compare mode.** Asks the same question once per group (tradition by default, or per book or role); each group's answer cites only its own books and can refuse independently.
- **Outlier flags.** Passages whose distance to their cluster centroid has a z-score above 2.5 are flagged, which tends to surface OCR noise, front matter and indexes.
- **Library manifest.** One row per loaded book: id, title, role, tradition, date, source URL, license status, pages, passages, added date; export to CSV or JSON for audit.

## Quick start

### Windows (PowerShell)

```powershell
# Python 3.11 from python.org, then:
git clone https://github.com/IgorCSIS/scripture-rag.git
cd scripture-rag
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt -e .
copy .env.example .env

# Tests and the command line demo (uses the small public-domain test fixtures)
python -m pytest
python demo.py --corpus tests\fixtures\books

# Local app at http://127.0.0.1:7860
python app_gradio.py
```

Optional real models (embedder and cross-encoder reranker):

```powershell
pip install -r requirements-models.txt          # sentence-transformers, BERTopic (downloads torch)
# In .env: EMBED_MODEL=intfloat/multilingual-e5-small, RERANK_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
python app_gradio.py
```

`make` is optional on Windows; every Makefile target is a single `python ...` command shown above.

### macOS / Linux

```bash
python3.11 -m venv .venv && source .venv/bin/activate
make install && make test
make demo CORPUS=tests/fixtures/books
make app
```

### Run the hosted demo (extractive)

```bash
streamlit run app_streamlit.py
```

On Streamlit Community Cloud, point a new app at `app_streamlit.py`; keep `EMBED_MODEL=hashing` or a small model to stay under the 2.7 GB memory cap. No secrets are needed. Not deployed yet.

### Browser demo

```bash
python -m http.server 8000 --directory site   # then open http://localhost:8000
```

Everything stays in your browser: libraries (chunks plus embeddings) and the Q&A history live in IndexedDB, with search, re-open, export/import JSON and clear all. Switch embeddings to `multilingual-e5-small` to download the model once from Hugging Face.

## Diagrams

<p align="center"><img src="docs/diagrams/architecture.svg" alt="Architecture: the browser app (UI, ingest, rag.js pipeline, IndexedDB, IA panel, transformers.js models, indexes, theme map) and the Python package (apps, ingest, RagPipeline, extractive answer, Internet Archive, models, indexes, eval harness), kept in parity" width="860"></p>
<p align="center"><img src="docs/diagrams/data_flow.svg" alt="Data flow: ingest (sanitize, normalize, verse chunk, index) and query (clean, detect language, translate, expand, BM25 plus dense, weighted RRF, rerank top 50, language gate, support gate, cited quotes or refusal)" width="860"></p>
<p align="center"><img src="docs/diagrams/state.svg" alt="Query state machine: Idle, Preparing, Retrieving, Reranking, Gating, Support check, then Answered or Refused, then Saved to history" width="860"></p>

Mermaid versions of all required diagrams (user story map, use case, architecture, sequence, data flow, ERD, state, class, GUI mockup) are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## How it works

<p align="center"><img src="site/assets/pipeline.svg" alt="Ingest pipeline: upload checks, parse pages, strip control characters, normalize archaic spelling, verse chunks of about 80 words with 1 verse overlap, embed and index" width="860"></p>
<p align="center"><img src="site/assets/retrieval.svg" alt="Answer flow: question, clean_query, detect language, translate to English, normalize and expand, BM25 and dense search, weighted RRF, rerank top 50, language gate, support gate, cited verse quotes or refusal" width="860"></p>

```mermaid
sequenceDiagram
  participant U as User
  participant P as Pipeline
  participant R as HybridRetriever
  participant RR as Reranker
  U->>P: question
  P->>P: clean_query, detect language, translate ES to EN, expand
  P->>R: expanded query + original query
  R-->>P: top 50 fused by weighted RRF
  P->>RR: rerank 50 candidates
  RR-->>P: top 5 + scores
  alt score < t[lang] or support < 0.3
    P-->>U: I could not find this in the loaded texts.
  else both gates pass
    P-->>U: top 3 cited verse quotes
  end
```

## Unsupervised theme map

<p align="center"><img src="site/assets/clustering.svg" alt="Theme map pipeline: passage vectors, k-means with K by silhouette, PCA to 2D, TF-IDF labels, outlier flags, theme map, similar passages" width="860"></p>
<p align="center"><img src="docs/screenshots/theme_map.png" alt="Theme map in the browser" width="820"></p>

Passage embeddings are clustered with k-means (K chosen by silhouette), projected to 2D with PCA and labeled with class-based TF-IDF keywords. The map re-clusters as books are added; clicking a point lists the most similar passages from other books. Clusters are statistical similarity, not doctrinal claims.

## Big-O summary

n = chunks, d = embedding dim, k = candidates, K = clusters, i = iterations, L = chunk tokens, q = query tokens, N = characters.

| Operation | Time | Measured slope |
|---|---|---|
| Chunking | O(N) | 1.05 |
| Brute-force search | O(n d + n log k) | 1.05 |
| HNSW search | about O(d log n) | 0.38 |
| BM25 query | O(postings), O(n q) in rank_bm25 | 1.21 |
| RRF | O(k log k) | |
| Cross-encoder rerank | O(k (q+L)^2 d) | |
| k-means | O(n K i d) | 0.62 |
| PCA to 2D | O(n d^2) | 0.25 (solver switching, see docs) |
| Outliers (z-score) | O(n d) | 0.53 |
| Role-scoped answer | O(n_s) | 0.86 |
| Compare by tradition | O(G n_g) | 0.83 |
| filterHits | O(r) | 1.14 |
| clusterHits | O(r^2 d) silhouette | 1.86 |
| rankRelated | O(r d + r log r) | 1.04 |
| Book selector (selectBooks) | O(B + s) | 0.72 |

<p align="center"><img src="docs/benchmarks/overview.png" alt="Log-log plot of median Python runtime vs n for chunking, brute-force search, HNSW, BM25, k-means and PCA" width="640"></p>

<p align="center">
<img src="docs/benchmarks/brute_force_search.png" width="410" alt="Brute-force search median latency vs n, linear and log-log, slope 1.05">
<img src="docs/benchmarks/hnsw_search.png" width="410" alt="HNSW search median latency vs n, linear and log-log, slope 0.38">
<img src="docs/benchmarks/kmeans.png" width="410" alt="k-means median latency vs n, linear and log-log, slope 0.62">
<img src="docs/benchmarks/bm25_query.png" width="410" alt="BM25 query median latency vs n, linear and log-log, slope 1.21">
</p>

Full table, theory (curse of dimensionality, NP-hardness of k-means, refusal as a decision problem, citations as a regular language, rank aggregation) and discussion: [docs/COMPLEXITY.md](docs/COMPLEXITY.md). Regenerate with `python bench/bench_complexity.py`.

## Evaluation

`python -m scripture_rag.eval --corpus tests/fixtures/books --gold tests/fixtures/gold_fixture.jsonl`

| Metric (fixture smoke set, 12 items) | Value |
|---|---|
| Recall@10 vector / BM25 / hybrid / hybrid+rerank | 1.00 / 1.00 / 1.00 / 1.00 |
| Answer rate (answerable) | 1.00 |
| Refusal recall (unanswerable) | 1.00 |

The fixture corpus is tiny, so these numbers validate the harness rather than quality. Real-book results are under [Measured results](#measured-results). See [docs/EVALUATION.md](docs/EVALUATION.md).

## Measured results

All numbers below are copied from files committed in the repo. Nothing is estimated.

### PDF stress test ([docs/eval/stress_results.md](docs/eval/stress_results.md))

Previous configuration (before verse chunks, translation and the two gates): two Internet Archive books (KJV 1611 reprint, Rodwell Koran), `multilingual-e5-small` + `mmarco-mMiniLMv2-L12-H384-v1`, 200/40 windows, single threshold 0.22, 81 items. Kept for history; the shipped configuration is measured in the [Final scorecard](#final-scorecard).

| Metric | Value |
|---|---|
| Overall pass rate | 0.765 |
| Answer accuracy (en / pt / es) | 0.682 (0.75 / 0.667 / 0.333) |
| Citation validity | 1.0 |
| Refusal recall / precision | 0.929 / 0.906 |
| False refusals on answerable | 3 |
| Trick / injection pass rate | 0.667 / 0.667 |
| Latency p50 (previous config) | 1972 ms |

### Grounding: source not loaded vs loaded ([docs/eval/loaded_phase_results.md](docs/eval/loaded_phase_results.md))

Previous configuration (single threshold 0.22).

| Condition | Pass | Answered | Abstained |
|---|---|---|---|
| Source not loaded | 11/13 | 2 | 11 |
| Source loaded | 11/13 | 9 | 4 |

### Browser ANN ([docs/benchmarks/browser_ann.json](docs/benchmarks/browser_ann.json))

| n | Brute p50 ms | HNSW p50 ms | HNSW build s | HNSW recall@10 |
|---|---|---|---|---|
| 1,000 | 0.62 | 0.39 | 0.94 | 1.00 |
| 5,000 | 3.19 | 0.28 | 3.90 | 0.85 |
| 20,000 | 15.97 | 0.82 | 20.54 | 0.79 |
| 50,000 | 43.86 | 1.31 | 73.64 | 0.94 |

### Browser library stages, ms ([docs/benchmarks/library_stages.json](docs/benchmarks/library_stages.json))

| Stage | n=500 | 1,000 | 2,000 | 4,000 | 8,000 |
|---|---|---|---|---|---|
| Outliers (z-score) | 7.30 | 11.12 | 7.39 | 18.00 | 35.46 |
| Book auto-tags | 0.11 | 0.07 | 0.11 | 0.53 | 0.45 |
| Manifest CSV | 0.07 | 0.11 | 0.07 | 0.13 | 0.28 |
| Role-scoped answer | 3.46 | 4.54 | 7.80 | 18.35 | 34.21 |
| Compare by tradition | 7.36 | 10.54 | 15.45 | 36.88 | 69.60 |

<p align="center"><img src="docs/benchmarks/library_stages.png" width="640" alt="Log-log plot of browser library stage latency vs passages n: outliers, auto-tags, manifest CSV, role-scoped answer, compare by tradition"></p>
<p align="center"><img src="docs/benchmarks/browser_ann.png" width="640" alt="Log-log plot of browser p50 search latency vs n passages, brute force vs HNSW, with HNSW recall@10 at each n"></p>

IA panel stages, ms by r search hits (for the book selector, B = r books):

| Stage | r=25 | 50 | 100 | 200 | 400 | Slope |
|---|---|---|---|---|---|---|
| filterHits | 0.008 | 0.011 | 0.040 | 0.219 | 0.089 | 1.14 |
| clusterHits | 12.389 | 34.572 | 130.509 | 494.891 | 2038.200 | 1.86 |
| rankRelated | 0.020 | 0.035 | 0.078 | 0.135 | 0.386 | 1.04 |
| Book selector (selectBooks) | 0.004 | 0.005 | 0.008 | 0.012 | 0.031 | 0.72 |

<p align="center"><img src="docs/benchmarks/ia_stages.png" width="640" alt="Log-log plot of Internet Archive panel stage latency vs r hits: filterHits, clusterHits, rankRelated, book selector"></p>

Server-side vector store comparison (NumPy, FAISS Flat/HNSW/IVF, LanceDB, Chroma): [docs/VECTOR_DB.md](docs/VECTOR_DB.md), raw data in [docs/benchmarks/vectordb.json](docs/benchmarks/vectordb.json).

### Final scorecard

From the strict tune / held-out run of [docs/eval/eval_set.jsonl](docs/eval/eval_set.jsonl), shipped configuration, copied from [docs/eval/SCORECARD.md](docs/eval/SCORECARD.md) at commit df4a640. Tuning used the tune split and a synthetic dev set. The shipped gate is the round 1 policy, restored after the round 2 held-out result was seen, and Portuguese was then removed for launch; held-out was re-scored once on English and Spanish items only, so these numbers are not an untouched estimate (details in the scorecard).

| Metric | Value |
|---|---|
| Items | English and Spanish: 96 tune, 114 held-out (Portuguese items excluded); numbers below are held-out |
| Answer accuracy (en / es) | 0.859 (0.862 / 0.846); target 0.90, not met |
| Citation validity | 1.0 (extractive quotes, valid by construction) |
| Refusal recall / precision | 0.885 / 0.833; 3 adversarial answered, 10 unsupported answers (incl. adversarial); injection pass 1.000 |
| Retrieval recall | not reported; accuracy is graded on cited evidence instead |
| Browser parity | Python model outputs: 248/248 decisions and citations (eval), 175/175 (dev); browser models (transformers.js in Node): 248/248 decisions, 242/248 identical quoted contexts ([parity_models.json](docs/eval/parity_models.json)) |
| Browser stage latency, median | translate 0 ms, retrieve 26 ms, rerank 1,892 ms (browser models in Node, [parity_models.json](docs/eval/parity_models.json)) |

## Limitations

- OCR noise in scanned Internet Archive books affects retrieval; scanned PDFs without a text layer yield no text.
- Citations use the PDF page index, which can differ from the printed page number.
- A chunk spanning two pages is cited by its first page; the Sources view shows the full range.
- The browser full-pipeline mode ships `multilingual-e5-small`, MarianMT and the mMiniLM cross-encoder. The Python defaults (`EMBED_MODEL=hashing`, `RERANK_MODEL=lexical`) are offline and match words, not meaning; install the real models for semantic and cross-language questions. The gate thresholds are calibrated for the e5 + mMiniLM pair.
- Portuguese is not supported at launch: its held-out accuracy was too low, so PT questions get a notice asking for English or Spanish.
- Clusters are statistical, not doctrinal.

## Project layout

```
src/scripture_rag/   config, sanitize, ingest, verses, chunker, normalize, text_utils, translate,
                     embed, index, bm25, hybrid, rerank, gate, generate (extractive), validate,
                     pipeline, topics, similar, library, ia_search, ia_import, discover, pdfgen,
                     eval, corpus
app_gradio.py        local app (extractive answers)
app_streamlit.py     extractive demo
demo.py              command line demo
bench/               benchmarks; plot_benchmarks.py re-plots every chart from the JSON
site/                static website and browser demo: js/rag.js (pipeline), models.js, ann.js (HNSW),
                     core.js, library.js, ia.js, archive-ui.js, db.js (IndexedDB), app.js, ui.js
scripts/             eval, parity and scorecard scripts
tests/               pytest suite and public-domain fixtures
docs/                PRD, architecture, complexity, evaluation, security, sources, progress
```

## Documentation

| Document | Contents |
|---|---|
| [docs/PRD.md](docs/PRD.md) | Requirements FR-01 to FR-20 with acceptance criteria and evidence |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | All mermaid diagrams |
| [docs/COMPLEXITY.md](docs/COMPLEXITY.md) | Big-O table, computational theory, benchmark plots |
| [docs/COURSE_MAPPING.md](docs/COURSE_MAPPING.md) | Feature to course module mapping |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Eval harness and smoke results |
| [docs/eval/SCORECARD.md](docs/eval/SCORECARD.md) | Final held-out scorecard, tuning steps, browser parity |
| [docs/eval/](docs/eval/) | Eval sets, stress and grounding results, raw JSON |
| [docs/research/TUNING_RESEARCH.md](docs/research/TUNING_RESEARCH.md) | Tuning research notes |
| [docs/benchmarks/](docs/benchmarks/) | Benchmark JSON and PNGs |
| [docs/VECTOR_DB.md](docs/VECTOR_DB.md) | Vector index choice |
| [docs/SECURITY.md](docs/SECURITY.md) | Uploads, prompt injection, storage, secrets |
| [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) | Text and model license evidence |
| [docs/PROGRESS.md](docs/PROGRESS.md) | Phase progress log |
| [docs/LAUNCH_CHECKLIST.md](docs/LAUNCH_CHECKLIST.md) | Steps to go public and enable Pages |

Every `.md` file in the repo:

```
README.md
DEVELOPMENT.md
CONTRIBUTING.md
data/demo/README.md
data/gold/README.md
docs/PRD.md
docs/ARCHITECTURE.md
docs/COMPLEXITY.md
docs/COURSE_MAPPING.md
docs/DATA_SOURCES.md
docs/EVALUATION.md
docs/LAUNCH_CHECKLIST.md
docs/PROGRESS.md
docs/SECURITY.md
docs/VECTOR_DB.md
docs/eval/SCORECARD.md
docs/eval/scorecard_notes.md
docs/eval/loaded_phase_results.md
docs/eval/stress_results.md
docs/research/TUNING_RESEARCH.md
```

## License

MIT, see [LICENSE](LICENSE). Bundled sample texts are public domain; see [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).
