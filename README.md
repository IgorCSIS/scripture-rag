<div align="center">

<img src="site/assets/logo.svg" alt="Scripture RAG logo" width="120">

# Scripture RAG

**Answers only from the texts you load. With citations, or an honest refusal.**

![Python 3.11](https://img.shields.io/badge/python-3.11-3b5bdb)
![License MIT](https://img.shields.io/badge/license-MIT-0ca678)
![Tests](https://img.shields.io/badge/tests-pytest%20%2B%20node-e8590c)
![Coverage](https://img.shields.io/badge/coverage-94%25-0ca678)
![Offline first](https://img.shields.io/badge/offline-first-495057)

[Quick start](#quick-start) | [How it works](#how-it-works) | [Theme map](#unsupervised-theme-map) | [Complexity](#big-o-summary) | [Docs](#documentation)

</div>

---

Load public-domain books (PDF or TXT, including Internet Archive downloads), ask in English, Portuguese or Spanish, and get the exact passages behind each answer as `[book p.N #chunk_id]`. When the text does not support an answer the app says `I could not find this in the loaded texts.`

<p align="center"><img src="docs/screenshots/ask.png" alt="Ask tab with a cited passage" width="820"></p>

## Features

| | | |
|---|---|---|
| **Hybrid retrieval**<br>BM25 plus embeddings, fused with Reciprocal Rank Fusion (k = 60) | **Reranking**<br>Lexical reranker offline, cross-encoder optional | **Citations everywhere**<br>Every sentence cites a chunk; a regex validator rejects anything else |
| **Refusal gate**<br>Low relevance means no answer and no LLM call | **Two modes**<br>`MODE=local` with Ollama, `MODE=extractive` with no LLM | **Theme map**<br>k-means, silhouette, PCA, TF-IDF topic labels, optional BERTopic |
| **Similar passages**<br>Nearest neighbors from other books | **Browser demo**<br>Runs fully client side, history in IndexedDB | **Safe uploads**<br>Magic bytes, size and page caps, parse timeout, escaped output |

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

Optional real models and local generation:

```powershell
pip install -r requirements-models.txt          # sentence-transformers, BERTopic (downloads torch)
nvidia-smi                                       # check VRAM: 8 GB fits qwen2.5:7b, 6 GB use llama3.2:3b
# Install Ollama for Windows from https://ollama.com, then:
ollama pull qwen2.5:7b
ollama pull llama3.2:3b
# In .env: MODE=local, EMBED_MODEL=intfloat/multilingual-e5-small, RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
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

## How it works

<p align="center"><img src="site/assets/pipeline.svg" alt="Ingest pipeline" width="860"></p>
<p align="center"><img src="site/assets/retrieval.svg" alt="Retrieval flow" width="860"></p>

```mermaid
sequenceDiagram
  participant U as User
  participant R as HybridRetriever
  participant RR as Reranker
  participant G as Generator
  participant V as Validator
  U->>R: question
  R->>RR: top 50 fused by RRF
  alt top score < threshold
    RR-->>U: I could not find this in the loaded texts.
  else
    RR->>G: top 5 fenced passages
    G->>V: draft with citations
    V-->>U: cited answer, or refusal if any citation is invalid
  end
```

<p align="center"><img src="site/assets/architecture.svg" alt="Architecture" width="860"></p>

All required diagrams (user story map, use case, architecture, sequence, data flow, ERD, state, class, GUI mockup) are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Unsupervised theme map

<p align="center"><img src="site/assets/clustering.svg" alt="Clustering" width="860"></p>
<p align="center"><img src="docs/screenshots/theme_map.png" alt="Theme map in the browser" width="820"></p>

Passage embeddings are clustered with k-means (K chosen by silhouette), projected to 2D with PCA and labeled with class-based TF-IDF keywords. The map re-clusters as books are added; clicking a point lists the most similar passages from other books. Clusters are statistical similarity, not doctrinal claims.

## Big-O summary

n = chunks, d = embedding dim, k = candidates, K = clusters, i = iterations, L = chunk tokens, q = query tokens, N = characters.

| Operation | Time | Measured slope |
|---|---|---|
| Chunking | O(N) | 0.95 |
| Brute-force search | O(n d + n log k) | 0.92 |
| HNSW search | about O(d log n) | 0.42 |
| BM25 query | O(postings), O(n q) in rank_bm25 | 1.15 |
| RRF | O(k log k) | |
| Cross-encoder rerank | O(k (q+L)^2 d) | |
| k-means | O(n K i d) | 0.63 |
| PCA to 2D | O(n d^2) | -0.04 (solver switching, see docs) |

<p align="center"><img src="docs/benchmarks/overview.png" alt="Runtime vs n, log-log" width="560"></p>

<p align="center">
<img src="docs/benchmarks/brute_force_search.png" width="410" alt="Brute force">
<img src="docs/benchmarks/hnsw_search.png" width="410" alt="HNSW">
<img src="docs/benchmarks/kmeans.png" width="410" alt="k-means">
<img src="docs/benchmarks/bm25_query.png" width="410" alt="BM25">
</p>

Full table, theory (curse of dimensionality, NP-hardness of k-means, refusal as a decision problem, citations as a regular language, rank aggregation) and discussion: [docs/COMPLEXITY.md](docs/COMPLEXITY.md). Regenerate with `python bench/bench_complexity.py`.

## Evaluation

`python -m scripture_rag.eval --corpus tests/fixtures/books --gold tests/fixtures/gold_fixture.jsonl`

| Metric (fixture smoke set, 12 items) | Value |
|---|---|
| Recall@10 vector / BM25 / hybrid / hybrid+rerank | 1.00 / 1.00 / 1.00 / 1.00 |
| Answer rate (answerable) | 1.00 |
| Refusal recall (unanswerable) | 1.00 |

The fixture corpus is tiny, so these numbers validate the harness rather than quality. The real 50 item gold set is pending book selection. See [docs/EVALUATION.md](docs/EVALUATION.md).

## Limitations

- OCR noise in scanned Internet Archive books affects retrieval; scanned PDFs without a text layer yield no text.
- Citations use the PDF page index, which can differ from the printed page number.
- A chunk spanning two pages is cited by its first page; the Sources view shows the full range.
- The default offline embedder and reranker match words, not meaning; use the real models for semantic and cross-language questions. The refusal threshold must be calibrated per model.
- Clusters are statistical, not doctrinal.

## Project layout

```
src/scripture_rag/   config, sanitize, ingest, chunker, embed, index, bm25, hybrid, rerank,
                     generate, validate, pipeline, topics, similar, ia_import, eval, corpus
app_gradio.py        local app (Ollama optional)
app_streamlit.py     extractive demo
demo.py              command line demo
bench/               complexity benchmarks
site/                static website and browser demo (js/core.js, js/db.js, js/app.js)
tests/               pytest suite and public-domain fixtures
docs/                PRD, architecture, complexity, evaluation, security, sources, progress
```

## Documentation

```
README.md
DEVELOPMENT.md
CONTRIBUTING.md
data/demo/README.md
data/gold/README.md
docs/PRD.md
docs/ARCHITECTURE.md
docs/COMPLEXITY.md
docs/EVALUATION.md
docs/COURSE_MAPPING.md
docs/SECURITY.md
docs/DATA_SOURCES.md
docs/PROGRESS.md
```

## License

MIT, see [LICENSE](LICENSE). Bundled sample texts are public domain; see [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).
