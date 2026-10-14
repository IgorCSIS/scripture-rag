# Product Requirements: Scripture RAG

Owner: Igor Lima. Version 0.1 (MVP). Date: 2026-10-14.

## Problem
Readers of religious and philosophical texts want answers they can verify. General chatbots blend sources and invent citations. Scripture RAG answers only from books the user loads, cites book, PDF page and chunk id for every sentence, and refuses with `I could not find this in the loaded texts.` when the text does not support an answer.

## Users
- Reader / student: loads public-domain books, asks questions, checks passages.
- Developer / grader: runs tests, evaluation and benchmarks.

## Functional requirements

| ID | Requirement | Acceptance criteria | Status | Evidence |
|---|---|---|---|---|
| FR-01 | Upload PDF/TXT | `.pdf`/`.txt` up to 50 MB and 2000 pages accepted; wrong type, fake extension (magic bytes), invalid UTF-8, empty and oversize files rejected with a message | Done | `tests/test_sanitize.py`, `tests/test_ingest_chunker.py` |
| FR-02 | Internet Archive import (stretch) | Metadata fetched; `licenseurl`, `rights`, `possible-copyright-status`, dates shown; download only after confirmation; PDF preferred else `_djvu.txt` | Done (mocked test) | `tests/test_extras.py::test_ia_import_mocked` |
| FR-03 | Page-aware chunks | Each chunk stores book_id, page_start, page_end, char offsets; 3 page fixture yields correct ranges; overlap verified | Done | `tests/test_ingest_chunker.py` |
| FR-04 | Hybrid retrieval | BM25 + vector + RRF (k=60); Recall@10 hybrid >= vector-only on gold set | Done on fixture gold set | `tests/test_retrieval.py::test_hybrid_not_worse_than_vector` |
| FR-05 | Rerank | Top 5 after rerank; ablation (vector, BM25, hybrid, hybrid+rerank) in `docs/EVALUATION.md` | Done (lexical reranker by default; cross-encoder optional) | `reports/eval.md` |
| FR-06 | Citations | 100% of answer sentences carry a citation resolving to a provided chunk; otherwise refuse | Done | `tests/test_answering.py::test_validator_*` |
| FR-07 | Refusal | Refusal recall >= 0.9 on unanswerable items; answer rate >= 0.8 on answerable | Done on fixture gold set (1.0 / 1.0) | `tests/test_extras.py::test_eval_cli` |
| FR-08 | Source view | Each citation expands to passage and page | Done (Gradio, Streamlit, browser) | `app_gradio.py::sources_md` |
| FR-09 | Theme map | Interactive 2D map with topics and book legend; exportable HTML | Done (k-means + PCA + TF-IDF; BERTopic optional) | `tests/test_extras.py::test_topics_deterministic` |
| FR-10 | Similar passages | Top 5 neighbors from other books with scores | Done | `tests/test_extras.py::test_similar_other_books` |
| FR-11 | Eval harness | `python -m scripture_rag.eval` writes `reports/eval.json` and `reports/eval.md` | Done | `tests/test_extras.py::test_eval_cli` |
| FR-12 | Benchmarks | `python bench/bench_complexity.py` writes PNGs to `docs/benchmarks/` | Done | `docs/benchmarks/*.png` |
| FR-13 | Two modes | `MODE=local` and `MODE=extractive` both pass smoke tests | Done (Ollama mocked) | `tests/test_answering.py::test_modes_smoke` |
| FR-14 | Multilingual | 5 PT and 5 ES gold items retrieve the right passage in top 5 | Not yet: needs real model and gold items | Planned |
| FR-15 | Browser history and persistence | Libraries (chunks + embeddings) and Q&A history stored in IndexedDB; search, re-open, export/import JSON, clear all; nothing sent to a server | Done | `site/js/db.js`, browser smoke test |
| FR-16 | Browser theme map | k-means + PCA + TF-IDF labels in the browser; re-clusters when books change; click for cross-book neighbors | Done | `site/tests/core.test.mjs` |
| FR-17 | Role-scoped answers | Primary and commentary books tagged by role; an answer filtered to one role cites only books with that role | Done | `site/tests/library.test.mjs` "answers by role never mix primary and commentary" |
| FR-18 | Compare mode | Same question answered per tradition, book or role; each group cites only its own books and can refuse on its own | Done | `site/tests/library.test.mjs` "compare groups by tradition and by book", "compare groups by role" |
| FR-19 | Outlier flags | Passages with centroid-distance z-score above 2.5 flagged on the theme map; a planted far point is flagged | Done | `site/tests/library.test.mjs` "outliers flag the far point" |
| FR-20 | Library manifest | One row per book with source URL, license status, pages, passages; CSV and JSON export round-trip the fields | Done | `site/tests/library.test.mjs` "manifest rows and CSV" |

MVP = FR-01, 03 to 08, 11 to 13. Stretch delivered: FR-02, 09, 10, 15 to 20.

## Non-functional
- Offline except model downloads and the optional IA import.
- No secrets in code; config from environment (`.env.example`).
- Streamlit demo under 2.7 GB memory (extractive, small or hashing embedder).
- Python 3.11, type hints, docstrings with Parameters/Returns, ruff clean, coverage >= 80%.

## Out of scope (v0.1)
Conversation memory, OCR of scanned PDFs, user accounts, hosted LLM.
