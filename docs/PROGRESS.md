# Progress log

| Date | Phase | Done | Next | Blockers |
|---|---|---|---|---|
| 2026-10-14 | 0 Plan + scaffold | PLAN.md, build prompt, repo layout, pyproject, Makefile, CI (ruff, pytest, node tests, pip-audit, detect-secrets) and Pages workflow (manual) | Confirm GPU VRAM, pick demo books | None |
| 2026-10-14 | 1 Ingest + chunk | Sanitizer (allow-list, magic bytes, caps, timeout), PDF/TXT parsing, page-aware chunker + tests | | None |
| 2026-10-14 | 2 Retrieval | BM25, brute force, FAISS HNSW, LanceDB, RRF, hybrid retriever + tests (HNSW recall@10 >= 0.9) | | None |
| 2026-10-14 | 3 Answering | Lexical and cross-encoder rerankers, extractive and Ollama generators, citation validator, refusal gate, injection test | Calibrate threshold with real models | Real models not run in CI |
| 2026-10-14 | 4 Eval + bench | Eval harness and fixture gold set; benchmark script run, PNGs committed | Real 50 item gold set (Igor reviews) | Demo books not chosen |
| 2026-10-14 | 5 UI | Gradio local app, Streamlit extractive app, browser demo with IndexedDB history | Deploy Streamlit demo after review | Repo private; Igor reviews first |
| 2026-10-14 | 6 Unsupervised | k-means + silhouette, PCA, TF-IDF labels, Plotly export, similar passages (Python and browser), BERTopic hook | Compare BERTopic vs k-means on real books | Needs real corpus |
| 2026-10-14 | 7 Polish | README, website, logo, SVG diagrams | Screenshots/GIF with real books, enable Pages | Pages on a private repo needs a paid plan |
| 2026-10-14 | 8 Library + docs | Role tags, compare mode, outlier flags, library manifest with benchmarks; README rebuilt with SVG diagrams and measured results; PRD FR-17 to FR-20; launch checklist | Final scorecard from eval worker; launch steps in docs/LAUNCH_CHECKLIST.md | Repo private; Pages not enabled |
| 2026-10-14 | Final | Extractive only (generation option removed), verse chunks, archaic normalization, ES translation (PT deferred), per-language gate plus support gate, browser parity, redrawn diagrams and logo | | See docs/eval/SCORECARD.md |
