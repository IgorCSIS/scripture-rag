# Development rules

Project rules for anyone (person or tool) changing this repository.

- Never guess library behavior: check official docs and pin versions in `requirements*.txt`.
- No API keys or secrets in code; read config from environment variables; keep `.env.example` current; `.env` stays ignored.
- No em dashes. No credit or attribution lines for tools in code, docs or commit messages.
- Python 3.11, type hints, docstrings with Parameters/Returns for every public function and class (Appendix A), small focused modules, brief comments only where logic is non-obvious, `ruff check .` clean.
- Everything runs offline except model downloads and the optional Internet Archive import.
- Keep `docs/ARCHITECTURE.md` diagrams in sync with the code and log each phase in `docs/PROGRESS.md`.
- Tests: `pytest` (fake embedder, no downloads; mark model tests `@pytest.mark.slow`) and `node --test site/tests/`. Coverage target 80%.
- Only bundle texts with confirmed public-domain evidence (`docs/DATA_SOURCES.md`).

## Screenshots and diagrams

- `NODE_PATH=... node tools/screenshots/capture.mjs docs/screenshots` recaptures every desktop and mobile, light and dark screenshot from a local server on :8765 (`make site` serves on :8000; set `SHOT_URL`).
- `SMOKE_SHOTS=docs/screenshots node scripts/smoke_browser.mjs` refreshes `compare-tab.png`, `books-to-search.png` and `ia-panel.png`.
- `python tools/diagrams/gen.py` and `python tools/diagrams/logo.py` rebuild all SVG diagrams and the logo (see `tools/diagrams/README.md`).
