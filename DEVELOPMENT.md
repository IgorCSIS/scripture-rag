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
