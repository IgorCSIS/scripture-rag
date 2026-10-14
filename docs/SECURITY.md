# Security

Defensive controls only. Threat model: untrusted uploaded documents (malformed files, oversized files, hidden instructions), untrusted network metadata (Internet Archive), and accidental secret leaks.

## Uploads (`sanitize.py`, `ingest.py`, `site/js/app.js`)
- Allow-list `.pdf`, `.txt`. Magic bytes checked: `%PDF-` for PDF; TXT must be valid UTF-8 and must not start with `%PDF-`.
- Caps: 50 MB, 2000 pages (configurable). Empty files rejected.
- PDF parsing runs with a timeout (thread based: the caller gets an error after `PARSE_TIMEOUT_S`; the worker thread cannot be force-killed, a known limitation; a subprocess would be stronger).
- Only page text is extracted (pypdf `extract_text`, PDF.js `getTextContent` with `isEvalSupported: false`). Embedded JavaScript, links, forms and annotations are never executed or followed.
- Control characters stripped. All document text is HTML-escaped (Gradio), shown with `st.text` (Streamlit) or inserted with `textContent` (browser). No document HTML is ever rendered.
- Uploads go to Gradio temp files with random names, removed when the session ends; Streamlit keeps bytes in memory only.

## Prompt injection
- Context blocks are numbered, fenced (`<<<DATA ... DATA>>>`), fence markers inside text are broken up, and the system prompt labels them as untrusted data.
- Retrieval gate refuses before any LLM call when relevance is low; the citation validator refuses when any sentence lacks a valid citation.
- Extractive mode quotes text and never follows it. Test: `test_prompt_injection_does_not_change_behavior`.

## Network
- Ollama URL must resolve to localhost (`Settings` raises otherwise).
- Internet Archive: identifier validated by regex; only `archive.org/metadata` and `archive.org/download` are called; license fields shown and explicit confirmation required.
- Browser demo: strict Content-Security-Policy (scripts only from self and jsDelivr; `connect-src` limited to jsDelivr and Hugging Face for model files). No backend; nothing is uploaded.

## Browser storage
- IndexedDB data stays on the device. "Clear all" deletes every store. Imports are shape-checked and vectors coerced to numbers.

## Secrets and supply chain
- No keys in code; `.env` ignored; `.env.example` documents variables.
- Dependencies pinned in `requirements*.txt`; CDN libraries pinned by exact version.
- CI runs `ruff`, `pytest`, `node --test`, `pip-audit` and `detect-secrets`.

## Known gaps
- Thread-based parse timeout (see above). No OCR. No subresource integrity hashes yet for CDN imports. CSP is set by a meta tag (GitHub Pages cannot set headers).
