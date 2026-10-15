// Browser app UI: library, ask with cited passages, passage drawer, theme map, similar passages, history.
import * as core from "./core.js";
import * as db from "./db.js";
import * as ia from "./ia.js";
import * as lib from "./library.js";
import * as rag from "./rag.js";
import { MODELS } from "./models.js";
import { $, el, icon, ICONS, toast, cleanFileName, formatBytes, showDrawer, hideDrawer, drawerIsOpen, trapFocus, verifiedImport, verifiedBlobUrl } from "./ui.js";

// Pinned CDN files with SHA-384 integrity hashes; nothing runs unless the hash matches.
const CDN = {
  pdf: { url: "https://cdn.jsdelivr.net/npm/pdfjs-dist@6.4.299/build/pdf.min.mjs", sri: "sha384-QCvhu/UPp22U/TNLVbQNONL82ES2yY717GAiqP9VVu37Y7+m9VPKvadGx9XlFvcp" }, // pragma: allowlist secret
  pdfWorker: { url: "https://cdn.jsdelivr.net/npm/pdfjs-dist@6.4.299/build/pdf.worker.min.mjs", sri: "sha384-NfW/OTMWezITv4oaALvfH7MFLwzPxEvpySmDpoALa1mtTY8MC200l0ZNFD+SOlrM" }, // pragma: allowlist secret
  transformers: { url: "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1/dist/transformers.min.js", sri: "sha384-AFNtlNOt2vbygtlRCUX2TDgkE+EyonjbCLxAfqezYwJJDVlozUhg+yKs4cPwyjHR" }, // pragma: allowlist secret
  // ONNX Runtime loader and binary are hash-checked too, then handed to the runtime as blob URLs.
  ortMjs: { url: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.0-dev.20260914-8d85527a0/dist/ort-wasm-simd-threaded.jsep.mjs", sri: "sha384-nvWLxdeiWbA+fFeE6qrofH8GNTzogS0wMzbtE+5FpSdY9eJVD9XAw02CuIR67LOV" }, // pragma: allowlist secret
  ortWasm: { url: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.0-dev.20260914-8d85527a0/dist/ort-wasm-simd-threaded.jsep.wasm", sri: "sha384-M3enaJRpYo4Wu07MxfAZnkbeFbm4uWtNn6W27Taw+akrBXSqWBJ45eyQ+zqre5tC" }, // pragma: allowlist secret
};
const MODEL = MODELS.embed.id;
const MAX_BYTES = 50 * 1024 * 1024;
const MAX_FILES = 10;
const MAX_PAGES = 2000;
const NOT_FOUND = "Not found in loaded texts";
const SUGGESTIONS = ["Who is my shepherd?", "Quem e o meu pastor?", "What restores the soul?"];

const state = { db: null, chunks: [], books: [], byBook: new Map(), bm25: new core.BM25(), tags: {}, roleScope: "", embedder: "hashing", model: null, rag: null, norm: null, aligner: null, reranker: null, translator: null, themes: null, busy: false, current: null };

// ---------- Status, progress ----------
function setProgress(prefix, label, pct) {
  const wrap = $(prefix === "model" ? "modelProgress" : "ingest");
  if (pct == null) { wrap.hidden = true; return; }
  wrap.hidden = false;
  $(`${prefix}Label`).textContent = label;
  $(`${prefix}Pct`).textContent = Number.isFinite(pct) ? `${Math.round(pct)}%` : "";
  const bar = $(`${prefix}Bar`);
  if (Number.isFinite(pct)) bar.value = pct; else bar.removeAttribute("value");
}

// ---------- Embeddings (model lazy-loaded on first use) ----------
async function loadModel() {
  if (state.model) return state.model;
  setProgress("model", "Downloading the multilingual model (first time only)", 0);
  toast("Downloading the multilingual model. This happens once.");
  const files = new Map();
  try {
    const tf = await verifiedImport(CDN.transformers.url, CDN.transformers.sri);
    if (tf.env?.backends?.onnx?.wasm) {
      tf.env.backends.onnx.wasm.wasmPaths = {
        mjs: await verifiedBlobUrl(CDN.ortMjs.url, CDN.ortMjs.sri),
        wasm: await verifiedBlobUrl(CDN.ortWasm.url, CDN.ortWasm.sri, "application/wasm"),
      };
    }
    state.tf = tf;
    state.model = await tf.pipeline("feature-extraction", MODEL, {
      dtype: MODELS.embed.dtype,
      progress_callback: (p) => {
        if (p.status === "progress" && p.total) {
          files.set(p.file, [p.loaded, p.total]);
          let a = 0, b = 0;
          for (const [l, t] of files.values()) { a += l; b += t; }
          setProgress("model", `Downloading model, ${formatBytes(a)} of ${formatBytes(b)}`, (a / b) * 100);
        }
      },
    });
    setProgress("model", "Model ready", 100);
    setTimeout(() => setProgress("model", "", null), 1200);
    return state.model;
  } catch (e) {
    setProgress("model", "", null);
    throw new Error(`Model download failed: ${e.message}`);
  }
}

// Reranker and PT/ES translator of the full pipeline, loaded on first use (same weights as Python).
async function loadExtra(key, label, make) {
  if (state[key]) return state[key];
  await loadModel();
  setProgress("model", `Downloading the ${label} (first time only)`, NaN);
  try {
    state[key] = await make(state.tf);
    return state[key];
  } catch (e) {
    throw new Error(`${label} download failed: ${e.message}`);
  } finally {
    setProgress("model", "", null);
  }
}

const pipelineModels = {
  embedQuery: async (t) => (await embed([t], true))[0],
  translate: async (q) => {
    const mt = await loadExtra("translator", "translation model", (tf) => tf.pipeline("translation", MODELS.translate.id, { dtype: MODELS.translate.dtype }));
    return (await mt(q, MODELS.translate.generate))[0].translation_text;
  },
  rerank: async (q, texts) => {
    const r = await loadExtra("reranker", "reranker", async (tf) => ({
      tok: await tf.AutoTokenizer.from_pretrained(MODELS.rerank.id),
      model: await tf.AutoModelForSequenceClassification.from_pretrained(MODELS.rerank.id, { dtype: MODELS.rerank.dtype, model_file_name: MODELS.rerank.file }),
    }));
    const out = [];
    for (let i = 0; i < texts.length; i += 16) {
      const part = texts.slice(i, i + 16);
      const { logits } = await r.model(r.tok(part.map(() => q), { text_pair: part, padding: true, truncation: true }));
      for (const x of logits.data) out.push(1 / (1 + Math.exp(-x)));
    }
    return out;
  },
};

// Archaic spelling normalizer and the numbered reference used to align unnumbered Bibles.
async function getNormalizer() {
  if (!state.norm) state.norm = new rag.ArchaicNormalizer((await (await fetch("assets/modern_vocab.txt")).text()).split(/\s+/).filter(Boolean));
  return state.norm;
}
async function getAligner() {
  if (!state.aligner) {
    const ref = (await (await fetch("assets/kjv_ref_genesis_joshua.json")).json()).map(([book, chapter, verse, text]) => ({ book, chapter, verse, text, page: 1 }));
    state.aligner = new rag.ReferenceAligner(ref, await getNormalizer());
  }
  return state.aligner;
}

async function ragIndex() {
  if (!state.rag) {
    const idx = new rag.RagIndex(await getNormalizer());
    idx.add(state.chunks, state.chunks.map((c) => c.vec));
    state.rag = idx;
  }
  return state.rag;
}

async function embed(texts, isQuery) {
  if (!texts.length) return [];
  if (state.embedder === "hashing") return texts.map((t) => core.hashEmbed(t));
  const model = await loadModel();
  const prefixed = texts.map((t) => (isQuery ? "query: " : "passage: ") + t);
  const out = await model(prefixed, { pooling: "mean", normalize: true });
  return out.tolist().map((v) => Float32Array.from(v));
}

// ---------- File reading ----------
let pdfjsPromise = null;
async function getPdfjs() {
  pdfjsPromise ||= (async () => {
    const pdfjs = await verifiedImport(CDN.pdf.url, CDN.pdf.sri);
    pdfjs.GlobalWorkerOptions.workerSrc = await verifiedBlobUrl(CDN.pdfWorker.url, CDN.pdfWorker.sri);
    return pdfjs;
  })();
  return pdfjsPromise;
}

function checkFile(file) {
  const ext = file.name.toLowerCase().split(".").pop();
  if (!["pdf", "txt"].includes(ext)) throw new Error("Only .pdf and .txt files are supported");
  if (file.type && !["application/pdf", "text/plain", ""].includes(file.type)) throw new Error(`Unexpected file type ${file.type}`);
  if (file.size > MAX_BYTES) throw new Error(`File is ${formatBytes(file.size)}; the limit is 50 MB`);
  if (file.size === 0) throw new Error("File is empty");
  return ext;
}

async function readPages(file, name) {
  const ext = checkFile(file);
  const bytes = new Uint8Array(await file.arrayBuffer());
  const isPdf = String.fromCharCode(...bytes.slice(0, 5)) === "%PDF-";
  if (ext === "pdf") {
    if (!isPdf) throw new Error("File has a .pdf extension but is not a PDF");
    setProgress("ingest", `Loading PDF reader`, NaN);
    const pdfjs = await getPdfjs();
    // Scripting disabled; only the text layer is read, links and annotations are ignored.
    const doc = await pdfjs.getDocument({ data: bytes, isEvalSupported: false, enableXfa: false }).promise;
    if (doc.numPages > MAX_PAGES) throw new Error(`PDF has ${doc.numPages} pages; limit is ${MAX_PAGES}`);
    const pages = [];
    for (let p = 1; p <= doc.numPages; p++) {
      const content = await (await doc.getPage(p)).getTextContent();
      pages.push(content.items.map((i) => i.str).join(" "));
      if (p % 5 === 0 || p === doc.numPages) setProgress("ingest", `Reading ${name}, page ${p} of ${doc.numPages}`, (p / doc.numPages) * 80);
    }
    return pages;
  }
  if (isPdf) throw new Error("File has a .txt extension but is a PDF");
  return core.splitTxtPages(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
}

// ---------- Library ----------
function rebuildIndex() {
  state.rag = null;
  state.bm25 = new core.BM25();
  state.byBook = new Map();
  for (const c of state.chunks) {
    state.bm25.add(c.chunkId, c.text);
    if (!state.byBook.has(c.bookId)) state.byBook.set(c.bookId, []);
    state.byBook.get(c.bookId).push(c);
  }
  for (const list of state.byBook.values()) list.sort((a, b) => a.chunkId.localeCompare(b.chunkId));
}

// Checkbox list choosing which loaded books answers draw from (empty selection = all).
function renderSearchBooks() {
  const box = $("searchBooks");
  if (!box) return;
  state.searchIds = lib.keepLoaded(state.searchIds, state.books);
  box.replaceChildren(...state.books.map((b) => {
    const c = document.createElement("input");
    c.type = "checkbox"; c.value = b.bookId; c.checked = state.searchIds.has(b.bookId);
    c.onchange = () => { if (c.checked) state.searchIds.add(b.bookId); else state.searchIds.delete(b.bookId); };
    const l = document.createElement("label");
    l.className = "small"; l.append(c, ` ${b.title || b.bookId}`);
    return l;
  }));
}

async function refresh() {
  state.books = await db.getAll(state.db, "books");
  state.chunks = await db.getAll(state.db, "chunks");
  const meta = await db.getAll(state.db, "meta");
  state.embedder = meta.find((m) => m.k === "embedder")?.v || "hashing";
  $("embedder").value = state.embedder;
  rebuildIndex();
  renderLibrary();
  renderSearchBooks();
  await renderHistory();
  state.themes = null;
  if (!$("panel-themes").hidden) buildThemes();
  else if (state.chunks.length >= 3) setTimeout(() => { if (!state.themes) buildThemes(); }, 50);
  if (!state.current) renderEmpty();
}

function renderLibrary() {
  const list = $("library");
  const q = $("libFilter").value.trim().toLowerCase();
  const sort = $("libSort").value;
  const count = (b) => state.byBook.get(b.bookId)?.length || 0;
  const scope = $("libRole").value;
  let books = state.books.filter((b) => (!q || b.title.toLowerCase().includes(q)) && (!scope || (b.role || "primary") === scope));
  books.sort(sort === "name" ? (a, b) => a.title.localeCompare(b.title) : sort === "size" ? (a, b) => count(b) - count(a) : (a, b) => (b.addedAt || 0) - (a.addedAt || 0));
  list.replaceChildren();
  if (!state.books.length) list.append(el("li", { class: "small muted" }, "No books yet. Add a file or the sample to begin."));
  else if (!books.length) list.append(el("li", { class: "small muted" }, "No books match this filter."));
  for (const b of books) {
    const rm = el("button", { class: "icon-btn", type: "button", "aria-label": `Remove ${b.title}`, title: "Remove" }, icon(ICONS.trash));
    rm.onclick = async () => { await db.deleteBook(state.db, b.bookId); toast(`Removed ${b.title}`); await refresh(); };
    const role = el("select", { class: "role-select", "aria-label": `Role of ${b.title}` },
      ...lib.ROLES.map((r) => { const o = el("option", { value: r }, r === "primary" ? "Primary" : "Commentary"); o.selected = (b.role || "primary") === r; return o; }));
    role.onchange = () => api().setRole(b.bookId, role.value).then(() => toast(`${b.title}: ${role.value}`));
    const tags = (state.tags[b.bookId] || []).filter(Boolean);
    list.append(el("li", { class: "book" },
      el("div", {}, el("div", { class: "book-title" }, b.title),
        el("div", { class: "book-meta" }, `${b.pages} pages, ${count(b)} passages${b.tradition ? `, ${b.tradition}` : ""}`),
        el("div", { class: "row wrap tag-row" }, role, ...tags.map((t) => el("span", { class: "tag", title: "Auto-tag from the theme clusters" }, t)))), rm));
  }
  renderManifest();
  $("count").textContent = `${state.books.length} ${state.books.length === 1 ? "book" : "books"}`;
}

async function addBook(name, pages, extra = {}) {
  let bookId = core.safeBookId(name), i = 2;
  while (state.books.some((b) => b.bookId === bookId)) bookId = `${core.safeBookId(name)}-${i++}`;
  const clean = pages.map(core.stripControl);
  // Same chunking as Python: verse-aware when the layout allows it, else windows.
  const windowTokens = Math.max(20, Number($("chunkSize").value) || rag.DEFAULTS.windowTokens);
  // Overlap 40 at the default 200 words (as in Python), scaled down for smaller windows.
  const overlap = Math.min(rag.DEFAULTS.windowOverlap, Math.floor(windowTokens / 5));
  const chunks = rag.chunkDocument(bookId, clean, { aligner: await getAligner(), windowTokens, overlap });
  if (!chunks.length) throw new Error("No text found (scanned PDFs need OCR first)");
  setProgress("ingest", `Indexing ${name}, ${chunks.length} passages`, 85);
  const norm = await getNormalizer();
  chunks.forEach((c) => { c.indexText = norm.text(c.text); });
  const vecs = await embed(chunks.map((c) => c.indexText), false);
  chunks.forEach((c, k) => { c.vec = vecs[k]; });
  await db.saveBook(state.db, { bookId, title: name.replace(/\.[^.]+$/, ""), pages: clean.length, addedAt: Date.now(), ...extra }, chunks);
  state.books.push({ bookId });
}

async function onFiles(fileList) {
  let files = [...fileList];
  if (files.length > MAX_FILES) { toast(`Only the first ${MAX_FILES} files were taken`, "error"); files = files.slice(0, MAX_FILES); }
  let ok = 0;
  for (const f of files) {
    const name = cleanFileName(f.name);
    try {
      setProgress("ingest", `Reading ${name}`, 5);
      await addBook(name, await readPages(f, name));
      ok++;
    } catch (e) {
      toast(`Rejected ${name}: ${e.message}`, "error", 6000);
    }
  }
  setProgress("ingest", "", null);
  if (ok) toast(`Loaded ${ok} ${ok === 1 ? "file" : "files"}`);
  $("files").value = "";
  await refresh();
}

async function loadSample() {
  try {
    const res = await fetch("assets/sample/psalm23_kjv.txt");
    await addBook("psalm23_kjv.txt", core.splitTxtPages(await res.text()));
    toast("Added Psalm 23 (KJV)");
  } catch (e) { toast(e.message, "error"); }
  setProgress("ingest", "", null);
  await refresh();
}

// ---------- Answer states ----------
const pageLabel = (c) => `p.${c.pageStart}${c.pageEnd !== c.pageStart ? "-" + c.pageEnd : ""}`;
const roleOf = (id) => state.books.find((b) => b.bookId === id)?.role || "primary";
const api = () => globalThis.ScriptureRAG;
const bookTitle = (id) => state.books.find((b) => b.bookId === id)?.title || id;

function suggestionButtons() {
  return el("div", { class: "suggest" }, SUGGESTIONS.map((s) => el("button", { class: "btn sm", type: "button", onclick: () => { $("ask").value = s; ask(); } }, s)));
}

function renderEmpty() {
  const box = $("answer");
  if (!state.books.length) {
    box.replaceChildren(el("div", { class: "empty" },
      el("img", { src: "assets/logo.svg", alt: "", width: 64, height: 64 }),
      el("h3", {}, "Your library is empty"),
      el("p", { class: "muted" }, "Add a PDF or TXT, or start with the Psalm 23 sample. Files stay on this device."),
      el("div", { class: "row" },
        el("button", { class: "btn primary", type: "button", onclick: loadSample }, "Add sample"),
        el("button", { class: "btn", type: "button", onclick: () => $("files").click() }, "Choose files"))));
  } else {
    box.replaceChildren(el("div", { class: "empty" },
      icon(ICONS.search), el("h3", {}, "Ask anything about your texts"),
      el("p", { class: "muted" }, "Answers quote the loaded passages with page citations. Press / to focus the question box."),
      suggestionButtons()));
  }
}

function renderSkeleton() {
  const card = () => el("div", { class: "sk-card" }, el("div", { class: "sk w90" }), el("div", { class: "sk" }), el("div", { class: "sk w70" }), el("div", { class: "sk chipish" }));
  $("answer").replaceChildren(el("div", { class: "skeleton", "aria-hidden": "true" }, el("div", { class: "sk title" }), card(), card()), el("p", { class: "sr-only" }, "Searching the loaded texts"));
}

function chip(c) {
  const b = el("button", { class: "chip", type: "button", "aria-label": `Open passage from ${bookTitle(c.bookId)}, page ${c.pageStart}` }, icon(ICONS.book), `${c.bookId} ${pageLabel(c)} #${c.chunkId.split("-").pop()}`, el("span", { class: `role-tag ${roleOf(c.bookId)}` }, roleOf(c.bookId)));
  b.onclick = () => openPassage(c, b);
  return b;
}

function renderAnswer(res, q) {
  const box = $("answer");
  if (res.unsupportedLanguage) {
    box.replaceChildren(el("div", { class: "state refusal", role: "alert" }, icon(ICONS.warn), el("h3", {}, "Language not supported yet"),
      el("p", {}, rag.PT_UNSUPPORTED_TEXT)));
    return;
  }
  if (res.refused) {
    box.replaceChildren(el("div", { class: "state refusal", role: "alert" }, icon(ICONS.warn), el("h3", {}, NOT_FOUND),
      el("p", {}, res.trace
        ? `No passage in your library answers "${q}" well enough (best score ${res.topScore.toFixed(2)}, ${res.lang.toUpperCase()} threshold ${res.trace.threshold}; ${res.reason}).`
        : `No passage in your library answers "${q}" well enough (best score ${res.topScore.toFixed(2)}, threshold ${$("threshold").value}).`),
      el("p", { class: "small" }, res.trace ? "Try different wording or add more texts." : "Try different wording, add more texts, or lower the refusal threshold in settings.")));
    return;
  }
  box.replaceChildren(el("div", { class: "answer-head" }, el("h3", {}, "From your texts"), el("span", { class: "badge" }, `${res.citations.length} cited ${res.citations.length === 1 ? "passage" : "passages"}`)),
    ...res.citations.map((c) => el("article", { class: "quote-card" }, el("p", {}, c.text), chip(c))));
}

function renderError(msg) {
  $("answer").replaceChildren(el("div", { class: "state error", role: "alert" }, icon(ICONS.warn), el("h3", {}, "Something went wrong"), el("p", {}, msg)));
}

async function ask() {
  const q = $("ask").value.trim().slice(0, 500);
  showTab("ask");
  if (!q || state.busy) { $("ask").focus(); return; }
  state.busy = true;
  $("askBtn").disabled = true;
  $("answer").setAttribute("aria-busy", "true");
  if (!state.chunks.length) {
    renderEmpty();
    toast("Load a text first", "error");
    state.busy = false; $("askBtn").disabled = false; $("answer").setAttribute("aria-busy", "false");
    return;
  }
  renderSkeleton();
  try {
    const started = performance.now();
    const scope = $("libRole").value;
    const ids = lib.keepLoaded(state.searchIds, state.books);
    let res;
    const lang = rag.detectLanguage(rag.cleanQuery(q));
    if (lang === "pt" && !rag.DEFAULTS.portuguese) {
      res = { refused: true, unsupportedLanguage: true, lang, text: rag.PT_UNSUPPORTED_TEXT, citations: [], topScore: 0, reason: "unsupported language (pt)" };
    } else if (state.embedder !== "hashing") {
      // Full pipeline with the built-in defaults scored in docs/eval/SCORECARD.md.
      const books = scope || ids.size ? lib.selectBooks(state.books, { role: scope || undefined, ids }) : null;
      res = await rag.ragAnswer(q, await ragIndex(), pipelineModels, books ? { books } : {});
    } else {
      // Offline hashing mode: lexical coverage gate with the threshold slider.
      const [qv] = await embed([q], true);
      const opts = { threshold: Number($("threshold").value), cosRange: null };
      res = scope || ids.size ? lib.answerIn(q, state.chunks, lib.selectBooks(state.books, { role: scope || undefined, ids }), qv, opts)
        : core.answer(q, state.chunks, state.bm25, qv, opts);
    }
    const wait = 260 - (performance.now() - started);
    if (wait > 0) await new Promise((r) => setTimeout(r, wait));
    state.current = { q, res };
    renderAnswer(res, q);
    await db.put(state.db, "history", {
      ts: Date.now(), question: q, refused: res.refused, topScore: res.topScore,
      citations: res.citations.map(({ chunkId, bookId, pageStart, pageEnd, text }) => ({ chunkId, bookId, pageStart, pageEnd, text })),
    });
    await renderHistory();
  } catch (e) {
    renderError(e.message);
  } finally {
    state.busy = false;
    $("askBtn").disabled = false;
    $("answer").setAttribute("aria-busy", "false");
  }
}

// ---------- Passage drawer ----------
function openPassage(c, from) {
  const full = state.chunks.find((x) => x.chunkId === c.chunkId) || c;
  const list = state.byBook.get(full.bookId) || [];
  const i = list.findIndex((x) => x.chunkId === full.chunkId);
  $("pd-book").textContent = bookTitle(full.bookId);
  $("pd-title").textContent = `Page ${pageLabel(full).slice(2)}, passage #${full.chunkId.split("-").pop()}`;
  $("pd-prev").textContent = i > 0 ? "... " + list[i - 1].text.slice(-220) : "";
  $("pd-text").textContent = full.text;
  $("pd-next").textContent = i >= 0 && i < list.length - 1 ? list[i + 1].text.slice(0, 220) + " ..." : "";
  $("pd-copy").onclick = async () => {
    try { await navigator.clipboard.writeText(`"${full.text}" ${core.citation(full)}`); toast("Citation copied"); } catch { toast("Copy is blocked in this browser", "error"); }
  };
  $("pd-similar").disabled = !full.vec;
  $("pd-similar").onclick = () => { hideDrawer(false); showSimilar(full); $("tab-similar").focus(); };
  showDrawer($("passage-drawer"), from);
}

// ---------- History ----------
async function renderHistory() {
  const q = $("historySearch").value.toLowerCase();
  const items = (await db.getAll(state.db, "history")).reverse()
    .filter((h) => !q || h.question.toLowerCase().includes(q) || h.citations.some((c) => c.text.toLowerCase().includes(q)));
  const list = $("history");
  list.replaceChildren();
  if (!items.length) list.append(el("li", { class: "small muted" }, q ? "No matches." : "Questions you ask will appear here."));
  for (const h of items.slice(0, 200)) {
    const b = el("button", { class: "hist", type: "button" },
      el("strong", {}, h.question),
      el("span", { class: h.refused ? "tag no" : "tag" }, h.refused ? NOT_FOUND : `${h.citations.length} cited`),
      el("span", { class: "small muted" }, `  ${new Date(h.ts).toLocaleString()}`));
    b.onclick = () => {
      hideDrawer(false);
      $("ask").value = h.question;
      state.current = { q: h.question, res: h };
      showTab("ask");
      renderAnswer({ ...h, topScore: h.topScore || 0 }, h.question);
      $("ask").focus();
    };
    list.append(el("li", {}, b));
  }
}

// ---------- Themes ----------
function buildThemes() {
  const info = $("themeInfo");
  if (state.chunks.length < 3) {
    $("themeMap").replaceChildren(); $("themeLegend").replaceChildren(); $("bookLegend").textContent = "";
    info.textContent = "Load at least a few passages to see the theme map.";
    return;
  }
  if (state.themes) return;
  const step = Math.max(1, Math.floor(state.chunks.length / 1500));
  const sample = state.chunks.filter((_, i) => i % step === 0);
  const vecs = sample.map((c) => c.vec);
  const t0 = performance.now();
  const { labels, k, silhouette } = core.autoCluster(vecs, Number($("maxK").value) || 8);
  const coords = core.pca2d(vecs);
  const keywords = core.clusterKeywords(sample.map((c) => c.text), labels);
  state.themes = { sample, labels, coords, keywords };
  state.tags = ia.bookTags(sample, labels, keywords);
  state.outliers = new Set(lib.outliers(vecs, labels).slice(0, 12).map((o) => o.index));
  renderLibrary();
  info.textContent = `${k} themes from ${sample.length} passages, silhouette ${silhouette.toFixed(3)}, ${Math.round(performance.now() - t0)} ms.`;
  $("themeLegend").replaceChildren(...keywords.map((kw, i) => el("li", {}, el("span", { class: `dot c${i % 10}`, "aria-hidden": "true" }), `T${i}: ${kw.join(", ")}`)));
  drawMap();
}

function drawMap() {
  const svg = $("themeMap");
  const { sample, labels, coords } = state.themes;
  const xs = coords.map((p) => p[0]), ys = coords.map((p) => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const sx = (x) => 30 + ((x - x0) / (x1 - x0 || 1)) * 540, sy = (y) => 30 + ((y - y0) / (y1 - y0 || 1)) * 340;
  const books = [...new Set(sample.map((c) => c.bookId))];
  const NS = "http://www.w3.org/2000/svg";
  svg.replaceChildren();
  sample.forEach((c, i) => {
    const shape = books.indexOf(c.bookId) % 3;
    const x = sx(coords[i][0]), y = sy(coords[i][1]);
    let node;
    if (shape === 0) { node = document.createElementNS(NS, "circle"); node.setAttribute("cx", x); node.setAttribute("cy", y); node.setAttribute("r", 6); }
    else { node = document.createElementNS(NS, "rect"); node.setAttribute("x", x - 5); node.setAttribute("y", y - 5); node.setAttribute("width", 10); node.setAttribute("height", 10); if (shape === 2) node.setAttribute("transform", `rotate(45 ${x} ${y})`); }
    const odd = state.outliers?.has(i);
    node.setAttribute("class", `pt c${labels[i] % 10}${odd ? " outlier" : ""}`);
    node.setAttribute("tabindex", sample.length <= 300 ? "0" : "-1");
    node.setAttribute("role", "button");
    node.setAttribute("aria-label", `${c.bookId} page ${c.pageStart}, theme ${labels[i]}${odd ? ", outlier" : ""}`);
    const title = document.createElementNS(NS, "title");
    title.textContent = `${c.bookId} p.${c.pageStart} T${labels[i]}: ${c.text.slice(0, 140)}`;
    node.append(title);
    node.addEventListener("click", () => showSimilar(c));
    node.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); showSimilar(c); } });
    svg.append(node);
  });
  const outs = [...(state.outliers || [])];
  $("outlierBox").replaceChildren(...(outs.length ? [el("h3", { class: "small" }, `Outliers (${outs.length}): passages far from their theme centre`),
    el("ul", { class: "list" }, ...outs.map((i) => { const b = el("button", { class: "linkish", type: "button" }, `${sample[i].bookId} ${pageLabel(sample[i])}, T${labels[i]}: ${sample[i].text.slice(0, 90)}`); b.onclick = () => showSimilar(sample[i]); return el("li", { class: "small" }, el("span", { class: "flag", "aria-hidden": "true" }, "!"), " ", b); }))] : []));
  $("bookLegend").textContent = "Shapes: " + books.map((b, i) => `${["circle", "square", "diamond"][i % 3]} = ${b}`).join(", ");
}

function showSimilar(chunk) {
  const out = $("similar");
  const res = core.similarAcrossBooks(chunk, state.chunks, 5);
  out.replaceChildren(
    el("div", { class: "quote-card" }, el("p", { class: "eyebrow" }, "Anchor passage"), el("p", {}, chunk.text.slice(0, 400)), chip(chunk)),
    el("h3", {}, "Closest passages in other books"));
  if (!res.length) out.append(el("div", { class: "empty" }, el("h3", {}, "Nothing to compare yet"), el("p", { class: "muted" }, "Load a second book to compare across texts.")));
  for (const [c, s] of res) out.append(el("div", { class: "sim-item" }, el("div", { class: "row between" }, chip(c), el("span", { class: "score" }, `similarity ${s.toFixed(2)}`)), el("p", {}, c.text.slice(0, 400))));
  showTab("similar");
}

function renderSimilarEmpty() {
  if ($("similar").childElementCount) return;
  $("similar").append(el("div", { class: "empty" }, el("h3", {}, "Pick a passage"), el("p", { class: "muted" }, "Choose a point on the theme map or open a citation and select Similar passages.")));
}

// ---------- Compare, manifest ----------
async function runCompare() {
  const q = $("ask").value.trim().slice(0, 500);
  const out = $("compareOut");
  if (!q) { toast("Type a question first", "error"); $("ask").focus(); return; }
  if (state.books.length < 2) { out.replaceChildren(el("div", { class: "empty" }, el("h3", {}, "Nothing to compare yet"), el("p", { class: "muted" }, "Load a second book to compare across texts."))); return; }
  const cols = await api().compare(q, $("compareBy").value);
  out.replaceChildren(...cols.map(({ group, result }) => el("section", { class: "compare-col" },
    el("h3", {}, group, " ", el("span", { class: "badge" }, result.refused ? "abstains" : `${result.citations.length} cited`)),
    ...(result.refused ? [el("p", { class: "muted small" }, `${NOT_FOUND} for this group (best score ${result.topScore.toFixed(2)}).`)]
      : result.citations.slice(0, 3).map((c) => el("article", { class: "quote-card" }, el("p", {}, c.text.slice(0, 400)), chip(c)))))));
}

function renderManifest() {
  const rows = lib.manifest(state.books, state.chunks);
  const t = $("manifestTable");
  if (!t) return;
  const cols = ["title", "role", "tradition", "date", "licenseStatus", "passages"];
  t.replaceChildren(el("thead", {}, el("tr", {}, ...cols.map((c) => el("th", { scope: "col" }, c)))),
    el("tbody", {}, ...(rows.length ? rows.map((r) => el("tr", {}, ...cols.map((c) => el("td", {}, String(r[c] ?? ""))))) : [el("tr", {}, el("td", { colspan: String(cols.length), class: "muted" }, "No books yet."))])));
}

function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = el("a", { href: url, download: name });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ---------- Tabs ----------
const TABS = ["ask", "themes", "similar", "compare", "archive"];
function showTab(name) {
  for (const t of TABS) {
    const on = t === name;
    const b = $(`tab-${t}`);
    b.setAttribute("aria-selected", String(on));
    b.tabIndex = on ? 0 : -1;
    $(`panel-${t}`).hidden = !on;
  }
  if (name === "themes") buildThemes();
  if (name === "similar") renderSimilarEmpty();
  document.dispatchEvent(new CustomEvent("srag:tab", { detail: name }));
}

// ---------- Onboarding ----------
function onboarding(force = false) {
  const dlg = $("onboard");
  try { if (!force && localStorage.getItem("srag-onboarded")) return; } catch { /* storage blocked */ }
  let step = 0;
  const steps = dlg.querySelectorAll(".ob-step");
  const render = () => {
    steps.forEach((s, i) => { s.hidden = i !== step; });
    dlg.querySelectorAll(".steps li").forEach((d, i) => d.classList.toggle("on", i <= step));
    dlg.querySelector(".steps").setAttribute("aria-label", `Step ${step + 1} of 3`);
    $("ob-back").disabled = step === 0;
    $("ob-next").textContent = step === 2 ? "Get started" : "Next";
  };
  const done = () => { try { localStorage.setItem("srag-onboarded", "1"); } catch { /* ignore */ } dlg.close(); };
  $("ob-next").onclick = () => { if (step < 2) { step++; render(); $("ob-next").focus(); } else { done(); $("ask").focus(); } };
  $("ob-back").onclick = () => { if (step > 0) { step--; render(); } };
  $("ob-skip").onclick = done;
  dlg.onclose = () => { try { localStorage.setItem("srag-onboarded", "1"); } catch { /* ignore */ } };
  render();
  dlg.showModal();
  $("ob-next").focus();
}

// ---------- Theme ----------
function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem("srag-theme", t); } catch { /* ignore */ }
  $("themeToggle").setAttribute("aria-label", t === "dark" ? "Switch to light mode" : "Switch to dark mode");
}

// ---------- Wiring ----------
function wire() {
  $("askForm").addEventListener("submit", (e) => { e.preventDefault(); ask(); });
  document.querySelectorAll("[data-tab]").forEach((b) => { b.onclick = () => showTab(b.dataset.tab); });
  $("tab-ask").parentElement.addEventListener("keydown", (e) => {
    const i = TABS.indexOf(document.activeElement?.dataset?.tab);
    if (i < 0) return;
    let n = null;
    if (e.key === "ArrowRight") n = (i + 1) % TABS.length;
    if (e.key === "ArrowLeft") n = (i + TABS.length - 1) % TABS.length;
    if (e.key === "Home") n = 0;
    if (e.key === "End") n = TABS.length - 1;
    if (n != null) { e.preventDefault(); showTab(TABS[n]); $(`tab-${TABS[n]}`).focus(); }
  });

  const dz = $("dropzone");
  dz.onclick = () => $("files").click();
  dz.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("files").click(); } };
  dz.ondragover = (e) => { e.preventDefault(); dz.classList.add("over"); };
  dz.ondragleave = () => dz.classList.remove("over");
  dz.ondrop = (e) => { e.preventDefault(); dz.classList.remove("over"); onFiles(e.dataTransfer.files); };
  $("files").onchange = (e) => onFiles(e.target.files);
  $("sample").onclick = loadSample;
  $("libFilter").oninput = renderLibrary;
  $("libSort").onchange = renderLibrary;
  $("libRole").onchange = renderLibrary;
  $("compareGo").onclick = () => runCompare().catch((e) => toast(`Compare failed: ${e.message}`, "error"));
  $("manifestCsv").onclick = () => download("library-manifest.csv", lib.manifestCsv(lib.manifest(state.books, state.chunks)), "text/csv");
  $("manifestJson").onclick = () => download("library-manifest.json", JSON.stringify(lib.manifest(state.books, state.chunks), null, 2), "application/json");
  $("historySearch").oninput = renderHistory;
  $("recluster").onclick = () => { state.themes = null; buildThemes(); };
  $("threshold").oninput = () => { $("thresholdVal").textContent = $("threshold").value; };

  $("embedder").onchange = async (e) => {
    const prev = state.embedder;
    state.embedder = e.target.value;
    try {
      setProgress("ingest", "Re-embedding stored passages", NaN);
      const norm = await getNormalizer();
      const vecs = await embed(state.chunks.map((c) => (c.indexText ??= norm.text(c.text))), false);
      state.chunks.forEach((c, i) => { c.vec = vecs[i]; });
      for (const b of state.books) await db.saveBook(state.db, b, state.byBook.get(b.bookId) || []);
      await db.put(state.db, "meta", { k: "embedder", v: state.embedder });
      toast("Embeddings updated");
    } catch (err) {
      state.embedder = prev; e.target.value = prev;
      toast(err.message, "error", 6000);
    }
    setProgress("ingest", "", null);
    await refresh();
  };

  $("export").onclick = async () => {
    const blob = new Blob([await db.exportJson(state.db)], { type: "application/json" });
    const a = el("a", { href: URL.createObjectURL(blob), download: "scripture-rag-library.json" });
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };
  $("import").onchange = async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    try {
      if (f.size > 200 * 1024 * 1024) throw new Error("Import file is larger than 200 MB");
      await db.importJson(state.db, await f.text());
      toast("Library imported");
    } catch (err) { toast(`Import failed: ${err.message}`, "error"); }
    e.target.value = "";
    await refresh();
  };

  const cf = $("confirmDlg");
  $("clear").onclick = () => { cf.showModal(); $("cf-cancel").focus(); };
  $("cf-cancel").onclick = () => cf.close();
  $("cf-ok").onclick = async () => {
    cf.close();
    await db.clearAll(state.db);
    try { localStorage.removeItem("srag-onboarded"); } catch { /* ignore */ }
    state.current = null;
    $("similar").replaceChildren();
    toast("All data deleted from this browser");
    await refresh();
  };

  const libPanel = $("library-panel");
  const openLib = () => { libPanel.classList.add("open"); $("scrim").hidden = false; showDrawer(libPanel, $("libraryToggle")); };
  $("libraryToggle").onclick = openLib;
  $("libraryClose").onclick = () => hideDrawer();
  $("historyToggle").onclick = (e) => showDrawer($("history-drawer"), e.currentTarget);
  document.querySelectorAll("[data-close]").forEach((b) => { b.onclick = () => hideDrawer(); });
  $("scrim").onclick = () => hideDrawer();
  $("themeToggle").onclick = () => applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  $("helpBtn").onclick = () => onboarding(true);

  document.addEventListener("keydown", (e) => {
    trapFocus(e);
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName) || document.activeElement?.isContentEditable;
    if (e.key === "Escape") {
      if (drawerIsOpen()) { e.preventDefault(); hideDrawer(); return; }
      if (document.activeElement === $("ask") && $("ask").value) { $("ask").value = ""; return; }
    }
    if (typing || e.metaKey || e.ctrlKey || e.altKey || document.querySelector("dialog[open]")) return;
    if (e.key === "/") { e.preventDefault(); $("ask").focus(); $("ask").select(); }
    else if (e.key === "h" || e.key === "H") { e.preventDefault(); showDrawer($("history-drawer")); }
    else if (e.key === "t" || e.key === "T") applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  });
}

async function main() {
  applyTheme(document.documentElement.dataset.theme || "light");
  wire();
  showTab("ask");
  renderSkeleton();
  state.db = await db.openDb();
  await refresh();
  onboarding();
}

// Data and API hooks for the interface layer (compare, roles, outliers, manifest).
globalThis.ScriptureRAG = {
  core, ia, lib, rag, state,
  async setRole(bookId, role) {
    if (!lib.ROLES.includes(role)) throw new Error("Unknown role");
    const b = state.books.find((x) => x.bookId === bookId);
    if (!b) return;
    b.role = role;
    await db.put(state.db, "books", b);
    await refresh();
  },
  async compare(question, by = "tradition") {
    const [qv] = await embed([question], true);
    return lib.compare(question, state.books, state.chunks, qv, { by, threshold: Number($("threshold")?.value || 0.5) });
  },
  async answerByRole(question) {
    const [qv] = await embed([question], true);
    return lib.answerByRole(question, state.books, state.chunks, qv, { threshold: Number($("threshold")?.value || 0.5) });
  },
  outliers: () => (state.themes ? lib.outliers(state.themes.sample.map((c) => c.vec), state.themes.labels).map((o) => ({ ...o, chunk: state.themes.sample[o.index] })) : []),
  manifest: () => lib.manifest(state.books, state.chunks),
  manifestCsv: () => lib.manifestCsv(lib.manifest(state.books, state.chunks)),
  roleOf: (bookId) => state.books.find((b) => b.bookId === bookId)?.role || "primary",
  // Ingest hooks used by archive-ui.js: bytes -> pages -> indexed book with source metadata.
  async importBytes(name, bytes, extra = {}) {
    const pages = await readPages(new File([bytes], name, { type: name.endsWith(".pdf") ? "application/pdf" : "text/plain" }), name);
    await addBook(name, pages, extra);
    setProgress("ingest", "", null);
    await refresh();
  },
  embed: (texts, isQuery = false) => embed(texts, isQuery),
  setSearchBooks(ids) { state.searchIds = lib.keepLoaded(new Set(ids), state.books); renderSearchBooks(); },
  cacheGet: async (k) => (await db.getAll(state.db, "meta")).find((m) => m.k === k),
  cachePut: (v) => db.put(state.db, "meta", v),
  toast,
};

main().then(() => import("./archive-ui.js")).then((m) => m.mount(document.getElementById("archive-mount"))).catch((e) => renderError(e.message));
