// Browser demo UI: load PDF/TXT, ask, cited passages only, theme map, similar passages, history.
import * as core from "./core.js";
import * as db from "./db.js";

const PDFJS = "https://cdn.jsdelivr.net/npm/pdfjs-dist@6.4.299/build/pdf.min.mjs";
const PDFJS_WORKER = "https://cdn.jsdelivr.net/npm/pdfjs-dist@6.4.299/build/pdf.worker.min.mjs";
const TRANSFORMERS = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1";
const MODEL = "Xenova/multilingual-e5-small";
const MAX_BYTES = 50 * 1024 * 1024;
const MAX_PAGES = 2000;

const $ = (id) => document.getElementById(id);
const state = { db: null, chunks: [], books: [], bm25: new core.BM25(), embedder: "hashing", model: null, themes: null };

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  for (const c of children) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}

function status(msg, kind = "info") {
  const s = $("status");
  s.textContent = msg;
  s.dataset.kind = kind;
}

async function embed(texts, isQuery) {
  if (!texts.length) return [];
  if (state.embedder === "hashing") return texts.map((t) => core.hashEmbed(t));
  if (!state.model) {
    status("Downloading the multilingual embedding model (first time only)...");
    const { pipeline } = await import(TRANSFORMERS);
    state.model = await pipeline("feature-extraction", MODEL);
  }
  const prefixed = texts.map((t) => (isQuery ? "query: " : "passage: ") + t);
  const out = await state.model(prefixed, { pooling: "mean", normalize: true });
  return out.tolist().map((v) => Float32Array.from(v));
}

async function readPages(file) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  if (bytes.length > MAX_BYTES) throw new Error("File is larger than 50 MB");
  const ext = file.name.toLowerCase().split(".").pop();
  const isPdf = String.fromCharCode(...bytes.slice(0, 5)) === "%PDF-";
  if (ext === "pdf") {
    if (!isPdf) throw new Error("File has a .pdf extension but is not a PDF");
    const pdfjs = await import(PDFJS);
    pdfjs.GlobalWorkerOptions.workerSrc = PDFJS_WORKER;
    // Scripting disabled; only the text layer is read, links and annotations are ignored.
    const doc = await pdfjs.getDocument({ data: bytes, isEvalSupported: false, enableXfa: false }).promise;
    if (doc.numPages > MAX_PAGES) throw new Error(`PDF has ${doc.numPages} pages; limit is ${MAX_PAGES}`);
    const pages = [];
    for (let p = 1; p <= doc.numPages; p++) {
      const content = await (await doc.getPage(p)).getTextContent();
      pages.push(content.items.map((i) => i.str).join(" "));
    }
    return pages;
  }
  if (ext !== "txt") throw new Error("Unsupported file type; use .pdf or .txt");
  if (isPdf) throw new Error("File has a .txt extension but is a PDF");
  const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  return core.splitTxtPages(text);
}

function rebuildLexical() {
  state.bm25 = new core.BM25();
  for (const c of state.chunks) state.bm25.add(c.chunkId, c.text);
}

async function refresh() {
  state.books = await db.getAll(state.db, "books");
  state.chunks = await db.getAll(state.db, "chunks");
  const meta = await db.getAll(state.db, "meta");
  state.embedder = meta.find((m) => m.k === "embedder")?.v || "hashing";
  $("embedder").value = state.embedder;
  rebuildLexical();
  renderLibrary();
  await renderHistory();
  state.themes = null;
  if (state.chunks.length >= 3) buildThemes();
}

function renderLibrary() {
  const list = $("library");
  list.replaceChildren();
  if (!state.books.length) list.append(el("li", { class: "muted" }, "No books yet. Load a PDF or TXT, or try the sample."));
  for (const b of state.books) {
    const n = state.chunks.filter((c) => c.bookId === b.bookId).length;
    const rm = el("button", { class: "link", title: "Remove" }, "remove");
    rm.onclick = async () => { await db.deleteBook(state.db, b.bookId); await refresh(); };
    list.append(el("li", {}, el("strong", {}, b.title), ` ${b.pages} pages, ${n} passages `, rm));
  }
  $("count").textContent = `${state.books.length} books, ${state.chunks.length} passages`;
}

async function addBook(name, pages) {
  let bookId = core.safeBookId(name), i = 2;
  while (state.books.some((b) => b.bookId === bookId)) bookId = `${core.safeBookId(name)}-${i++}`;
  const clean = pages.map(core.stripControl);
  const chunks = core.chunkPages(bookId, clean, Number($("chunkSize").value) || 50, 10);
  if (!chunks.length) throw new Error("No text found (scanned PDFs need OCR first)");
  const vecs = await embed(chunks.map((c) => c.text), false);
  chunks.forEach((c, k) => { c.vec = vecs[k]; });
  await db.saveBook(state.db, { bookId, title: name.replace(/\.[^.]+$/, ""), pages: clean.length, addedAt: Date.now() }, chunks);
}

async function onFiles(files) {
  for (const f of files) {
    try {
      status(`Reading ${f.name}...`);
      await addBook(f.name, await readPages(f));
      status(`Loaded ${f.name}`, "ok");
    } catch (e) {
      status(`Rejected ${f.name}: ${e.message}`, "error");
    }
  }
  await refresh();
}

function renderAnswer(container, res) {
  container.replaceChildren();
  if (res.refused) {
    container.append(el("p", { class: "refusal" }, res.text), el("p", { class: "muted" }, `Top score ${res.topScore.toFixed(2)} is below the threshold.`));
    return;
  }
  for (const c of res.citations) {
    const details = el("details", { open: "" },
      el("summary", {}, `${c.bookId}, PDF page ${c.pageStart}${c.pageEnd !== c.pageStart ? "-" + c.pageEnd : ""}, #${c.chunkId}`),
      el("blockquote", {}, c.text));
    container.append(details);
  }
}

async function ask() {
  const q = $("question").value.trim();
  if (!q) return;
  const [qv] = await embed([q], true);
  const res = core.answer(q, state.chunks, state.bm25, qv, {
    threshold: Number($("threshold").value),
    // Uncalibrated starting range for e5 cosine; tune on a gold set.
    cosRange: state.embedder === "hashing" ? null : [0.8, 0.9],
  });
  renderAnswer($("answer"), res);
  await db.put(state.db, "history", {
    ts: Date.now(), question: q, refused: res.refused, topScore: res.topScore,
    citations: res.citations.map(({ chunkId, bookId, pageStart, pageEnd, text }) => ({ chunkId, bookId, pageStart, pageEnd, text })),
  });
  await renderHistory();
}

async function renderHistory() {
  const filter = $("historySearch").value.toLowerCase();
  const items = (await db.getAll(state.db, "history")).reverse()
    .filter((h) => !filter || h.question.toLowerCase().includes(filter) || h.citations.some((c) => c.text.toLowerCase().includes(filter)));
  const list = $("history");
  list.replaceChildren();
  for (const h of items.slice(0, 100)) {
    const open = el("button", { class: "link" }, h.question);
    open.onclick = () => {
      $("question").value = h.question;
      renderAnswer($("answer"), { refused: h.refused, text: core.REFUSAL_TEXT, citations: h.citations, topScore: h.topScore });
      showTab("ask");
    };
    list.append(el("li", {}, open, el("span", { class: "muted" }, ` ${new Date(h.ts).toLocaleString()} ${h.refused ? "refused" : h.citations.length + " citations"}`)));
  }
  if (!items.length) list.append(el("li", { class: "muted" }, "No questions yet."));
}

function buildThemes() {
  if (state.chunks.length < 3) { $("themeInfo").textContent = "Load at least three passages."; return; }
  const sample = state.chunks.slice(0, 1500); // keeps the in-browser silhouette search responsive
  const vecs = sample.map((c) => c.vec);
  const t0 = performance.now();
  const { labels, k, silhouette } = core.autoCluster(vecs, Number($("maxK").value) || 8);
  const coords = core.pca2d(vecs);
  const keywords = core.clusterKeywords(sample.map((c) => c.text), labels);
  state.themes = { sample, labels, coords, keywords };
  $("themeInfo").textContent = `${k} themes from ${sample.length} passages, silhouette ${silhouette.toFixed(3)}, ${Math.round(performance.now() - t0)} ms. Clusters are statistical similarity, not doctrinal claims.`;
  const legend = $("themeLegend");
  legend.replaceChildren(...keywords.map((kw, i) => el("li", {}, el("span", { class: `dot c${i % 10}` }), `T${i}: ${kw.join(", ")}`)));
  drawMap();
}

function drawMap() {
  const svg = $("themeMap");
  const { sample, labels, coords } = state.themes;
  const xs = coords.map((p) => p[0]), ys = coords.map((p) => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const sx = (x) => 30 + ((x - x0) / (x1 - x0 || 1)) * 540, sy = (y) => 30 + ((y - y0) / (y1 - y0 || 1)) * 340;
  const books = [...new Set(sample.map((c) => c.bookId))];
  const shapes = ["circle", "rect", "diamond"];
  svg.replaceChildren();
  const NS = "http://www.w3.org/2000/svg";
  sample.forEach((c, i) => {
    const shape = shapes[books.indexOf(c.bookId) % shapes.length];
    const x = sx(coords[i][0]), y = sy(coords[i][1]);
    let node;
    if (shape === "circle") { node = document.createElementNS(NS, "circle"); node.setAttribute("cx", x); node.setAttribute("cy", y); node.setAttribute("r", 5); }
    else { node = document.createElementNS(NS, "rect"); node.setAttribute("x", x - 4); node.setAttribute("y", y - 4); node.setAttribute("width", 8); node.setAttribute("height", 8); if (shape === "diamond") node.setAttribute("transform", `rotate(45 ${x} ${y})`); }
    node.setAttribute("class", `pt c${labels[i] % 10}`);
    const title = document.createElementNS(NS, "title");
    title.textContent = `${c.bookId} p.${c.pageStart} T${labels[i]}: ${c.text.slice(0, 140)}`;
    node.append(title);
    node.addEventListener("click", () => showSimilar(c));
    svg.append(node);
  });
  $("bookLegend").textContent = "Shapes: " + books.map((b, i) => `${["circle", "square", "diamond"][i % 3]} = ${b}`).join(", ");
}

function showSimilar(chunk) {
  const out = $("similar");
  out.replaceChildren(el("p", {}, el("strong", {}, `Anchor ${core.citation(chunk)} `), chunk.text.slice(0, 240)));
  const res = core.similarAcrossBooks(chunk, state.chunks, 5);
  if (!res.length) out.append(el("p", { class: "muted" }, "Load a second book to compare across texts."));
  for (const [c, s] of res) out.append(el("details", {}, el("summary", {}, `${s.toFixed(2)} ${core.citation(c)}`), el("blockquote", {}, c.text)));
  showTab("similar");
}

function showTab(name) {
  document.querySelectorAll("[data-tab]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  document.querySelectorAll(".panel").forEach((p) => { p.hidden = p.id !== `panel-${name}`; });
}

async function loadSample() {
  const res = await fetch("assets/sample/psalm23_kjv.txt");
  await addBook("psalm23_kjv.txt", core.splitTxtPages(await res.text()));
  await refresh();
}

async function main() {
  state.db = await db.openDb();
  document.querySelectorAll("[data-tab]").forEach((b) => { b.onclick = () => showTab(b.dataset.tab); });
  $("files").onchange = (e) => onFiles([...e.target.files]);
  $("askBtn").onclick = ask;
  $("question").onkeydown = (e) => { if (e.key === "Enter") ask(); };
  $("historySearch").oninput = renderHistory;
  $("recluster").onclick = buildThemes;
  $("sample").onclick = loadSample;
  $("threshold").oninput = () => { $("thresholdVal").textContent = $("threshold").value; };
  $("embedder").onchange = async (e) => {
    state.embedder = e.target.value;
    await db.put(state.db, "meta", { k: "embedder", v: state.embedder });
    status("Re-embedding stored passages...");
    const vecs = await embed(state.chunks.map((c) => c.text), false);
    state.chunks.forEach((c, i) => { c.vec = vecs[i]; });
    for (const b of state.books) await db.saveBook(state.db, b, state.chunks.filter((c) => c.bookId === b.bookId));
    status("Embeddings updated", "ok");
    await refresh();
  };
  $("export").onclick = async () => {
    const blob = new Blob([await db.exportJson(state.db)], { type: "application/json" });
    const a = el("a", { href: URL.createObjectURL(blob), download: "scripture-rag-library.json" });
    a.click();
    URL.revokeObjectURL(a.href);
  };
  $("import").onchange = async (e) => {
    try { await db.importJson(state.db, await e.target.files[0].text()); status("Library imported", "ok"); }
    catch (err) { status(`Import failed: ${err.message}`, "error"); }
    await refresh();
  };
  $("clear").onclick = async () => {
    if (confirm("Delete all books, passages and history stored in this browser?")) { await db.clearAll(state.db); await refresh(); }
  };
  showTab("ask");
  await refresh();
}

main().catch((e) => status(e.message, "error"));
