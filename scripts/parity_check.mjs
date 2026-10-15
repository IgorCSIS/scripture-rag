// Replay the browser pipeline (site/js/rag.js) on the inputs exported by scripts/export_parity.py and
// compare chunks, index texts, candidates, scores and answer/refuse decisions with Python.
// Usage: node scripts/parity_check.mjs [name ...]   (default: eval). Writes docs/eval/parity.json.
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import * as rag from "../site/js/rag.js";

const DIR = "data/cache/parity";
const names = process.argv.slice(2).length ? process.argv.slice(2) : ["eval"];
const pages = JSON.parse(readFileSync(`${DIR}/pages.json`, "utf8"));
const { chunks: pyChunks, index_texts: pyIndex, dim } = JSON.parse(readFileSync(`${DIR}/chunks.json`, "utf8"));
const buf = readFileSync(`${DIR}/pvec.f32`);
const all = new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
const vocab = readFileSync("site/assets/modern_vocab.txt", "utf8").split(/\s+/).filter(Boolean);
const norm = new rag.ArchaicNormalizer(vocab);

// 1. Chunking from the same pages.
const ref = JSON.parse(readFileSync("site/assets/kjv_ref_genesis_joshua.json", "utf8")).map(([book, chapter, verse, text]) => ({ book, chapter, verse, text, page: 1 }));
const aligner = new rag.ReferenceAligner(ref, norm);
const jsChunks = Object.entries(pages).flatMap(([id, p]) => rag.chunkDocument(id, p, { aligner }));
const key = (c) => `${c[0]}\u0001${c[2]}\u0001${c[5]}`;
const pySet = new Set(pyChunks.map(key));
const jsKeys = jsChunks.map((c) => key([c.chunkId, c.bookId, c.text, c.pageStart, c.pageEnd, c.ref]));
const chunkMatch = jsKeys.filter((k) => pySet.has(k)).length;
const perBook = {};
for (const c of pyChunks) (perBook[c[1]] ??= { python: 0, browser: 0, identical: 0 }).python++;
jsChunks.forEach((c, i) => { const b = (perBook[c.bookId] ??= { python: 0, browser: 0, identical: 0 }); b.browser++; if (pySet.has(jsKeys[i])) b.identical++; });

// 2. Index texts (archaic normalization).
const jsIndex = pyChunks.map((c) => norm.text(c[2]));
const indexMatch = jsIndex.filter((t, i) => t === pyIndex[i]).length;

// 3. Retrieval, rerank and gate on the Python chunks with Python vectors and model outputs.
const index = new rag.RagIndex(null);
index.add(pyChunks.map(([chunkId, bookId, text, pageStart, pageEnd, r], i) => ({ chunkId, bookId, text, pageStart, pageEnd, ref: r, indexText: jsIndex[i] })),
  pyChunks.map((_, i) => all.subarray(i * dim, (i + 1) * dim)));
const report = { chunks: { python: pyChunks.length, browser: jsChunks.length, identical: chunkMatch, per_book: perBook },
  index_texts: { n: pyIndex.length, identical: indexMatch }, sets: {} };
for (const name of names) {
  const f = `${DIR}/items_${name}.json`;
  if (!existsSync(f)) continue;
  const { policy, candidates, items } = JSON.parse(readFileSync(f, "utf8"));
  const s = { n: items.length, decision_agree: 0, context_agree: 0, top_agree: 0, lang_agree: 0, missing_scores: 0, max_score_diff: 0, max_support_diff: 0, disagreements: [] };
  for (const it of items) {
    let missing = 0;
    const models = {
      translate: async () => it.q_en,
      embedQuery: async (t) => Float32Array.from(it.qvecs[t]),
      rerankPassages: async (_q, texts) => texts.map((t) => { if (!(t in it.rerank)) { missing++; return 0; } return it.rerank[t]; }),
    };
    const res = await rag.ragAnswer(it.question, index, models, { thresholds: policy.t, minSupport: policy.s_min, candidates, defaultThreshold: policy.t.en });
    if (missing) s.missing_scores++;
    if (res.lang === it.lang) s.lang_agree++;
    const answered = !res.refused;
    const ctx = answered ? res.citations.map((c) => c.chunkId) : [];
    const agree = answered === it.py.answered;
    if (agree) s.decision_agree++;
    if (agree && JSON.stringify(ctx) === JSON.stringify(it.py.context)) s.context_agree++;
    const jr = res.trace?.ranked || [];
    if (jr[0]?.[0] === it.py.ranked[0]?.[0]) s.top_agree++;
    jr.forEach((r, i) => { const p = it.py.ranked[i]; if (p && p[0] === r[0]) { s.max_score_diff = Math.max(s.max_score_diff, Math.abs(p[1] - r[1])); s.max_support_diff = Math.max(s.max_support_diff, Math.abs(p[2] - r[2])); } });
    if (!agree || JSON.stringify(ctx) !== JSON.stringify(it.py.context)) s.disagreements.push({ id: it.id, python: it.py.context, browser: ctx });
  }
  report.sets[name] = s;
}
writeFileSync("docs/eval/parity.json", JSON.stringify(report, null, 2) + "\n");
console.log(JSON.stringify(report, (k, v) => (k === "disagreements" ? v.slice(0, 8) : v), 2));
