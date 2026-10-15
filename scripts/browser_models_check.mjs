// Run the browser pipeline (site/js/rag.js) with the browser models (transformers.js ONNX: e5-small
// embeddings, mMiniLM cross-encoder, opus-mt translation) on the exported eval inputs and compare the
// answer/refuse decisions with Python (torch). Passage vectors are re-embedded with transformers.js.
// Usage: TRANSFORMERS=/path/to/@huggingface/transformers node scripts/browser_models_check.mjs [name]
// Writes docs/eval/parity_models.json. Needs network on first run (model download).
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import * as rag from "../site/js/rag.js";
import { MODELS } from "../site/js/models.js";

const tf = await import(process.env.TRANSFORMERS || "@huggingface/transformers");
const name = process.argv[2] || "eval";
const DIR = "data/cache/parity";
const { chunks: pyChunks, index_texts: pyIndex, dim } = JSON.parse(readFileSync(`${DIR}/chunks.json`, "utf8"));
const { policy, candidates, bm25, rrf_k: rrfK, items } = JSON.parse(readFileSync(`${DIR}/items_${name}.json`, "utf8"));
const buf = readFileSync(`${DIR}/pvec.f32`);
const pyVec = new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);

const embedder = await tf.pipeline("feature-extraction", MODELS.embed.id, { dtype: MODELS.embed.dtype });
const embed = async (texts, isQuery) => (await embedder(texts.map((t) => (isQuery ? "query: " : "passage: ") + t), { pooling: "mean", normalize: true })).tolist();
const tok = await tf.AutoTokenizer.from_pretrained(MODELS.rerank.id);
const ce = await tf.AutoModelForSequenceClassification.from_pretrained(MODELS.rerank.id, { dtype: MODELS.rerank.dtype, model_file_name: MODELS.rerank.file });
const rerank = async (q, texts) => {
  const out = [];
  for (let i = 0; i < texts.length; i += 16) {
    const part = texts.slice(i, i + 16);
    const enc = tok(part.map(() => q), { text_pair: part, padding: true, truncation: true });
    const { logits } = await ce(enc);
    for (const x of logits.data) out.push(1 / (1 + Math.exp(-x)));
  }
  return out;
};
const mt = await tf.pipeline("translation", MODELS.translate.id, { dtype: MODELS.translate.dtype });
const translate = async (q) => (await mt(q, MODELS.translate.generate))[0].translation_text;

const cacheFile = `${DIR}/pvec_js_${MODELS.embed.dtype}.f32`;
let jsVec;
if (existsSync(cacheFile)) { const b = readFileSync(cacheFile); jsVec = new Float32Array(b.buffer, b.byteOffset, b.length / 4); }
else {
  jsVec = new Float32Array(pyChunks.length * dim);
  const t0 = Date.now();
  for (let i = 0; i < pyIndex.length; i += 32) {
    (await embed(pyIndex.slice(i, i + 32), false)).forEach((v, j) => jsVec.set(v, (i + j) * dim));
    if (i % 1024 === 0) console.error(`embedded ${i}/${pyIndex.length} in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
  }
  writeFileSync(cacheFile, Buffer.from(jsVec.buffer));
}
let cosMin = 1;
for (let i = 0; i < pyChunks.length; i++) {
  let d = 0;
  for (let j = 0; j < dim; j++) d += jsVec[i * dim + j] * pyVec[i * dim + j];
  cosMin = Math.min(cosMin, d);
}
const index = new rag.RagIndex(null, bm25 ? { k1: bm25[0], b: bm25[1] } : {});
index.add(pyChunks.map(([chunkId, bookId, text, pageStart, pageEnd, ref], i) => ({ chunkId, bookId, text, pageStart, pageEnd, ref, indexText: pyIndex[i] })),
  pyChunks.map((_, i) => jsVec.subarray(i * dim, (i + 1) * dim)));
const s = { set: name, n: items.length, models: MODELS, passage_cosine_min: cosMin, decision_agree: 0, context_agree: 0,
  translation_identical: 0, translated: 0, by_split: {}, disagreements: [], ms: { translate: [], retrieve: [], rerank: [] } };
for (const it of items) {
  const res = await rag.ragAnswer(it.question, index, { translate, rerank, embedQuery: async (t) => (await embed([t], true))[0] },
    { thresholds: policy.t, minSupport: policy.s_min, candidates, rrfK: rrfK ?? 60, defaultThreshold: policy.t.en });
  if (res.trace) {
    for (const k of Object.keys(s.ms)) s.ms[k].push(res.trace.ms[k]);
    if (it.lang !== "en" && res.lang !== "en") { s.translated++; if (res.trace.qEn === it.q_en) s.translation_identical++; }
  }
  const ctx = res.refused ? [] : res.citations.map((c) => c.chunkId);
  const agree = !res.refused === it.py.answered;
  const b = (s.by_split[it.split || "all"] ??= { n: 0, agree: 0 });
  b.n++; if (agree) b.agree++;
  if (agree) s.decision_agree++;
  if (agree && JSON.stringify(ctx) === JSON.stringify(it.py.context)) s.context_agree++;
  else s.disagreements.push({ id: it.id, python_answered: it.py.answered, browser_answered: !res.refused, python: it.py.context, browser: ctx, q_en_python: it.q_en, q_en_browser: res.trace?.qEn, top_browser: res.topScore, top_python: it.py.ranked[0]?.[1] });
}
const med = (a) => { const x = [...a].sort((p, q) => p - q); return x.length ? Math.round(x[Math.floor(x.length / 2)]) : null; };
s.median_ms = Object.fromEntries(Object.entries(s.ms).map(([k, v]) => [k, med(v)]));
delete s.ms;
writeFileSync(`docs/eval/parity_models${name === "eval" ? "" : "_" + name}.json`, JSON.stringify(s, null, 2) + "\n");
console.log(JSON.stringify({ ...s, disagreements: s.disagreements.length }, null, 2));
