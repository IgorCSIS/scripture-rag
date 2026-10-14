// Browser-side index benchmark (run under Node): brute force vs ann.js HNSW, d=384 unit vectors.
// Writes docs/benchmarks/browser_ann.json.
import { writeFileSync } from "node:fs";
import { HNSW } from "../site/js/ann.js";
import { vectorSearch } from "../site/js/core.js";

const d = 384;
let s = 11;
const r = () => ((s = (s * 1103515245 + 12345) >>> 0) / 4294967296);
const unit = () => { const v = Float32Array.from({ length: d }, () => r() - 0.5); const n = Math.hypot(...v); return v.map((x) => x / n); };
const pct = (a, p) => [...a].sort((x, y) => x - y)[Math.floor(p * (a.length - 1))];
const rows = [];
for (const n of [1000, 5000, 20000, 50000]) {
  const centers = Array.from({ length: 50 }, unit);
  const chunks = Array.from({ length: n }, (_, i) => {
    const c = centers[i % 50], v = unit();
    const m = c.map((x, j) => x + 0.6 * v[j]), nn = Math.hypot(...m);
    return { chunkId: `c${i}`, vec: m.map((x) => x / nn) };
  });
  const qs = Array.from({ length: 50 }, (_, i) => chunks[(i * 7919) % n].vec.map((x, j) => x + 0.05 * (r() - 0.5)));
  let t = performance.now();
  const ix = new HNSW();
  chunks.forEach((c) => ix.add(c.vec));
  const build = (performance.now() - t) / 1000;
  const tb = [], th = [];
  let hit = 0;
  for (const q of qs) {
    t = performance.now(); const ex = vectorSearch(chunks, q, 10, { ann: null }); tb.push(performance.now() - t);
    t = performance.now(); const ap = ix.search(q, 10); th.push(performance.now() - t);
    const e = new Set(ex.map(([id]) => id)); hit += ap.filter(([i]) => e.has(chunks[i].chunkId)).length;
  }
  const row = { n, brute_p50_ms: pct(tb, 0.5), brute_p95_ms: pct(tb, 0.95), hnsw_p50_ms: pct(th, 0.5),
    hnsw_p95_ms: pct(th, 0.95), hnsw_build_s: build, recall_at_10: hit / 500 };
  console.log(JSON.stringify(row));
  rows.push(row);
}
writeFileSync("docs/benchmarks/browser_ann.json", JSON.stringify(rows, null, 1));
