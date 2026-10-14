import test from "node:test";
import assert from "node:assert/strict";
import { HNSW, ANN_THRESHOLD } from "../js/ann.js";
import { vectorSearch } from "../js/core.js";

function unit(d, r) {
  const v = Float32Array.from({ length: d }, () => r() - 0.5);
  const n = Math.hypot(...v);
  return v.map((x) => x / n);
}
function rng(s) { return () => ((s = (s * 1103515245 + 12345) >>> 0) / 4294967296); }

test("HNSW recall@10 against brute force", () => {
  const r = rng(7), d = 32, chunks = Array.from({ length: 3000 }, (_, i) => ({ chunkId: `c${i}`, vec: unit(d, r) }));
  const ix = new HNSW();
  chunks.forEach((c) => ix.add(c.vec));
  let hit = 0;
  for (let q = 0; q < 50; q++) {
    const qv = unit(d, r);
    const exact = new Set(vectorSearch(chunks, qv, 10, { ann: null }).map(([id]) => id));
    hit += ix.search(qv, 10).filter(([i]) => exact.has(chunks[i].chunkId)).length;
  }
  assert.ok(hit / 500 >= 0.9, `recall ${hit / 500}`);
});

test("vectorSearch switches to HNSW above the threshold", async () => {
  await import("../js/ann.js");
  assert.ok(vectorSearch.ann && ANN_THRESHOLD === 20000);
  const r = rng(3), chunks = Array.from({ length: 200 }, (_, i) => ({ chunkId: `c${i}`, vec: unit(16, r) }));
  const res = vectorSearch(chunks, chunks[5].vec, 3, { threshold: 100 });
  assert.equal(res[0][0], "c5");
});
