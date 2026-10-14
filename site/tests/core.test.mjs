// Unit tests for the browser core: run with `node --test site/tests/`.
import test from "node:test";
import assert from "node:assert/strict";
import * as core from "../js/core.js";

const pages = ["The LORD is my shepherd; I shall not want.", "For hatred does not cease by hatred at any time: hatred ceases by love."];

test("chunkPages keeps page ranges and overlap", () => {
  const chunks = core.chunkPages("b", ["a b c d e", "f g h i j"], 4, 1);
  assert.equal(chunks[0].pageStart, 1);
  assert.equal(chunks.at(-1).pageEnd, 2);
  assert.equal(chunks[0].text.split(" ").at(-1), chunks[1].text.split(" ")[0]);
});

test("rrf and topK", () => {
  const fused = Object.fromEntries(core.rrf([["x", "y"], ["y", "z"]]));
  assert.ok(Math.abs(fused.y - (1 / 61 + 1 / 62)) < 1e-12);
  assert.deepEqual(core.topK([["a", 1], ["b", 3], ["c", 2]], 2).map((p) => p[0]), ["b", "c"]);
});

test("answer cites or refuses", () => {
  const chunks = pages.flatMap((p, i) => core.chunkPages(`book${i}`, [p], 50, 5)).map((c) => ({ ...c, vec: core.hashEmbed(c.text) }));
  const bm = new core.BM25();
  chunks.forEach((c) => bm.add(c.chunkId, c.text));
  const q = "How does hatred cease?";
  const ok = core.answer(q, chunks, bm, core.hashEmbed(q));
  assert.equal(ok.refused, false);
  assert.match(ok.text, /\[book1 p\.1 #book1-0000\]/);
  const no = core.answer("capital of France", chunks, bm, core.hashEmbed("capital of France"));
  assert.equal(no.text, core.REFUSAL_TEXT);
});

test("kmeans separates two blobs deterministically and labels with keywords", () => {
  const vecs = [...Array(6)].map((_, i) => Float32Array.from(i < 3 ? [1, 0.01 * i] : [0.01 * i, 1]));
  const a = core.kmeans(vecs, 2, { seed: 1 }).labels, b = core.kmeans(vecs, 2, { seed: 1 }).labels;
  assert.deepEqual(a, b);
  assert.notEqual(a[0], a[5]);
  assert.ok(core.silhouette(vecs, a) > 0.9);
  assert.equal(core.autoCluster(vecs).k, 2);
  const kw = core.clusterKeywords(["water river", "water sea", "water lake", "fire flame", "fire heat", "fire ash"], a);
  assert.ok(kw.flat().includes("water") && kw.flat().includes("fire"));
  assert.equal(core.pca2d(vecs).length, 6);
});

test("similar passages exclude the same book", () => {
  const chunks = [["a", "x y"], ["a", "x y z"], ["b", "x y"]].map(([bookId, t], i) => ({ chunkId: String(i), bookId, text: t, vec: core.hashEmbed(t) }));
  const res = core.similarAcrossBooks(chunks[0], chunks);
  assert.deepEqual(res.map(([c]) => c.bookId), ["b"]);
});

test("sanitizers", () => {
  assert.equal(core.stripControl("a\u0000b\fc"), "ab\fc");
  assert.equal(core.safeBookId("My Book (1881).pdf"), "my_book_1881");
  assert.deepEqual(core.splitTxtPages("a\fb"), ["a", "b"]);
});
