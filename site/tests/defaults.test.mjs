import test from "node:test";
import assert from "node:assert/strict";
import { DEFAULTS } from "../js/rag.js";

test("browser defaults match the shipped Python gate", () => {
  assert.deepEqual({ ...DEFAULTS.thresholds }, { en: 0.12, pt: 0.72, es: 0.16 });
  assert.equal(DEFAULTS.minSupport, 0.3);
  assert.equal(DEFAULTS.candidates, 50);
  assert.equal(DEFAULTS.bm25K1, 1.5);
  assert.equal(DEFAULTS.rrfK, 60);
});

test("Portuguese questions get the unsupported message without retrieval", async () => {
  const { RagIndex, ragAnswer, PT_UNSUPPORTED_TEXT } = await import("../js/rag.js");
  const index = new RagIndex();
  index.chunks.push({ chunkId: "x-0001", bookId: "x", text: "The Lord is my shepherd", pageStart: 1, pageEnd: 1 });
  const fail = async () => { throw new Error("models must not be called"); };
  const res = await ragAnswer("Quem é o meu pastor? O Senhor não me deixa faltar nada?", index, { translate: fail, embedQuery: fail, rerankPassages: fail });
  assert.equal(res.unsupportedLanguage, true);
  assert.equal(res.refused, true);
  assert.equal(res.text, PT_UNSUPPORTED_TEXT);
  assert.equal(DEFAULTS.portuguese, false);
});
