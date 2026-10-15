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
