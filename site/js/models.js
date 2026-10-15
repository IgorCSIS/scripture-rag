// Browser model choices, shared by app.js and the Node parity check (scripts/browser_models_check.mjs).
// Same weights as the Python pipeline: intfloat/multilingual-e5-small (MIT), the official ONNX export of
// cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (Apache-2.0) and Helsinki-NLP/opus-mt-ROMANCE-en (Apache-2.0).
export const MODELS = Object.freeze({
  embed: { id: "Xenova/multilingual-e5-small", dtype: "fp32" },
  rerank: { id: "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", dtype: "fp32", file: "model" },
  translate: { id: "Xenova/opus-mt-ROMANCE-en", dtype: "fp32", generate: { max_new_tokens: 96, num_beams: 4 } },
});
