/**
 * Small HNSW (hierarchical navigable small world) index for unit vectors, inner-product metric.
 *
 * Maintained JS/WASM options were checked first: hnswlib-wasm 0.8.2 has had no release since
 * July 2023, so the browser build ships this dependency-free version instead (no extra download,
 * no CSP change). USearch is the maintained alternative if a WASM build is wanted later.
 *
 * Complexity: build O(n log n * M * d) expected, query O(ef * M * d * log n) expected,
 * memory O(n * (d + M)).
 */
import { dot, vectorSearch } from "./core.js";

/** Collections above this size switch from brute force to HNSW. */
export const ANN_THRESHOLD = 20000;

function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}

/** Binary-heap-free sorted candidate list; ef is small so insertion sort is cheap. */
function insert(list, item, cap) {
  let i = list.length;
  list.push(item);
  while (i > 0 && list[i - 1][1] < item[1]) { list[i] = list[i - 1]; i--; }
  list[i] = item;
  if (list.length > cap) list.pop();
}

export class HNSW {
  constructor({ M = 16, efConstruction = 100, efSearch = 64, seed = 42 } = {}) {
    Object.assign(this, { M, efConstruction, efSearch });
    this.rand = rng(seed);
    this.mL = 1 / Math.log(M);
    this.vecs = [];
    this.links = []; // links[node][level] = neighbor ids
    this.entry = -1;
    this.top = -1;
  }

  get size() { return this.vecs.length; }

  searchLayer(q, entries, ef, level) {
    const seen = new Set(entries);
    const cand = entries.map((e) => [e, dot(q, this.vecs[e])]).sort((a, b) => b[1] - a[1]);
    const best = cand.slice(0, ef);
    while (cand.length) {
      const [c, cs] = cand.shift();
      if (best.length >= ef && cs < best[best.length - 1][1]) break;
      for (const nb of this.links[c][level] || []) {
        if (seen.has(nb)) continue;
        seen.add(nb);
        const s = dot(q, this.vecs[nb]);
        if (best.length < ef || s > best[best.length - 1][1]) {
          insert(best, [nb, s], ef);
          insert(cand, [nb, s], Infinity);
        }
      }
    }
    return best;
  }

  add(vec) {
    const id = this.vecs.length;
    const level = Math.floor(-Math.log(1 - this.rand()) * this.mL);
    this.vecs.push(vec);
    this.links.push(Array.from({ length: level + 1 }, () => []));
    if (this.entry < 0) { this.entry = id; this.top = level; return id; }
    let ep = [this.entry];
    for (let l = this.top; l > level; l--) ep = [this.searchLayer(vec, ep, 1, l)[0][0]];
    for (let l = Math.min(level, this.top); l >= 0; l--) {
      const found = this.searchLayer(vec, ep, this.efConstruction, l);
      const cap = l === 0 ? this.M * 2 : this.M;
      const nbrs = found.slice(0, this.M).map(([n]) => n);
      this.links[id][l] = nbrs;
      for (const n of nbrs) {
        const ln = this.links[n][l];
        ln.push(id);
        if (ln.length > cap) {
          const v = this.vecs[n];
          ln.sort((a, b) => dot(v, this.vecs[b]) - dot(v, this.vecs[a]));
          ln.length = cap;
        }
      }
      ep = found.map(([n]) => n);
    }
    if (level > this.top) { this.top = level; this.entry = id; }
    return id;
  }

  /** Return [[id, score], ...] best first. */
  search(q, k = 10, ef = this.efSearch) {
    if (this.entry < 0) return [];
    let ep = [this.entry];
    for (let l = this.top; l > 0; l--) ep = [this.searchLayer(q, ep, 1, l)[0][0]];
    return this.searchLayer(q, ep, Math.max(ef, k), 0).slice(0, k);
  }
}

const cache = new WeakMap();

/** Get (or incrementally extend) the HNSW index for a chunk array. */
export function indexFor(chunks, opts) {
  let ix = cache.get(chunks);
  if (!ix) { ix = new HNSW(opts); cache.set(chunks, ix); }
  for (let i = ix.size; i < chunks.length; i++) ix.add(chunks[i].vec);
  return ix;
}

// Register with core.vectorSearch so large libraries switch automatically.
vectorSearch.ann = { ANN_THRESHOLD, indexFor };
