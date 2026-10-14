// Pure retrieval, clustering and answering logic for the browser demo.
// No DOM access here, so the same file is unit tested with `node --test`.

export const REFUSAL_TEXT = "I could not find this in the loaded texts.";

const STOPWORDS = new Set(`a an and are as at be by for from has have he her his i in is it its of on or
that the their them they this to was were what when where which who whom why will with you your our we us my me does did
do how not no shall unto thee thou o e el la los las de del que en un una por para con se do da dos das
um uma os ao com qual quem como`.split(/\s+/));

/** Lowercase, strip accents and split into word tokens. */
export function tokenize(text, dropStopwords = true) {
  const words = text.toLowerCase().normalize("NFKD").replace(/[\u0300-\u036f]/g, "").match(/[\p{L}\p{N}_]+/gu) || [];
  return dropStopwords ? words.filter((w) => !STOPWORDS.has(w)) : words;
}

/** Split TXT into pages on form feed, else virtual pages of 3000 chars. */
export function splitTxtPages(text) {
  if (text.includes("\f")) return text.split("\f");
  const pages = [];
  for (let i = 0; i < Math.max(text.length, 1); i += 3000) pages.push(text.slice(i, i + 3000));
  return pages;
}

/** Remove control characters except tab, newline, carriage return and form feed. */
export function stripControl(text) {
  return text.replace(/[\u0000-\u0008\u000b\u000e-\u001f\u007f]/g, "");
}

/** Citation-safe id from a file name. */
export function safeBookId(name) {
  const stem = name.replace(/\.[^.]+$/, "").toLowerCase().replace(/[^a-z0-9._-]+/g, "_").replace(/^[._-]+|[._-]+$/g, "");
  return stem || "book";
}

/**
 * Page-aware sliding window chunker. O(N) in characters.
 * Returns chunks with pageStart/pageEnd (1-based PDF page index).
 */
export function chunkPages(bookId, pages, target = 400, overlap = 60) {
  const tokens = [];
  pages.forEach((page, p) => {
    for (const m of page.matchAll(/\S+/g)) tokens.push([m[0], p + 1]);
  });
  const chunks = [];
  const step = target - overlap;
  for (let start = 0, idx = 0; start < tokens.length; start += step, idx++) {
    const win = tokens.slice(start, start + target);
    chunks.push({
      chunkId: `${bookId}-${String(idx).padStart(4, "0")}`,
      bookId,
      text: win.map((t) => t[0]).join(" "),
      pageStart: win[0][1],
      pageEnd: win[win.length - 1][1],
    });
    if (start + target >= tokens.length) break;
  }
  return chunks;
}

/** BM25 Okapi with an inverted index (term -> postings). */
export class BM25 {
  constructor(k1 = 1.5, b = 0.75) {
    this.k1 = k1; this.b = b;
    this.ids = []; this.lengths = []; this.postings = new Map(); this.totalLen = 0;
  }
  /** Add one document. O(document tokens). */
  add(id, text) {
    const doc = this.ids.length;
    const toks = tokenize(text);
    const tf = new Map();
    for (const t of toks) tf.set(t, (tf.get(t) || 0) + 1);
    for (const [t, f] of tf) {
      if (!this.postings.has(t)) this.postings.set(t, []);
      this.postings.get(t).push([doc, f]);
    }
    this.ids.push(id); this.lengths.push(toks.length); this.totalLen += toks.length;
  }
  /** Score only documents in the postings of query terms. */
  search(query, k = 10) {
    const n = this.ids.length;
    if (!n) return [];
    const avg = this.totalLen / n;
    const scores = new Map();
    for (const t of new Set(tokenize(query))) {
      const plist = this.postings.get(t);
      if (!plist) continue;
      const idf = Math.log(1 + (n - plist.length + 0.5) / (plist.length + 0.5));
      for (const [doc, f] of plist) {
        const s = idf * (f * (this.k1 + 1)) / (f + this.k1 * (1 - this.b + (this.b * this.lengths[doc]) / avg));
        scores.set(doc, (scores.get(doc) || 0) + s);
      }
    }
    return topK([...scores].map(([doc, s]) => [this.ids[doc], s]), k);
  }
}

/** Min-heap based top-k over [id, score] pairs. O(n log k). */
export function topK(pairs, k) {
  const heap = [];
  const up = (i) => { while (i > 0) { const p = (i - 1) >> 1; if (heap[p][1] <= heap[i][1]) break; [heap[p], heap[i]] = [heap[i], heap[p]]; i = p; } };
  const down = (i) => {
    for (;;) {
      const l = 2 * i + 1, r = l + 1; let m = i;
      if (l < heap.length && heap[l][1] < heap[m][1]) m = l;
      if (r < heap.length && heap[r][1] < heap[m][1]) m = r;
      if (m === i) return;
      [heap[m], heap[i]] = [heap[i], heap[m]]; i = m;
    }
  };
  for (const pair of pairs) {
    if (heap.length < k) { heap.push(pair); up(heap.length - 1); }
    else if (k > 0 && pair[1] > heap[0][1]) { heap[0] = pair; down(0); }
  }
  return heap.sort((a, b) => b[1] - a[1]);
}

/** Offline fallback embedder: hashing trick into `dim` buckets, L2 normalized. */
export function hashEmbed(text, dim = 256) {
  const v = new Float32Array(dim);
  for (const tok of tokenize(text)) {
    let h = 2166136261;
    for (let i = 0; i < tok.length; i++) { h ^= tok.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; }
    v[h % dim] += (h & 0x80000000) ? -1 : 1;
  }
  return normalize(v);
}

export function normalize(v) {
  let s = 0;
  for (const x of v) s += x * x;
  const n = Math.sqrt(s) || 1;
  return Float32Array.from(v, (x) => x / n);
}

export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** Brute-force cosine search. O(n d + n log k). */
export function vectorSearch(chunks, qv, k = 10) {
  return topK(chunks.map((c) => [c.chunkId, dot(c.vec, qv)]), k);
}

/** Reciprocal Rank Fusion: sum of 1 / (k + rank). */
export function rrf(rankings, k = 60) {
  const scores = new Map();
  for (const ranking of rankings) ranking.forEach((id, i) => scores.set(id, (scores.get(id) || 0) + 1 / (k + i + 1)));
  return [...scores].sort((a, b) => b[1] - a[1]);
}

/** Share of distinct query content words present in the passage, 0 to 1. */
export function coverage(query, text) {
  const q = new Set(tokenize(query));
  if (!q.size) return 0;
  const p = new Set(tokenize(text));
  let hit = 0;
  for (const t of q) if (p.has(t)) hit++;
  return hit / q.size;
}

/**
 * Relevance in [0, 1]. Lexical coverage blended with cosine; when `cosRange`
 * [lo, hi] is given (semantic model), rescaled cosine can also carry the score,
 * which lets questions in another language pass the gate.
 */
export function relevance(question, c, qv, cosRange = null) {
  const cov = coverage(question, c.text), cos = Math.max(0, dot(c.vec, qv));
  if (!cosRange) return 0.8 * cov + 0.2 * cos;
  const [lo, hi] = cosRange;
  return Math.max(cov, Math.min(1, Math.max(0, (cos - lo) / (hi - lo))));
}

export function citation(c) {
  return `[${c.bookId} p.${c.pageStart} #${c.chunkId}]`;
}

/**
 * Hybrid retrieve, rerank by coverage blended with cosine, gate, and quote.
 * Returns { refused, text, citations, topScore }.
 */
export function answer(question, chunks, bm25, qv, { threshold = 0.5, topN = 3, depth = 50, cosRange = null } = {}) {
  const byId = new Map(chunks.map((c) => [c.chunkId, c]));
  if (!chunks.length || !question.trim()) return { refused: true, text: REFUSAL_TEXT, citations: [], topScore: 0 };
  const lex = bm25.search(question, depth).map((p) => p[0]);
  const vec = vectorSearch(chunks, qv, depth).map((p) => p[0]);
  const fused = rrf([lex, vec]).slice(0, depth).map(([id]) => byId.get(id));
  const ranked = fused
    .map((c) => [c, relevance(question, c, qv, cosRange)])
    .sort((a, b) => b[1] - a[1])
    .slice(0, topN);
  const topScore = ranked.length ? ranked[0][1] : 0;
  if (topScore < threshold) return { refused: true, text: REFUSAL_TEXT, citations: [], topScore };
  const cites = ranked.filter(([, s]) => s >= threshold * 0.6).map(([c]) => c);
  const text = cites.map((c) => `${c.text} ${citation(c)}`).join("\n");
  return { refused: false, text, citations: cites, topScore };
}

/** Seeded PRNG (mulberry32) so clustering is reproducible. */
export function rng(seed = 42) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** k-means with k-means++ init (Lloyd iterations). O(n K i d). */
export function kmeans(vecs, K, { iters = 50, seed = 42 } = {}) {
  const n = vecs.length, d = vecs[0].length, rand = rng(seed);
  const centers = [Float32Array.from(vecs[Math.floor(rand() * n)])];
  const dist = new Float64Array(n).fill(Infinity);
  while (centers.length < K) {
    let total = 0;
    for (let i = 0; i < n; i++) {
      const c = centers[centers.length - 1];
      let s = 0;
      for (let j = 0; j < d; j++) { const x = vecs[i][j] - c[j]; s += x * x; }
      dist[i] = Math.min(dist[i], s); total += dist[i];
    }
    let r = rand() * total, pick = 0;
    for (; pick < n - 1 && r > dist[pick]; pick++) r -= dist[pick];
    centers.push(Float32Array.from(vecs[pick]));
  }
  const labels = new Int32Array(n);
  for (let it = 0; it < iters; it++) {
    let changed = 0;
    for (let i = 0; i < n; i++) {
      let best = 0, bestD = Infinity;
      for (let c = 0; c < K; c++) {
        let s = 0;
        for (let j = 0; j < d; j++) { const x = vecs[i][j] - centers[c][j]; s += x * x; }
        if (s < bestD) { bestD = s; best = c; }
      }
      if (labels[i] !== best || it === 0) { changed++; labels[i] = best; }
    }
    const sums = Array.from({ length: K }, () => new Float64Array(d)), counts = new Int32Array(K);
    for (let i = 0; i < n; i++) { counts[labels[i]]++; for (let j = 0; j < d; j++) sums[labels[i]][j] += vecs[i][j]; }
    for (let c = 0; c < K; c++) if (counts[c]) for (let j = 0; j < d; j++) centers[c][j] = sums[c][j] / counts[c];
    if (!changed) break;
  }
  return { labels: Array.from(labels), centers };
}

/** Mean silhouette with Euclidean distance; uses up to `sample` points. O(s n d). */
export function silhouette(vecs, labels, sample = 300) {
  const n = vecs.length, K = Math.max(...labels) + 1;
  if (K < 2 || n < 3) return NaN;
  const step = Math.max(1, Math.floor(n / sample));
  let total = 0, count = 0;
  for (let i = 0; i < n; i += step) {
    const sum = new Float64Array(K), cnt = new Int32Array(K);
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      sum[labels[j]] += Math.sqrt(vecs[i].reduce((s, x, t) => s + (x - vecs[j][t]) ** 2, 0)); cnt[labels[j]]++;
    }
    const own = labels[i];
    if (!cnt[own]) continue;
    const a = sum[own] / cnt[own];
    let b = Infinity;
    for (let c = 0; c < K; c++) if (c !== own && cnt[c]) b = Math.min(b, sum[c] / cnt[c]);
    total += (b - a) / Math.max(a, b); count++;
  }
  return count ? total / count : NaN;
}

/** Try K in [2, maxK] and keep the best silhouette. */
export function autoCluster(vecs, maxK = 8, seed = 42) {
  if (vecs.length < 3) return { labels: vecs.map(() => 0), k: 1, silhouette: NaN };
  let best = null;
  for (let K = 2; K <= Math.min(maxK, vecs.length - 1); K++) {
    const { labels } = kmeans(vecs, K, { seed });
    const s = silhouette(vecs, labels);
    if (!best || s > best.silhouette) best = { labels, k: K, silhouette: s };
  }
  return best;
}

/** 2D PCA by power iteration with deflation. O(n d) per iteration. */
export function pca2d(vecs, iters = 60, seed = 7) {
  const n = vecs.length, d = vecs[0]?.length || 0, rand = rng(seed);
  const mean = new Float64Array(d);
  for (const v of vecs) for (let j = 0; j < d; j++) mean[j] += v[j] / n;
  const X = vecs.map((v) => Float64Array.from(v, (x, j) => x - mean[j]));
  const comps = [];
  for (let c = 0; c < 2; c++) {
    let w = Float64Array.from({ length: d }, () => rand() - 0.5);
    for (let it = 0; it < iters; it++) {
      const next = new Float64Array(d);
      for (const x of X) { const p = dot(x, w); for (let j = 0; j < d; j++) next[j] += p * x[j]; }
      for (const prev of comps) { const p = dot(next, prev); for (let j = 0; j < d; j++) next[j] -= p * prev[j]; }
      const norm = Math.sqrt(dot(next, next)) || 1;
      w = next.map((x) => x / norm);
    }
    comps.push(w);
  }
  return X.map((x) => [dot(x, comps[0]), dot(x, comps[1])]);
}

/** Class-based TF-IDF: top words per cluster. */
export function clusterKeywords(texts, labels, topN = 4) {
  const K = Math.max(...labels) + 1, tf = Array.from({ length: K }, () => new Map()), df = new Map();
  texts.forEach((t, i) => { for (const w of tokenize(t)) if (w.length > 2 && !/^\d+$/.test(w)) tf[labels[i]].set(w, (tf[labels[i]].get(w) || 0) + 1); });
  for (const m of tf) for (const w of m.keys()) df.set(w, (df.get(w) || 0) + 1);
  return tf.map((m) => [...m].map(([w, f]) => [w, f * Math.log(1 + K / df.get(w))]).sort((a, b) => b[1] - a[1]).slice(0, topN).map((p) => p[0]));
}

/** Top-k neighbors of a chunk restricted to other books. */
export function similarAcrossBooks(chunk, chunks, k = 5) {
  return topK(chunks.filter((c) => c.bookId !== chunk.bookId).map((c) => [c, dot(c.vec, chunk.vec)]), k);
}
