// Browser port of the Python answer pipeline (src/scripture_rag): verse-aware chunking,
// archaic normalization, query cleaning, language detection, BM25 Okapi (rank_bm25 variant),
// weighted rank fusion with BM25 on the original query, max-window cross-encoder reranking,
// per-language thresholds and the lexical support gate. Pure functions; models are injected,
// so `node --test` checks the same decisions as Python (site/tests/parity.test.mjs).

export const REFUSAL_TEXT = "I could not find this in the loaded texts.";

/** Shipped defaults, tuned on the tune split plus the synthetic dev set (docs/eval/SCORECARD.md). */
export const DEFAULTS = Object.freeze({
  thresholds: Object.freeze({ en: 0.18, pt: 0.04, es: 0.34 }),
  defaultThreshold: 0.5,
  minSupport: 0.3,
  candidates: 50,
  denseWeight: 0.5,
  rrfK: 20,
  bm25K1: 1.2,
  bm25B: 0.75,
  topRanked: 5,
  topContext: 3,
  rerankWindow: 60,
  rerankStride: 30,
  windowTokens: 200,
  windowOverlap: 40,
  verseWords: 80,
});

const STOPWORDS = new Set(`a an and are as at be by for from has have he her his i in is it its of on or
that the their them they this to was were what when where which who whom why will
with you your our we us my me does did do how not no shall unto thee thou o e el la los las de del
que en un una por para con se do da dos das um uma os ao com qual quem como`.split(/\s+/));

/** Python text_utils.tokenize: lowercase, NFKD, drop combining marks, \w+ words. */
export function tokenize(text, dropStopwords = true) {
  const words = text.toLowerCase().normalize("NFKD").replace(/\p{Mn}/gu, "").match(/[\p{L}\p{N}_]+/gu) || [];
  return dropStopwords ? words.filter((w) => !STOPWORDS.has(w)) : words;
}

/** Python str.split(): whitespace tokens. */
export const words = (t) => t.split(/\s+/).filter(Boolean);

// ---------------------------------------------------------------- query cleaning and language

const ROLE_RE = /^\s*(?:system|assistant|user|developer|admin)\s*[:>\]-]+\s*/i;
const PREAMBLE_RE = new RegExp(
  String.raw`^\s*(?:(?:developer|debug|admin|god|jailbreak|dan)\s+mode\s+(?:is\s+)?(?:enabled|on|activated)` +
  String.raw`|ignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above)\s+(?:instructions|rules|prompts?)` +
  String.raw`|you\s+are\s+now\s+[^.?!]{0,60}` +
  String.raw`|new\s+instructions?)\s*[.!:;,]+\s*`, "i");

/** sanitize.clean_query: strip role tags and instruction preambles from the front. */
export function cleanQuery(question) {
  let q = question;
  for (let i = 0; i < 4; i++) {
    const next = q.replace(ROLE_RE, "").replace(PREAMBLE_RE, "");
    if (next === q) break;
    q = next;
  }
  return q.trim() || question.trim();
}

const HINTS = {
  pt: new Set(("o a os as um uma que não em do da dos das no na nos nas com para por como quem qual " +
    "quando onde porque é são foi está seu sua ao à também mais muito você").split(" ")),
  es: new Set(("el la los las un una que no en del al con para por como quién quien cuál cuándo dónde " +
    "porque es son fue está su sus también más muy usted qué").split(" ")),
  en: new Set(("the a an of to in is are was what who which how why when where does did do and with " +
    "for from by that this his her their").split(" ")),
};
const MARKERS = { "ç": "pt", "ã": "pt", "õ": "pt", "ê": "pt", "ñ": "es", "¿": "es", "¡": "es" };

/** translate.detect_language: function words and diacritics; ties prefer English. */
export function detectLanguage(text) {
  const low = text.toLowerCase();
  const s = { pt: 0, es: 0, en: 0 };
  for (const [ch, lg] of Object.entries(MARKERS)) if (low.includes(ch)) s[lg] += 2;
  for (const w of low.match(/[a-záéíóúâêôãõçñü]+/g) || []) for (const lg of ["pt", "es", "en"]) if (HINTS[lg].has(w)) s[lg] += 1;
  let best = null;
  for (const lg of ["pt", "es", "en"]) {
    if (best === null || s[lg] > s[best] || (s[lg] === s[best] && lg === "en")) best = lg;
  }
  return best;
}

// ---------------------------------------------------------------- archaic normalization

const QUERY_SYNONYMS = {
  donkey: "ass", donkeys: "asses", quran: "koran", qur: "koran", saturday: "sabbath",
  you: "thee", your: "thy", clothes: "raiment", clothing: "raiment", boat: "ark",
};

/** normalize.expand_query: append archaic equivalents of modern words. */
export function expandQuery(query) {
  const extra = (query.toLowerCase().match(/[a-z]+/g) || []).filter((w) => w in QUERY_SYNONYMS).map((w) => QUERY_SYNONYMS[w]);
  return extra.length ? `${query} ${extra.join(" ")}` : query;
}

const sub = (re, repl) => (w) => { const out = w.replace(re, repl); return out !== w ? out : null; };
const RULES = [
  sub(/e$/, ""), sub(/([bcdfgklmnprst])\1$/, "$1"), sub(/([bcdfgklmnprst])\1/, "$1"), sub(/ie$/, "y"),
  sub(/ee/, "ea"), sub(/ee/, "e"), sub(/^v/, "u"), sub(/(?<=[aeiou])u(?=[aeiou])/, "v"), sub(/^i(?=[aeiou])/, "j"),
  sub(/^l(?=[aeiou])/, "j"), sub(/^l(?=[aeiou])/, "i"), sub(/es$/, "s"), sub(/oo/, "o"), sub(/neth$/, "nth"),
  sub(/eth$/, "es"), sub(/eth$/, "s"), sub(/y$/, "ie"), sub(/ck$/, "c"), sub(/our$/, "or"), sub(/ie(?=[a-z])/, "y"),
  sub(/(?<=[a-z])y(?=[a-z])/, "i"),
];

/** normalize.ArchaicNormalizer: breadth-first spelling rules (depth <= 3) into a modern vocabulary. */
export class ArchaicNormalizer {
  constructor(vocab, maxDepth = 3) { this.vocab = new Set(vocab); this.depth = maxDepth; this.cache = new Map(); }
  word(w) {
    if (w.length < 3) return w;
    if (this.vocab.has(w)) {
      if (w.endsWith("eth") && w.length >= 5) {
        for (const c of [w.slice(0, -3) + "es", w.slice(0, -3) + "s"]) if (this.vocab.has(c)) return c;
      }
      return w;
    }
    if (this.cache.has(w)) return this.cache.get(w);
    let found = w;
    const queue = [[w, 0]], seen = new Set([w]);
    outer: for (let qi = 0; qi < queue.length; qi++) {
      const [cur, d] = queue[qi];
      if (d === this.depth) continue;
      for (const rule of RULES) {
        const nxt = rule(cur);
        if (nxt === null || seen.has(nxt) || nxt.length < 2) continue;
        if (this.vocab.has(nxt)) { found = nxt; break outer; }
        seen.add(nxt); queue.push([nxt, d + 1]);
      }
    }
    this.cache.set(w, found);
    return found;
  }
  text(t) { return t.replace(/[A-Za-z]+/g, (m) => this.word(m.toLowerCase())); }
}

// ---------------------------------------------------------------- chunking

const BIBLE_BOOKS = ["Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua", "Judges", "Ruth", "1 Samuel",
  "2 Samuel", "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah", "Esther", "Job",
  "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon", "Isaiah", "Jeremiah", "Lamentations", "Ezekiel",
  "Daniel", "Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai",
  "Zechariah", "Malachi", "Matthew", "Mark", "Luke", "John", "Acts", "Romans", "1 Corinthians",
  "2 Corinthians", "Galatians", "Ephesians", "Philippians", "Colossians", "1 Thessalonians",
  "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews", "James", "1 Peter", "2 Peter",
  "1 John", "2 John", "3 John", "Jude", "Revelation"];
/** Python sorted() order for strings (code point order). */
const cmp = (a, b) => { const x = [...a], y = [...b]; for (let i = 0; i < Math.min(x.length, y.length); i++) { const d = x[i].codePointAt(0) - y[i].codePointAt(0); if (d) return d; } return x.length - y.length; };
const verseRef = (v) => `${v.book} ${v.chapter}:${v.verse}`;
const pyUpper = (ch) => ch !== ch.toLowerCase() && ch === ch.toUpperCase();
const pyAlpha = (ch) => /\p{L}/u.test(ch);

/** chunker.Chunker: whitespace-token windows with overlap. */
export function windowChunks(bookId, pages, target = DEFAULTS.windowTokens, overlap = DEFAULTS.windowOverlap, idPrefix = "") {
  const toks = [];
  pages.forEach((p, i) => { for (const m of p.matchAll(/\S+/g)) toks.push([m[0], i + 1]); });
  const out = [], step = target - overlap;
  for (let idx = 0, start = 0; start < toks.length; idx++, start += step) {
    const win = toks.slice(start, start + target);
    out.push({ chunkId: `${bookId}-${idPrefix}${String(idx).padStart(4, "0")}`, bookId, text: win.map((t) => t[0]).join(" "),
      pageStart: win[0][1], pageEnd: win[win.length - 1][1], ref: "" });
    if (start + target >= toks.length) break;
  }
  return out;
}

/** verses.parse_numbered: editions that print chapter:verse before each verse. */
export function parseNumbered(text, books = BIBLE_BOOKS, page = 1) {
  const marks = [...text.matchAll(/(?<![\d:])(\d{1,3}):(\d{1,3})\s/g)];
  const out = [];
  let b = -1;
  marks.forEach((m, i) => {
    const ch = Number(m[1]), vs = Number(m[2]);
    if (ch === 1 && vs === 1) b++;
    if (b < 0 || b >= books.length) return;
    const end = i + 1 < marks.length ? marks[i + 1].index : text.length;
    out.push({ book: books[b], chapter: ch, verse: vs, text: words(text.slice(m.index + m[0].length, end)).join(" "), page });
  });
  return out;
}

const ROMAN = { I: 1, V: 5, X: 10, L: 50, C: 100 };
const OCR_ROMAN = { n: "I", l: "I", i: "I", v: "V", x: "X", c: "C", e: "C" };
export function roman(s) {
  s = [...s.replaceAll("U", "II")].map((c) => OCR_ROMAN[c] || c).join("");
  if (!s || [...s].some((c) => !(c in ROMAN))) return 0;
  let total = 0;
  for (let i = 0; i < s.length; i++) {
    const v = ROMAN[s[i]], nx = s[i + 1];
    total += nx && ROMAN[nx] > v ? -v : v;
  }
  return total;
}

/** verses.parse_suras: Quran translations with SURA headings, one verse per line. */
export function parseSuras(pages) {
  const out = [];
  let sura = 0, n = 0, cur = [], curPage = 1;
  const flush = () => { if (sura && cur.length && n) out.push({ book: "Sura", chapter: sura, verse: n, text: words(cur.join(" ")).join(" "), page: curPage }); };
  const SURA = /^\s*S\s?U\s?R\s?A\S?\s+(?:\d\s+)?([IVXLCivxlcnUe]+)(?=[\W\d]|$)/;
  pages.forEach((page, pi) => {
    const pageNo = pi + 1, lines = page.split("\n");
    for (let k = 0; k < lines.length; k++) {
      const s = lines[k].trim();
      if (!s || (k === 0 && s.length < 40)) continue;
      const m = SURA.exec(s);
      if (m) { flush(); sura = roman(m[1]); n = 0; cur = []; sura = sura > 0 && sura <= 114 ? sura : 0; continue; }
      if ("*^\u2020\u2021".includes(s[0]) || (/^\d\s?[A-Z][a-z]+\./.test(s) && n > 3)) break;
      if (!sura || /(\d{1,3})\s+Verses/.test(s) || s.startsWith("In the Name of God") || s.startsWith("[")) continue;
      const pieces = [];
      let last = 0;
      for (const m2 of (" " + s).matchAll(/\s(\d{1,3})\s+(?=[A-Z"'])/g)) {
        const v = Number(m2[1]);
        if (v % 5 === 0 && Math.abs(v - n) <= 15 && v > 0) {
          pieces.push([s.slice(last, Math.max(0, m2.index - 1)), null]);
          last = m2.index + m2[0].length - 1;
          pieces.push(["", v]);
        }
      }
      pieces.push([s.slice(last), null]);
      for (let [text, mark] of pieces) {
        text = text.trim();
        if (mark !== null) { flush(); n = mark - 1; cur = []; curPage = pageNo; continue; }
        if (!text) continue;
        const num = /^(\d{1,3})\s+(?=\S)/.exec(text);
        if (num && Math.abs(Number(num[1]) - n) <= 15 && Number(num[1]) % 5 === 0) {
          flush(); n = Number(num[1]); cur = [text.slice(num[0].length)]; curPage = pageNo;
        } else if (pyUpper(text[0]) || (n === 0 && pyAlpha(text[0]))) {
          if (cur.length || n) flush();
          n += 1; cur = [text]; curPage = pageNo;
        } else if (cur.length) cur.push(text);
      }
    }
  });
  flush();
  return out;
}

/** verses.ReferenceAligner: tag sentences of an unnumbered edition with verses of a numbered one. */
export class ReferenceAligner {
  constructor(reference, normalizer = null) {
    this.ref = reference; this.norm = normalizer;
    this.post = new Map(); this.terms = reference.map((v) => new Set(tokenize(v.text)));
    const df = new Map();
    this.terms.forEach((ts, i) => { for (const t of ts) { df.set(t, (df.get(t) || 0) + 1); if (!this.post.has(t)) this.post.set(t, []); this.post.get(t).push(i); } });
    const n = reference.length;
    this.idf = new Map([...df].map(([t, c]) => [t, Math.log(n / c)]));
    this.maxDf = Math.max(50, Math.floor(n / 50));
  }
  match(sentence, prev) {
    const text = this.norm ? this.norm.text(sentence) : sentence;
    const known = [...new Set(tokenize(text))].filter((t) => this.idf.has(t)).sort(cmp);
    const denom = known.reduce((a, t) => a + this.idf.get(t), 0) || 1;
    const acc = new Map();
    for (const t of known) {
      const pl = this.post.get(t);
      if (pl.length > this.maxDf) continue;
      for (const i of pl) acc.set(i, (acc.get(i) || 0) + this.idf.get(t));
    }
    for (const t of known) {
      if (this.post.get(t).length > this.maxDf) {
        for (const i of acc.keys()) if (i - prev >= 0 && i - prev <= 40 && this.terms[i].has(t)) acc.set(i, acc.get(i) + this.idf.get(t));
      }
    }
    let best = -1, bestS = 0;
    for (const [i, s] of acc) {
      const d = i - prev;
      const score = s / denom + (d >= 0 && d <= 3 ? 0.25 : d >= 0 && d <= 40 ? 0.1 : 0);
      if (score > bestS) { best = i; bestS = score; }
    }
    return [best, Math.min(1, bestS)];
  }
  contained(i, terms) {
    const vt = [...this.terms[i]].filter((t) => this.idf.has(t)).sort(cmp);
    const total = vt.reduce((a, t) => a + this.idf.get(t), 0);
    return total > 0 && vt.filter((t) => terms.has(t)).reduce((a, t) => a + this.idf.get(t), 0) / total >= 0.7;
  }
  align(pages, minScore = 0.45) {
    const out = [];
    let prev = -10;
    const same = (a, b) => verseRef(a) === verseRef(b);
    pages.forEach((page, pi) => {
      const pageNo = pi + 1, flat = words(page).join(" ");
      for (const sent of flat.split(/(?<=[.;:?!,])\s+(?=[A-Z])/)) {
        if (words(sent).length < 3) {
          if (out.length && out[out.length - 1].page === pageNo) { const l = out[out.length - 1]; out[out.length - 1] = { ...l, text: `${l.text} ${sent}` }; }
          continue;
        }
        const [i, s] = this.match(sent, prev);
        if (i < 0 || s < minScore) continue;
        const r = this.ref[i];
        prev = i;
        if (out.length && same(out[out.length - 1], r)) { const l = out[out.length - 1]; out[out.length - 1] = { book: r.book, chapter: r.chapter, verse: r.verse, text: `${l.text} ${sent}`, page: l.page }; }
        else out.push({ book: r.book, chapter: r.chapter, verse: r.verse, text: sent, page: pageNo });
        const terms = new Set(tokenize(this.norm ? this.norm.text(sent) : sent));
        while (prev + 1 < this.ref.length && this.contained(prev + 1, terms)) {
          prev++;
          const nx = this.ref[prev];
          if (nx.book !== r.book || nx.chapter !== r.chapter) { prev--; break; }
          out.push({ book: nx.book, chapter: nx.chapter, verse: nx.verse, text: "", page: pageNo });
        }
      }
    });
    return out;
  }
}

/** verses.verse_chunks: groups of about `target` words inside one chapter, one verse overlap. */
export function verseChunks(bookId, verses, target = DEFAULTS.verseWords, overlapVerses = 1) {
  const groups = [], chunks = [];
  for (const v of verses) {
    const g = groups[groups.length - 1];
    if (g && g[g.length - 1].book === v.book && g[g.length - 1].chapter === v.chapter) g.push(v); else groups.push([v]);
  }
  for (const g of groups) {
    let i = 0;
    while (i < g.length) {
      let j = i, wc = 0;
      while (j < g.length && (wc < target || j === i)) { wc += words(g[j].text).length; j++; }
      const part = g.slice(i, j), a = part[0], b = part[part.length - 1];
      chunks.push({ chunkId: `${bookId}-${String(chunks.length).padStart(4, "0")}`, bookId,
        text: part.filter((v) => v.text).map((v) => v.text).join(" "), pageStart: a.page, pageEnd: b.page,
        ref: a.verse === b.verse ? verseRef(a) : `${verseRef(a)}-${b.verse}` });
      if (j >= g.length) break;
      i = Math.max(i + 1, j - overlapVerses);
    }
  }
  return chunks;
}

/**
 * pipeline.RagPipeline.split: numbered verses, else suras (plus note windows), else alignment to the
 * bundled reference (Genesis to Joshua), else plain windows.
 */
export function chunkDocument(bookId, pages, { aligner = null, windowTokens = DEFAULTS.windowTokens, overlap = DEFAULTS.windowOverlap } = {}) {
  const text = pages.join("\n");
  let verses = [], suras = false;
  if ((text.match(/(?<![\d:])\d{1,3}:\d{1,3}\s/g) || []).length > 50) verses = parseNumbered(text);
  else if ((text.match(/^\s*SURA\s/gm) || []).length >= 3) { verses = parseSuras(pages); suras = true; }
  else if (aligner) {
    verses = aligner.align(pages);
    if (verses.length < Math.max(50, 0.2 * Math.max(1, Math.floor(words(text).length / 30)))) verses = []; // not this work
  }
  if (!verses.length) return windowChunks(bookId, pages, windowTokens, overlap);
  const chunks = verseChunks(bookId, verses);
  if (suras) chunks.push(...windowChunks(bookId, pages, windowTokens, overlap, "n"));
  return chunks;
}

// ---------------------------------------------------------------- lexical scoring

/** rank_bm25.BM25Okapi (k1 1.5, b 0.75, epsilon 0.25) over tokenize(); zero-score hits dropped. */
export class BM25Okapi {
  constructor({ k1 = 1.5, b = 0.75, epsilon = 0.25 } = {}) { Object.assign(this, { k1, b, epsilon }); this.ids = []; this.docs = []; this.dirty = true; }
  add(ids, texts) {
    texts.forEach((t, i) => {
      const toks = tokenize(t);
      const tf = new Map();
      for (const w of (toks.length ? toks : ["_empty_"])) tf.set(w, (tf.get(w) || 0) + 1);
      this.ids.push(ids[i]); this.docs.push({ tf, len: toks.length || 1 });
    });
    this.dirty = true;
  }
  build() {
    const nd = new Map(), post = new Map();
    let total = 0;
    this.docs.forEach((d, i) => {
      total += d.len;
      for (const [w, f] of d.tf) { nd.set(w, (nd.get(w) || 0) + 1); if (!post.has(w)) post.set(w, []); post.get(w).push([i, f]); }
    });
    const n = this.docs.length;
    this.avgdl = total / n;
    this.idf = new Map();
    let sum = 0;
    const neg = [];
    for (const [w, f] of nd) { const v = Math.log(n - f + 0.5) - Math.log(f + 0.5); this.idf.set(w, v); sum += v; if (v < 0) neg.push(w); }
    const eps = this.epsilon * (sum / this.idf.size);
    for (const w of neg) this.idf.set(w, eps);
    this.post = post; this.dirty = false;
  }
  /** Scores in the same per-term order as numpy so sums match bit for bit. */
  search(query, k) {
    const terms = tokenize(query);
    if (!this.docs.length || !terms.length) return [];
    if (this.dirty) this.build();
    const score = new Float64Array(this.docs.length);
    const { k1, b, avgdl } = this;
    for (const q of terms) {
      const idf = this.idf.get(q) || 0, pl = this.post.get(q);
      // numpy adds idf * 0 for absent docs; adding 0 does not change a float sum.
      if (!pl) continue;
      for (const [i, f] of pl) score[i] += idf * ((f * (k1 + 1)) / (f + k1 * (1 - b + (b * this.docs[i].len) / avgdl)));
    }
    return topIndices(score, k).filter((i) => score[i] > 0).map((i) => [this.ids[i], score[i]]);
  }
}

/** Indices of the k largest values, ties broken by lower index (matches index.top_k). */
export function topIndices(scores, k) {
  const idx = Array.from(scores.keys());
  idx.sort((a, b) => scores[b] - scores[a] || a - b);
  return idx.slice(0, Math.min(k, idx.length));
}

const stems = (text) => new Set(tokenize(text).filter((t) => [...t].length > 2).map((t) => [...t].slice(0, 5).join("")));

/** gate.IdfTable plus gate.support: IDF-weighted share of question stems found in the passage. */
export class IdfTable {
  constructor(texts) {
    this.df = new Map(); this.n = 0;
    for (const t of texts) { this.n++; for (const s of stems(t)) this.df.set(s, (this.df.get(s) || 0) + 1); }
    this.n = Math.max(this.n, 1);
  }
  idf(s) { return Math.log((this.n + 1) / ((this.df.get(s) || 0) + 0.5)); }
  support(question, passage) {
    const q = stems(question);
    if (!q.size) return 0;
    const p = stems(passage);
    let total = 0, hit = 0;
    for (const s of q) { const w = this.idf(s); total += w; if (p.has(s)) hit += w; }
    return total ? hit / total : 0;
  }
}

/** rerank.windows: overlapping word windows (at least one). */
export function rerankWindows(text, size = DEFAULTS.rerankWindow, stride = DEFAULTS.rerankStride) {
  const w = words(text);
  if (w.length <= size) return [text];
  const out = [];
  for (let i = 0; i < w.length - size + stride; i += stride) out.push(w.slice(i, i + size).join(" "));
  return out;
}

/** Weighted reciprocal rank fusion; ties keep first-seen order. */
export function weightedRrf(lists, k = DEFAULTS.rrfK) {
  const s = new Map();
  for (const [ids, w] of lists) ids.forEach((id, r) => s.set(id, (s.get(id) || 0) + w / (k + r + 1)));
  return [...s].sort((a, b) => b[1] - a[1]).map(([id]) => id);
}

// ---------------------------------------------------------------- index and answer

/**
 * Searchable index over chunks. `normalizer` rewrites archaic spelling for BM25, the embedder input
 * and the support check; the display text and the reranker see the original.
 */
export class RagIndex {
  constructor(normalizer = null, { k1 = DEFAULTS.bm25K1, b = DEFAULTS.bm25B } = {}) { this.norm = normalizer; this.chunks = []; this.indexText = []; this.vecs = []; this.bm25 = new BM25Okapi({ k1, b }); this.idfTable = null; this.byId = new Map(); }
  indexTextFor(chunk) { return chunk.indexText ?? (this.norm ? this.norm.text(chunk.text) : chunk.text); }
  add(chunks, vecs) {
    const texts = chunks.map((c) => this.indexTextFor(c));
    chunks.forEach((c, i) => { this.byId.set(c.chunkId, this.chunks.length); this.chunks.push(c); this.indexText.push(texts[i]); this.vecs.push(vecs[i]); });
    this.bm25.add(chunks.map((c) => c.chunkId), texts);
    this.idfTable = null;
  }
  idf() { return (this.idfTable ??= new IdfTable(this.indexText)); }
  /** Dense ranking: float64 dot products, ties by index. */
  dense(qv, k) {
    const n = this.chunks.length, s = new Float64Array(n);
    for (let i = 0; i < n; i++) { const v = this.vecs[i]; let d = 0; for (let j = 0; j < v.length; j++) d += v[j] * qv[j]; s[i] = d; }
    return topIndices(s, k).map((i) => this.chunks[i].chunkId);
  }
}

/**
 * Full answer flow. `models` = { embedQuery(text) -> vector, translate(text, lang) -> English,
 * rerank(query, texts) -> [0, 1] scores }, all async. Returns the decision and the trace.
 */
export async function ragAnswer(question, index, models, opts = {}) {
  const o = { ...DEFAULTS, ...opts, thresholds: { ...DEFAULTS.thresholds, ...(opts.thresholds || {}) } };
  const empty = { refused: true, text: REFUSAL_TEXT, citations: [], topScore: 0, reason: "no text loaded or empty question" };
  if (!index.chunks.length || !question.trim()) return empty;
  const q = cleanQuery(question);
  const lang = detectLanguage(q);
  const t0 = now();
  const qEn = lang !== "en" && models.translate ? await models.translate(q, lang) : q;
  const t1 = now();
  const keep = o.books ? (c) => o.books.has(c.bookId) : null;
  // pipeline.ask: with a book filter, over-fetch 4x, fuse, then keep the selected books.
  const k = o.candidates, pool = keep ? k * 4 : k;
  const bm = (text) => index.bm25.search(text, pool).map(([id]) => id);
  const lists = [[bm(o.expand === false ? qEn : expandQuery(qEn)), 1 - o.denseWeight]];
  if (qEn !== q) lists.push([bm(q), 1 - o.denseWeight]);
  for (const qq of [...new Set([q, qEn])]) lists.push([index.dense(await models.embedQuery(qq), pool), o.denseWeight]);
  let fused = weightedRrf(lists, o.rrfK).slice(0, pool).map((id) => index.chunks[index.byId.get(id)]);
  if (keep) fused = fused.filter(keep);
  const cand = fused.slice(0, k);
  const t2 = now();
  const scores = cand.length ? await rerankPassages(models, qEn, cand.map((c) => c.text), o) : [];
  const t3 = now();
  const order = cand.map((_, j) => j).sort((a, b) => scores[b] - scores[a] || a - b).slice(0, o.topRanked);
  const idf = index.idf();
  const ranked = order.map((j) => ({ chunk: cand[j], score: scores[j], support: idf.support(qEn, index.indexTextFor(cand[j])) }));
  const t = o.thresholds[lang] ?? o.defaultThreshold;
  const top = ranked[0] || { score: 0, support: 0 };
  const trace = { lang, qEn, ranked: ranked.map((r) => [r.chunk.chunkId, r.score, r.support]), threshold: t,
    ms: { translate: t1 - t0, retrieve: t2 - t1, rerank: t3 - t2 } };
  const base = { topScore: top.score, lang, trace };
  if (!ranked.length || top.score < t) return { ...base, refused: true, text: REFUSAL_TEXT, citations: [], reason: `retrieval gate: score below threshold (${lang})` };
  if (top.support < o.minSupport) return { ...base, refused: true, text: REFUSAL_TEXT, citations: [], reason: "support check: passage does not cover the question" };
  let context = ranked.filter((r) => r.score >= t && r.support >= o.minSupport).slice(0, o.topContext);
  if (o.verify) {
    const kept = await o.verify(q, qEn, context);
    if (!kept.length) return { ...base, refused: true, text: REFUSAL_TEXT, citations: [], reason: "evidence check: cited passages do not contain the asked entity" };
    context = kept;
  }
  const cites = context.map((r) => r.chunk);
  const text = cites.map((c) => `${c.text} [${c.bookId} p.${c.pageStart} #${c.chunkId}]`).join("\n");
  return { ...base, refused: false, text, citations: cites, scores: context.map((r) => r.score) };
}

/** Max-window cross-encoder score per passage (rerank.Reranker.rerank with window_words). */
export async function rerankPassages(models, query, texts, o = DEFAULTS) {
  if (models.rerankPassages) return models.rerankPassages(query, texts);
  const spans = texts.map((t) => rerankWindows(t, o.rerankWindow, o.rerankStride));
  const flat = await models.rerank(query, spans.flat());
  const out = [];
  let pos = 0;
  for (const sp of spans) { out.push(Math.max(...flat.slice(pos, pos + sp.length))); pos += sp.length; }
  return out;
}

const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
