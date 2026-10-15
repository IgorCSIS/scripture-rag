// Internet Archive search, polite fetching and discovery helpers for the browser demo.
// Endpoints (all send Access-Control-Allow-Origin: *):
//   https://archive.org/advancedsearch.php   search, O(1) requests, O(r) parse for r rows
//   https://archive.org/metadata/{id}        files, sizes, license fields
//   https://archive.org/download/{id}/{file} _djvu.txt (CORS enabled) or, if no OCR text, PDF bytes
import * as core from "./core.js";

export const SEARCH_URL = "https://archive.org/advancedsearch.php";
export const MAX_BYTES = 50 * 1024 * 1024;
export const LANGUAGES = { en: "(eng OR English)", pt: "(por OR Portuguese)", es: "(spa OR Spanish)" };
export const TRADITIONS = {
  christianity: "(Bible OR Christianity OR Gospels OR Theology)",
  judaism: "(Talmud OR Mishnah OR Judaism OR Torah OR Midrash)",
  islam: "(Koran OR Quran OR Islam OR Hadith OR Tafsir)",
  buddhism: "(Buddhism OR Dhammapada OR Sutra OR Pali)",
  hinduism: "(Hinduism OR Vedas OR Upanishads OR Bhagavadgita)",
  taoism: "(Taoism OR Tao OR Laozi OR Zhuangzi)",
};
export const SORTS = { relevance: "", date_asc: "date asc", date_desc: "date desc", downloads: "downloads desc", title: "titleSorter asc" };
const FIELDS = ["identifier", "title", "creator", "date", "language", "possible-copyright-status", "licenseurl", "item_size", "description", "subject", "collection"];
const RETRY = new Set([429, 500, 502, 503, 504]);

const clean = (s) => String(s || "").replace(/["()[\]{}:\\^~*?!]/g, "").trim();
const flat = (v) => (Array.isArray(v) ? v.join("; ") : v == null ? "" : String(v));

/** Lucene query from words and filters. */
export function buildQuery(words, f = {}) {
  const w = clean(words);
  if (!w && !(f.tradition || f.creator || f.collection || f.subject)) throw new Error("Empty search query");
  const parts = w ? [`(${w})`] : [];
  parts.push("mediatype:texts");
  if (f.pdfOnly !== false) parts.push("format:PDF");
  if (f.language) parts.push(`language:${LANGUAGES[f.language]}`);
  if (f.tradition) parts.push(`subject:${TRADITIONS[f.tradition]}`);
  for (const k of ["subject", "creator", "collection"]) if (clean(f[k])) parts.push(`${k}:"${clean(f[k])}"`);
  if (f.yearFrom || f.yearTo) parts.push(`date:[${f.yearFrom ? `${f.yearFrom}-01-01` : "*"} TO ${f.yearTo ? `${f.yearTo}-12-31` : "*"}]`);
  if (f.publicDomain) parts.push("possible-copyright-status:NOT_IN_COPYRIGHT");
  return parts.join(" AND ");
}

export function searchUrl(q, { rows = 30, page = 1, sort = "relevance" } = {}) {
  const p = new URLSearchParams({ q, rows: String(Math.min(50, Math.max(1, rows))), page: String(page), output: "json" });
  for (const f of FIELDS) p.append("fl[]", f);
  if (SORTS[sort]) p.append("sort[]", SORTS[sort]);
  return `${SEARCH_URL}?${p}`;
}

export function parseHits(json) {
  return (json?.response?.docs || []).filter((d) => d.identifier).map((d) => ({
    identifier: flat(d.identifier), title: flat(d.title), creator: flat(d.creator), date: flat(d.date).slice(0, 10),
    copyrightStatus: flat(d["possible-copyright-status"]), licenseUrl: flat(d.licenseurl),
    sizeMb: Math.round((Number(d.item_size) || 0) / 1e5) / 10, description: flat(d.description).slice(0, 500),
    language: flat(d.language), subject: flat(d.subject), collection: flat(d.collection), preview: `https://archive.org/details/${flat(d.identifier)}`,
  }));
}

/** Spaced, retrying fetch: min gap between calls, exponential backoff on 429/5xx, per call timeout. */
export class PoliteFetcher {
  constructor({ minIntervalMs = 1000, maxRetries = 3, timeoutMs = 20000, fetchFn = (...a) => globalThis.fetch(...a), sleep = (ms) => new Promise((r) => setTimeout(r, ms)), now = () => Date.now() } = {}) {
    Object.assign(this, { minIntervalMs, maxRetries, timeoutMs, fetchFn, sleep, now, last: -1e12 });
  }
  async get(url, opts = {}) {
    let delay = 1000;
    for (let attempt = 0; ; attempt++) {
      const wait = this.minIntervalMs - (this.now() - this.last);
      if (wait > 0) await this.sleep(wait);
      this.last = this.now();
      const ctl = typeof AbortController !== "undefined" ? new AbortController() : null;
      const timer = ctl ? setTimeout(() => ctl.abort(), opts.timeoutMs || this.timeoutMs) : null;
      let res;
      try { res = await this.fetchFn(url, { signal: ctl?.signal }); } finally { if (timer) clearTimeout(timer); }
      if (!RETRY.has(res.status) || attempt >= this.maxRetries) {
        if (!res.ok) throw new Error(`HTTP ${res.status} from archive.org`);
        return res;
      }
      const ra = Number(res.headers?.get?.("Retry-After"));
      await this.sleep(Number.isFinite(ra) && ra > 0 ? ra * 1000 : delay);
      delay *= 2;
    }
  }
}

/** Search with a cache (any get/put store with { k, v, ts }; IndexedDB "meta" in the app). */
export async function search(fetcher, cache, words, filters = {}, { ttlMs = 3600e3, rows = 30 } = {}) {
  const q = buildQuery(words, filters);
  const url = searchUrl(q, { rows, sort: filters.sort });
  const key = `ia:${url}`;
  const hit = await cache.get(key);
  if (hit && fetcher.now() - hit.ts < ttlMs) return hit.v;
  const hits = parseHits(await (await fetcher.get(url)).json());
  await cache.put({ k: key, v: hits, ts: fetcher.now() });
  return hits;
}

/** Choose a file: PDF under the size cap, else the OCR text (_djvu.txt) as a fallback. */
// archive.org serves _djvu.txt with CORS headers but most item PDFs without them, so a browser
// fetch of the PDF fails. Prefer the OCR text of the same scan; use the PDF only when there is none.
export function chooseFile(files, maxBytes = MAX_BYTES) {
  const txt = files.find((f) => /_djvu\.txt$/i.test(f.name) && Number(f.size || 0) <= maxBytes);
  if (txt) return { name: txt.name, kind: "txt", size: Number(txt.size || 0) };
  const pdf = files.find((f) => /\.pdf$/i.test(f.name) && Number(f.size || 0) <= maxBytes);
  if (pdf) return { name: pdf.name, kind: "pdf", size: Number(pdf.size || 0) };
  return null;
}

/** Read bytes with a hard size cap even when Content-Length is missing. */
export async function readCapped(res, maxBytes = MAX_BYTES) {
  const len = Number(res.headers?.get?.("Content-Length") || 0);
  if (len > maxBytes) throw new Error("File is larger than 50 MB");
  if (!res.body?.getReader) {
    const buf = new Uint8Array(await res.arrayBuffer());
    if (buf.length > maxBytes) throw new Error("File is larger than 50 MB");
    return buf;
  }
  const reader = res.body.getReader();
  const parts = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.length;
    if (total > maxBytes) { await reader.cancel(); throw new Error("File is larger than 50 MB"); }
    parts.push(value);
  }
  const out = new Uint8Array(total);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

export const hitText = (h) => `${h.title}. ${h.subject}. ${h.description}`;

/** Group hits by k-means over embeddings of title, subject and description; TF-IDF labels. */
export function clusterHits(hits, vecs, maxK = 5) {
  if (hits.length < 4) return [{ label: "results", hits }];
  const { labels } = core.autoCluster(vecs, Math.min(maxK, hits.length - 1));
  const kw = core.clusterKeywords(hits.map(hitText), labels, 3);
  const groups = new Map();
  hits.forEach((h, i) => { if (!groups.has(labels[i])) groups.set(labels[i], []); groups.get(labels[i]).push(h); });
  return [...groups.entries()].sort((a, b) => b[1].length - a[1].length)
    .map(([c, hs]) => ({ label: (kw[c] || []).join(" / ") || `group ${c}`, hits: hs }));
}

export function centroid(vecs) {
  const d = vecs[0].length, c = new Float32Array(d);
  for (const v of vecs) for (let i = 0; i < d; i++) c[i] += v[i];
  return core.normalize(c);
}

/** Most frequent content words of the library, used as the related-books query. */
export function relatedQuery(texts, n = 4) {
  const tf = new Map();
  for (const t of texts) for (const w of core.tokenize(t)) if (w.length > 3) tf.set(w, (tf.get(w) || 0) + 1);
  return [...tf.entries()].sort((a, b) => b[1] - a[1]).slice(0, n).map(([w]) => w).join(" ");
}

/** Rank hits by cosine to the library centroid; skip identifiers already loaded. O(r d + r log r). */
export function rankRelated(hits, vecs, c, exclude = new Set(), topN = 10) {
  return hits.map((h, i) => [h, core.dot(vecs[i], c)]).filter(([h]) => !exclude.has(h.identifier))
    .sort((a, b) => b[1] - a[1]).slice(0, topN);
}

/** Tag books with the labels of the clusters most of their passages fall in. O(n). */
export function bookTags(chunks, labels, keywords, topN = 2) {
  const per = new Map();
  chunks.forEach((c, i) => {
    if (!per.has(c.bookId)) per.set(c.bookId, new Map());
    const m = per.get(c.bookId);
    m.set(labels[i], (m.get(labels[i]) || 0) + 1);
  });
  const out = {};
  for (const [b, m] of per) out[b] = [...m.entries()].sort((x, y) => y[1] - x[1]).slice(0, topN).map(([k]) => (keywords[k] || []).slice(0, 2).join(" / "));
  return out;
}

/** Public domain per the uploader's fields: NOT_IN_COPYRIGHT status or a PD / CC0 license URL. */
export function isPublicDomain(h) {
  return h.copyrightStatus === "NOT_IN_COPYRIGHT" || /publicdomain|\/zero\//i.test(h.licenseUrl || "");
}

/** Short human license label shown next to every hit. */
export function licenseLabel(h) {
  const u = h.licenseUrl || "";
  if (/publicdomain\/mark/i.test(u)) return "Public Domain Mark";
  if (/\/zero\//i.test(u)) return "CC0";
  const cc = u.match(/licenses\/([a-z-]+)\/([\d.]+)/i);
  if (cc) return `CC ${cc[1].toUpperCase()} ${cc[2]}`;
  if (h.copyrightStatus === "NOT_IN_COPYRIGHT") return "Public domain (not in copyright)";
  return u ? "Other license" : "License not stated";
}

const year = (d) => { const m = String(d || "").match(/\d{4}/); return m ? Number(m[0]) : null; };
const has = (hay, needle) => String(hay || "").toLowerCase().includes(String(needle).toLowerCase().trim());

/** Client-side pass over returned hits so filters hold even when IA metadata is loose. O(r). */
export function filterHits(hits, f = {}) {
  return hits.filter((h) => {
    const y = year(h.date);
    if (f.yearFrom && (y == null || y < f.yearFrom)) return false;
    if (f.yearTo && (y == null || y > f.yearTo)) return false;
    if (clean(f.creator) && !has(h.creator, clean(f.creator))) return false;
    if (clean(f.collection) && !has(h.collection, clean(f.collection))) return false;
    if (f.publicDomain && !isPublicDomain(h)) return false;
    return true;
  });
}
