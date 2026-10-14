// Library data hooks for the UI: role tags, compare mode, outliers and the manifest.
// Pure functions over the chunk/book records stored in IndexedDB; no DOM access.
import * as core from "./core.js";

export const ROLES = ["primary", "commentary"];
const COMMENTARY = /\b(commentar\w*|tafsir|exegesis|notes on|annotat\w*|introduction to|history of|essays?|studies|scholar\w*|criticism|critical|discourse|dictionary|concordance|handbook|lectures?|treatise|companion|guide to|interpretation|theology|sermons?)\b/i;
const PRIMARY = /\b(bible|koran|qur.?an|alcoran|talmud|mishna\w*|torah|gospels?|psalms?|dhammapada|upanishads?|bhagavad\w*|tao te ching|sutras?|vedas?|scriptures?)\b/i;
const TRADITIONS = {
  christianity: /bible|gospel|testament|christian|psalm/g, judaism: /talmud|mishna|torah|midrash|jew|hebrew/g,
  islam: /koran|qur.?an|alcoran|islam|hadith|tafsir|sura|muhammad/g, buddhism: /buddh|dhamma|sutta|pali/g,
  hinduism: /veda|upanishad|bhagavad|hindu/g, taoism: /\btao\b|laozi|lao.tzu|chuang/g,
};

/** "primary" or "commentary" from title and subject; commentary cues win. O(t). */
export function suggestRole(title = "", subject = "") {
  const t = `${title} ${subject}`;
  if (COMMENTARY.test(t)) return "commentary";
  return PRIMARY.test(t) || !t.trim() ? "primary" : "commentary";
}

export function suggestTradition(title = "", subject = "") {
  const t = `${title} ${subject}`.toLowerCase();
  let best = "", n = 0;
  for (const [k, re] of Object.entries(TRADITIONS)) { const m = (t.match(re) || []).length; if (m > n) { best = k; n = m; } }
  return best;
}

/** Book ids passing role/tradition/explicit filters (empty filters keep all). */
export function selectBooks(books, { role, tradition, ids } = {}) {
  return new Set(books.filter((b) => (!role || (b.role || "primary") === role) && (!tradition || b.tradition === tradition) && (!ids?.size || ids.has(b.bookId))).map((b) => b.bookId));
}

/** Run core.answer over a subset of chunks with its own BM25. O(n_subset) build + ask. */
export function answerIn(question, chunks, bookIds, qv, opts) {
  const pool = chunks.filter((c) => bookIds.has(c.bookId));
  const bm = new core.BM25();
  for (const c of pool) bm.add(c.chunkId, c.text);
  return core.answer(question, pool, bm, qv, opts);
}

/** Answers kept apart by role: { primary: res, commentary: res }. */
export function answerByRole(question, books, chunks, qv, opts) {
  const out = {};
  for (const role of ROLES) { const ids = selectBooks(books, { role }); if (ids.size) out[role] = answerIn(question, chunks, ids, qv, opts); }
  return out;
}

/** Compare mode: one answer per book or per tradition, each citing only its group. O(G * ask). */
export function compare(question, books, chunks, qv, { by = "tradition", ...opts } = {}) {
  const groups = new Map();
  for (const b of books) {
    const key = by === "book" ? b.bookId : by === "role" ? b.role || "primary" : b.tradition || "unlabeled";
    if (!groups.has(key)) groups.set(key, new Set());
    groups.get(key).add(b.bookId);
  }
  return [...groups.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([group, ids]) => ({ group, result: answerIn(question, chunks, ids, qv, opts) }));
}

/** Passages far from their cluster centroid: z-score of distance within each cluster. O(n d). */
export function outliers(vecs, labels, zThreshold = 2.5) {
  const byC = new Map();
  labels.forEach((c, i) => { if (!byC.has(c)) byC.set(c, []); byC.get(c).push(i); });
  const out = [];
  for (const [c, idx] of byC) {
    if (c < 0 || idx.length < 3) continue;
    const d = vecs[idx[0]].length, cen = new Float64Array(d);
    for (const i of idx) for (let j = 0; j < d; j++) cen[j] += vecs[i][j] / idx.length;
    const dist = idx.map((i) => Math.sqrt(vecs[i].reduce((s, v, j) => s + (v - cen[j]) ** 2, 0)));
    const mean = dist.reduce((a, b) => a + b, 0) / dist.length;
    const sd = Math.sqrt(dist.reduce((a, b) => a + (b - mean) ** 2, 0) / dist.length);
    if (!sd) continue;
    idx.forEach((i, k) => { const z = (dist[k] - mean) / sd; if (z > zThreshold) out.push({ index: i, cluster: c, distance: dist[k], z }); });
  }
  return out.sort((a, b) => b.z - a.z);
}

export const MANIFEST_FIELDS = ["bookId", "title", "role", "tradition", "date", "sourceUrl", "licenseStatus", "pages", "passages", "addedAt"];

/** One audit row per book. O(B + n). */
export function manifest(books, chunks) {
  const counts = new Map();
  for (const c of chunks) counts.set(c.bookId, (counts.get(c.bookId) || 0) + 1);
  return books.map((b) => ({
    bookId: b.bookId, title: b.title, role: b.role || "primary", tradition: b.tradition || "", date: b.date || "",
    sourceUrl: b.source ? `https://archive.org/details/${b.source}` : "local file", licenseStatus: b.license || "not recorded",
    pages: b.pages, passages: counts.get(b.bookId) || 0, addedAt: b.addedAt ? new Date(b.addedAt).toISOString() : "",
  }));
}

export function manifestCsv(rows) {
  const esc = (v) => { const s = String(v ?? ""); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
  return [MANIFEST_FIELDS.join(","), ...rows.map((r) => MANIFEST_FIELDS.map((f) => esc(r[f])).join(","))].join("\n");
}
