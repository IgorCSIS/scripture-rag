// Browser library stages vs n passages: outliers, auto-tags, manifest, role-scoped answer, compare.
// Usage: node bench/bench_library.mjs > docs/benchmarks/library_stages.json
import * as core from "../site/js/core.js";
import * as ia from "../site/js/ia.js";
import * as lib from "../site/js/library.js";

let seed = 7;
const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31);
const WORDS = "lord shepherd water bread light law prophet king mercy faith prayer temple river mountain covenant".split(" ");
const text = () => Array.from({ length: 40 }, () => WORDS[Math.floor(rnd() * WORDS.length)]).join(" ");
const median = (a) => a.sort((x, y) => x - y)[Math.floor(a.length / 2)];
const time = (fn, reps = 5) => median(Array.from({ length: reps }, () => { const t = performance.now(); fn(); return performance.now() - t; }));

const out = { sizes: [], stages: {} };
for (const n of [500, 1000, 2000, 4000, 8000]) {
  const B = 8;
  const books = Array.from({ length: B }, (_, b) => ({ bookId: `b${b}`, title: `Book ${b}`, role: b % 2 ? "commentary" : "primary", tradition: ["christianity", "islam", "judaism"][b % 3], pages: 10 }));
  const chunks = Array.from({ length: n }, (_, i) => { const t = text(); return { chunkId: `b${i % B}-${i}`, bookId: `b${i % B}`, text: t, pageStart: 1, pageEnd: 1, vec: core.hashEmbed(t) }; });
  const labels = chunks.map(() => Math.floor(rnd() * 6));
  const kw = Array.from({ length: 6 }, (_, k) => [WORDS[k], WORDS[k + 6]]);
  const q = "who is the shepherd", qv = core.hashEmbed(q);
  const vecs = chunks.map((c) => c.vec);
  const r = {
    outliers: time(() => lib.outliers(vecs, labels)),
    auto_tags: time(() => ia.bookTags(chunks, labels, kw)),
    manifest_csv: time(() => lib.manifestCsv(lib.manifest(books, chunks))),
    role_scoped_answer: time(() => lib.answerIn(q, chunks, lib.selectBooks(books, { role: "commentary" }), qv, { threshold: 0.3 }), 3),
    compare_by_tradition: time(() => lib.compare(q, books, chunks, qv, { by: "tradition", threshold: 0.3 }), 3),
  };
  out.sizes.push(n);
  for (const [k, v] of Object.entries(r)) (out.stages[k] ||= []).push(Number(v.toFixed(3)));
}
// Internet Archive panel stages vs r hits, plus the book selector vs B books.
out.ia = { sizes: [], stages: {} };
for (const r of [25, 50, 100, 200, 400]) {
  const hits = Array.from({ length: r }, (_, i) => ({ identifier: `id${i}`, title: text().slice(0, 40), subject: text().slice(0, 30), description: text(), creator: i % 3 ? "Rodwell" : "Sale", collection: "americana", date: String(1600 + (i % 300)), copyrightStatus: i % 4 ? "NOT_IN_COPYRIGHT" : "", licenseUrl: "" }));
  const vecs = hits.map((h) => core.hashEmbed(ia.hitText(h)));
  const c = ia.centroid(vecs);
  const books = Array.from({ length: r }, (_, b) => ({ bookId: `b${b}`, role: b % 2 ? "commentary" : "primary", tradition: "islam" }));
  const ids = new Set(books.filter((_, b) => b % 3 === 0).map((b) => b.bookId));
  const t = {
    filterHits: time(() => ia.filterHits(hits, { yearFrom: 1650, yearTo: 1800, creator: "rodwell", collection: "americana", publicDomain: true })),
    clusterHits: time(() => ia.clusterHits(hits, vecs), 3),
    rankRelated: time(() => ia.rankRelated(hits, vecs, c)),
    book_selector: time(() => lib.selectBooks(books, { role: "primary", tradition: "islam", ids })),
  };
  out.ia.sizes.push(r);
  for (const [k, v] of Object.entries(t)) (out.ia.stages[k] ||= []).push(Number(v.toFixed(4)));
}
console.log(JSON.stringify(out, null, 2));
