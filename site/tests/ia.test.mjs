import test from "node:test";
import assert from "node:assert/strict";
import * as ia from "../js/ia.js";
import * as core from "../js/core.js";

const DOCS = { response: { docs: [{ identifier: "koranrodw00rodwuoft", title: "The Koran", creator: ["Rodwell"], date: "1915-01-01T00:00:00Z", "possible-copyright-status": "NOT_IN_COPYRIGHT", item_size: 27000000 }] } };
const resp = (status, body, headers = {}) => ({ status, ok: status < 400, json: async () => body, headers: { get: (k) => headers[k] } });

function clock() {
  const c = { t: 0, slept: [] };
  c.now = () => c.t;
  c.sleep = async (ms) => { c.slept.push(ms); c.t += ms; };
  return c;
}

test("buildQuery applies every filter and strips syntax", () => {
  const q = ia.buildQuery('talmud "x" (y)', { language: "pt", tradition: "judaism", creator: "Rodkinson", collection: "americana", yearFrom: 1800, yearTo: 1925, publicDomain: true });
  for (const p of ["(talmud x y)", "format:PDF", "language:(por", "subject:(Talmud", 'creator:"Rodkinson"', 'collection:"americana"', "date:[1800-01-01 TO 1925-12-31]", "NOT_IN_COPYRIGHT"]) assert.ok(q.includes(p), p);
  assert.throws(() => ia.buildQuery("  "));
  assert.ok(ia.searchUrl("x", { sort: "date_asc" }).includes("sort%5B%5D=date+asc"));
});

test("parseHits flattens fields", () => {
  const [h] = ia.parseHits(DOCS);
  assert.equal(h.creator, "Rodwell");
  assert.equal(h.date, "1915-01-01");
  assert.equal(h.sizeMb, 27);
  assert.ok(h.preview.endsWith("/details/koranrodw00rodwuoft"));
});

test("PoliteFetcher spaces calls and backs off on 429/503", async () => {
  const c = clock();
  const queue = [resp(429, null, { "Retry-After": "3" }), resp(503, null), resp(200, DOCS), resp(200, DOCS)];
  const f = new ia.PoliteFetcher({ minIntervalMs: 1000, fetchFn: async () => queue.shift(), sleep: c.sleep, now: c.now });
  await f.get("u");
  assert.deepEqual(c.slept, [3000, 2000]); // Retry-After, then doubled backoff
  await f.get("u");
  assert.equal(c.slept.at(-1), 1000);
  const bad = new ia.PoliteFetcher({ minIntervalMs: 0, maxRetries: 1, fetchFn: async () => resp(503), sleep: c.sleep, now: c.now });
  await assert.rejects(bad.get("u"));
});

test("search caches results", async () => {
  const c = clock();
  let calls = 0;
  const f = new ia.PoliteFetcher({ minIntervalMs: 0, fetchFn: async () => { calls++; return resp(200, DOCS); }, sleep: c.sleep, now: c.now });
  const store = new Map();
  const cache = { get: async (k) => store.get(k), put: async (v) => store.set(v.k, v) };
  const a = await ia.search(f, cache, "koran", { language: "en" });
  const b = await ia.search(f, cache, "koran", { language: "en" });
  assert.equal(calls, 1);
  assert.deepEqual(a, b);
});

test("chooseFile prefers a PDF under the cap, falls back to text", () => {
  assert.equal(ia.chooseFile([{ name: "a.pdf", size: "10" }, { name: "a_djvu.txt", size: "5" }]).kind, "pdf");
  assert.equal(ia.chooseFile([{ name: "a.pdf", size: String(80e6) }, { name: "a_djvu.txt", size: "5" }]).kind, "txt");
  assert.equal(ia.chooseFile([{ name: "a.epub" }]), null);
});

test("readCapped enforces the size limit", async () => {
  const body = (chunks) => ({ headers: { get: () => null }, body: { getReader: () => { let i = 0; return { read: async () => (i < chunks.length ? { done: false, value: chunks[i++] } : { done: true }), cancel: async () => {} }; } } });
  assert.equal((await ia.readCapped(body([new Uint8Array(3), new Uint8Array(2)]), 10)).length, 5);
  await assert.rejects(ia.readCapped(body([new Uint8Array(8), new Uint8Array(8)]), 10));
});

test("clusterHits, rankRelated and bookTags", () => {
  const mk = (id, t) => ({ identifier: id, title: t, subject: "", description: "" });
  const hits = [mk("a", "talmud sabbath mishnah tract"), mk("b", "talmud sabbath mishnah laws"), mk("c", "mishnah talmud sabbath"), mk("d", "koran sura mecca"), mk("e", "koran sura medina"), mk("f", "koran sura notes")];
  const vecs = hits.map((h) => core.hashEmbed(ia.hitText(h)));
  const groups = ia.clusterHits(hits, vecs, 2);
  assert.equal(groups.reduce((n, g) => n + g.hits.length, 0), 6);
  const sets = groups.map((g) => g.hits.map((h) => h.identifier).sort().join(""));
  assert.ok(sets.includes("abc") && sets.includes("def"));
  const cen = ia.centroid([core.hashEmbed("talmud sabbath mishnah")]);
  const ranked = ia.rankRelated(hits, vecs, cen, new Set(["a"]));
  assert.ok(["b", "c"].includes(ranked[0][0].identifier));
  assert.ok(ia.relatedQuery(["talmud talmud sabbath"]).startsWith("talmud"));
  assert.deepEqual(ia.bookTags([{ bookId: "t" }, { bookId: "t" }, { bookId: "k" }], [0, 0, 1], [["sabbath", "law"], ["sura", "mecca"]]), { t: ["sabbath / law"], k: ["sura / mecca"] });
});

test("filterHits enforces year range, author, collection and public domain", () => {
  const hits = [
    { identifier: "a", date: "1880-01-01", creator: "Rodkinson, Michael", collection: "americana", copyrightStatus: "NOT_IN_COPYRIGHT", licenseUrl: "" },
    { identifier: "b", date: "1990", creator: "Someone", collection: "americana", copyrightStatus: "", licenseUrl: "http://creativecommons.org/licenses/by-nc/4.0/" },
    { identifier: "c", date: "", creator: "Rodkinson", collection: "toronto", copyrightStatus: "", licenseUrl: "http://creativecommons.org/publicdomain/zero/1.0/" },
  ];
  const ids = (f) => ia.filterHits(hits, f).map((h) => h.identifier).join("");
  assert.equal(ids({}), "abc");
  assert.equal(ids({ yearFrom: 1800, yearTo: 1925 }), "a");
  assert.equal(ids({ creator: "rodkinson" }), "ac");
  assert.equal(ids({ collection: "Americana" }), "ab");
  assert.equal(ids({ publicDomain: true }), "ac");
});

test("licenseLabel names the license shown with each hit", () => {
  assert.equal(ia.licenseLabel({ licenseUrl: "http://creativecommons.org/publicdomain/mark/1.0/" }), "Public Domain Mark");
  assert.equal(ia.licenseLabel({ licenseUrl: "http://creativecommons.org/publicdomain/zero/1.0/" }), "CC0");
  assert.equal(ia.licenseLabel({ licenseUrl: "https://creativecommons.org/licenses/by-sa/3.0/" }), "CC BY-SA 3.0");
  assert.equal(ia.licenseLabel({ copyrightStatus: "NOT_IN_COPYRIGHT" }), "Public domain (not in copyright)");
  assert.equal(ia.licenseLabel({}), "License not stated");
});

test("clusterHits separates themes and rankRelated prefers the library topic", () => {
  const mk = (t) => ({ identifier: t, title: t, subject: "", description: "" });
  const texts = ["koran sura prophet", "koran sura mecca", "koran prophet mecca", "buddha sutra dharma", "buddha dharma monk", "sutra monk dharma"];
  const hits = texts.map(mk), vecs = texts.map((t) => core.hashEmbed(t));
  const groups = ia.clusterHits(hits, vecs, 3);
  assert.ok(groups.length >= 2);
  for (const g of groups) assert.ok(g.hits.every((h) => h.title.includes("koran")) || g.hits.every((h) => !h.title.includes("koran")));
  const cen = ia.centroid([core.hashEmbed("koran sura prophet mecca")]);
  const ranked = ia.rankRelated(hits, vecs, cen, new Set(["koran sura prophet"]), 2);
  assert.ok(ranked.every(([h]) => h.title.includes("koran")));
  assert.ok(!ranked.some(([h]) => h.identifier === "koran sura prophet"));
  assert.ok(ia.parseHits({ response: { docs: [{ identifier: "x", language: ["eng"] }] } })[0].language === "eng");
});
