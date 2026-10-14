import test from "node:test";
import assert from "node:assert/strict";
import * as lib from "../js/library.js";
import * as core from "../js/core.js";

test("role and tradition suggestions", () => {
  assert.equal(lib.suggestRole("The Koran", "Koran"), "primary");
  assert.equal(lib.suggestRole("A comprehensive commentary on the Quran"), "commentary");
  assert.equal(lib.suggestRole("The history of the Talmud"), "commentary");
  assert.equal(lib.suggestTradition("New edition of the Babylonian Talmud"), "judaism");
  assert.equal(lib.suggestTradition("Cookbook"), "");
});

const books = [
  { bookId: "bible", title: "Bible", role: "primary", tradition: "christianity", pages: 1, source: "x", license: "NOT_IN_COPYRIGHT" },
  { bookId: "notes", title: "Notes on the Psalms", role: "commentary", tradition: "christianity", pages: 1 },
  { bookId: "koran", title: "Koran", role: "primary", tradition: "islam", pages: 1 },
];
const chunks = [
  ...core.chunkPages("bible", ["The Lord is my shepherd; I shall not want. He maketh me to lie down in green pastures."], 50, 5),
  ...core.chunkPages("notes", ["The shepherd image in the psalm describes care and guidance for the flock."], 50, 5),
  ...core.chunkPages("koran", ["Praise be to God, Lord of the worlds, the compassionate, the merciful."], 50, 5),
];
chunks.forEach((c) => { c.vec = core.hashEmbed(c.text); });

test("answers by role never mix primary and commentary", () => {
  const q = "Who is my shepherd?";
  const r = lib.answerByRole(q, books, chunks, core.hashEmbed(q), { threshold: 0.1 });
  assert.ok(r.primary.citations.every((c) => c.bookId !== "notes"));
  assert.ok(r.commentary.citations.every((c) => c.bookId === "notes"));
});

test("compare groups by tradition and by book", () => {
  const q = "shepherd";
  const t = lib.compare(q, books, chunks, core.hashEmbed(q), { by: "tradition", threshold: 0.1 });
  assert.deepEqual(t.map((g) => g.group), ["christianity", "islam"]);
  assert.ok(t[0].result.citations.every((c) => ["bible", "notes"].includes(c.bookId)));
  assert.equal(lib.compare(q, books, chunks, core.hashEmbed(q), { by: "book" }).length, 3);
  assert.equal(lib.selectBooks(books, { role: "primary", tradition: "islam" }).size, 1);
});

test("outliers flag the far point", () => {
  const vecs = Array.from({ length: 30 }, (_, i) => [Math.sin(i) * 0.05, Math.cos(i) * 0.05]);
  vecs.push([3, 3]);
  const out = lib.outliers(vecs, vecs.map(() => 0));
  assert.equal(out[0].index, 30);
  assert.deepEqual(lib.outliers(vecs.slice(0, 2), [0, 0]), []);
});

test("manifest rows and CSV", () => {
  const rows = lib.manifest(books, chunks);
  assert.equal(rows[0].sourceUrl, "https://archive.org/details/x");
  assert.equal(rows[2].licenseStatus, "not recorded");
  const csv = lib.manifestCsv(rows).split("\n");
  assert.ok(csv[0].startsWith("bookId,title,role"));
  assert.equal(csv.length, 4);
});
