// Internet Archive search panel, mounted into #archive-mount. Uses the hooks on globalThis.ScriptureRAG.
import "./ann.js";
import * as core from "./core.js";
import * as ia from "./ia.js";
import * as lib from "./library.js";
import { el } from "./ui.js";

const fetcher = new ia.PoliteFetcher();
const opt = (v, label) => { const o = el("option", { value: v }, label); return o; };
const sel = (id, options) => el("select", { id }, ...options.map(([v, l]) => opt(v, l)));
const field = (label, node) => el("label", { class: "ia-field" }, el("span", { class: "small muted" }, label), node);

export function mount(root) {
  if (!root || root.dataset.mounted) return;
  root.dataset.mounted = "1";
  const api = globalThis.ScriptureRAG;
  const cache = { get: api.cacheGet, put: api.cachePut };
  const q = el("input", { id: "iaQuery", type: "search", placeholder: "Search words, e.g. talmud rodkinson", "aria-label": "Internet Archive search" });
  const lang = sel("iaLang", [["", "Any language"], ["en", "English"], ["pt", "Portuguese"], ["es", "Spanish"]]);
  const trad = sel("iaTrad", [["", "Any tradition"], ...Object.keys(ia.TRADITIONS).map((t) => [t, t[0].toUpperCase() + t.slice(1)])]);
  const sort = sel("iaSort", [["relevance", "Relevance"], ["date_asc", "Oldest"], ["date_desc", "Newest"], ["downloads", "Most downloaded"], ["title", "Title"]]);
  const creator = el("input", { id: "iaCreator", type: "text" });
  const coll = el("input", { id: "iaColl", type: "text" });
  const from = el("input", { id: "iaFrom", type: "number", min: "0", max: "2100", inputmode: "numeric" });
  const to = el("input", { id: "iaTo", type: "number", min: "0", max: "2100", inputmode: "numeric" });
  const check = (id, label, on = true) => { const c = el("input", { id, type: "checkbox" }); c.checked = on; return el("label", { class: "small" }, c, ` ${label}`); };
  const pdf = check("iaPdf", "PDF only"), pd = check("iaPd", "Public domain only"), group = check("iaGroup", "Group by topic");
  const go = el("button", { class: "btn primary", type: "button", id: "iaGo" }, "Search");
  const rel = el("button", { class: "btn", type: "button", id: "iaRelated" }, "Related to my library");
  const results = el("div", { id: "iaResults", "aria-live": "polite" });
  root.replaceChildren(
    el("div", { class: "ia-search" }, q, go, rel),
    el("div", { class: "ia-filters" }, field("Language", lang), field("Tradition", trad), field("Sort", sort), field("Creator", creator), field("Collection", coll), field("From year", from), field("To year", to)),
    el("div", { class: "ia-filters" }, pdf, pd, group),
    el("p", { class: "small muted" }, "License fields come from the uploader and are evidence, not proof. Results are cached in this browser for an hour, and requests are spaced one second apart."),
    results);

  const filters = () => ({
    language: lang.value || undefined, tradition: trad.value || undefined, sort: sort.value, creator: creator.value, collection: coll.value,
    yearFrom: Number(from.value) || undefined, yearTo: Number(to.value) || undefined,
    pdfOnly: pdf.querySelector("input").checked, publicDomain: pd.querySelector("input").checked,
  });

  function row(h, score) {
    const ok = el("input", { type: "checkbox" });
    const btn = el("button", { class: "btn", type: "button" }, "Import");
    btn.disabled = true;
    ok.onchange = () => { btn.disabled = !ok.checked; };
    btn.onclick = () => importHit(h, btn);
    const role = lib.suggestRole(h.title, h.subject);
    return el("li", { class: "ia-hit" },
      el("a", { href: h.preview, target: "_blank", rel: "noopener noreferrer" }, h.title || h.identifier),
      el("div", { class: "small muted" }, `${h.creator || "Unknown creator"}, ${h.date || "n.d."}${score != null ? `, similarity ${score.toFixed(2)}` : ""}, suggested role: ${role}`),
      el("div", { class: "small" }, `Copyright status: ${h.copyrightStatus || "not stated"}. License: ${h.licenseUrl || "not stated"}.`),
      el("label", { class: "small" }, ok, " I reviewed the license evidence"), btn);
  }

  function render(hits, scores) {
    results.replaceChildren();
    if (!hits.length) { results.append(el("p", { class: "muted" }, "No results.")); return; }
    if (group.querySelector("input").checked && hits.length >= 4 && !scores) {
      const groups = ia.clusterHits(hits, hits.map((h) => core.hashEmbed(ia.hitText(h))));
      for (const g of groups) results.append(el("h4", {}, `${g.label} (${g.hits.length})`), el("ul", { class: "list" }, ...g.hits.map((h) => row(h))));
    } else {
      results.append(el("ul", { class: "list" }, ...hits.map((h, i) => row(h, scores?.[i]))));
    }
  }

  async function search() {
    try { render(await ia.search(fetcher, cache, q.value, filters())); } catch (e) { api.toast(`Search failed: ${e.message}`, "error"); }
  }

  async function related() {
    const chunks = api.state.chunks.slice(0, 2000);
    if (!chunks.length) { api.toast("Load a book first", "error"); return; }
    try {
      const hits = await ia.search(fetcher, cache, ia.relatedQuery(chunks.map((c) => c.text)), { publicDomain: true }, { rows: 40 });
      const cen = ia.centroid(chunks.map((c) => core.hashEmbed(c.text)));
      const loaded = new Set(api.state.books.map((b) => b.source).filter(Boolean));
      const ranked = ia.rankRelated(hits, hits.map((h) => core.hashEmbed(ia.hitText(h))), cen, loaded);
      render(ranked.map(([h]) => h), ranked.map(([, s]) => s));
    } catch (e) { api.toast(`Search failed: ${e.message}`, "error"); }
  }

  async function importHit(h, btn) {
    btn.disabled = true;
    try {
      const meta = await (await fetcher.get(`https://archive.org/metadata/${encodeURIComponent(h.identifier)}`)).json();
      const files = meta.files || [];
      const file = ia.chooseFile(files);
      if (!file) throw new Error("No PDF or OCR text under 50 MB in this item");
      const dl = (name) => `https://archive.org/download/${encodeURIComponent(h.identifier)}/${name.split("/").map(encodeURIComponent).join("/")}`;
      api.toast(`Downloading ${file.name}`);
      let name = file.kind === "pdf" ? `${h.identifier}.pdf` : `${h.identifier}.txt`, bytes;
      try {
        bytes = await ia.readCapped(await fetcher.get(dl(file.name), { timeoutMs: 180000 }));
      } catch (e) {
        // Fallback: the OCR text file is smaller and served with the same CORS headers.
        const txt = files.find((f) => /_djvu\.txt$/i.test(f.name));
        if (!txt || file.kind === "txt") throw e;
        name = `${h.identifier}.txt`;
        bytes = await ia.readCapped(await fetcher.get(dl(txt.name), { timeoutMs: 180000 }));
      }
      const m = meta.metadata || {};
      const subject = [].concat(m.subject || []).join("; ");
      await api.importBytes(name, bytes, {
        title: h.title || h.identifier, source: h.identifier, date: h.date, license: `${h.copyrightStatus || "not stated"}; ${h.licenseUrl || "no license URL"}`,
        role: lib.suggestRole(h.title, subject), tradition: lib.suggestTradition(h.title, subject),
      });
      api.toast(`Imported ${h.title}`);
    } catch (e) { api.toast(`Import failed: ${e.message}`, "error", 6000); btn.disabled = false; }
  }

  go.onclick = search;
  rel.onclick = related;
  q.onkeydown = (e) => { if (e.key === "Enter") search(); };
}
