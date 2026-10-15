"""Round 2 experiments on the selection data (tune split + synthetic dev set). Held-out is never read here.

Each variant re-tunes the gate (per-language thresholds + support) on tune+dev and reports tune, dev and
pooled cards. Writes docs/eval/round2.json.

Usage::

    python scripts/round2.py [--only base,rrf,bm25,entity,diag]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import elite_eval as ee  # noqa: E402

from scripture_rag.normalize import default_normalizer  # noqa: E402
from scripture_rag.text_utils import STOPWORDS, tokenize  # noqa: E402

_CAP_RE = re.compile(r"(?<![.?!]\s)(?<!^)\b([A-Z][a-z]{2,})\b")
_NUM_RE = re.compile(r"\b\d+\b")
GENERIC = {"God", "Lord", "Koran", "Quran", "Bible", "What", "Who", "Which", "How", "Why", "When", "Where"}


def key_terms(q_en: str) -> set[str]:
    """Names (capitalized words not at the start) and numbers in the English question, normalized."""
    normz = default_normalizer()
    names = {normz.word(m.lower()) for m in _CAP_RE.findall(q_en) if m not in GENERIC}
    names = {n for n in names if n not in STOPWORDS}
    return names | set(_NUM_RE.findall(q_en))


def entity_ok(q_en: str, text: str) -> bool:
    """True when every key term of the question appears in the passage (5-letter stems, after normalization)."""
    keys = key_terms(q_en)
    if not keys:
        return True
    words = {t[:5] for t in tokenize(default_normalizer().text(text))}
    return all(k[:5] in words for k in keys)


def with_entity_filter(runs, chunks):
    """Drop ranked passages that miss a key name or number of the question."""
    out = []
    for r in runs:
        ranked = [x for x in r["ranked"] if entity_ok(r["q_en"], chunks[x[0]].text)]
        out.append({**r, "ranked": ranked})
    return out


_NLI: dict = {}


def nli_probs(premises: list[str], hypothesis: str) -> list[float]:
    """Entailment probability of the hypothesis given each premise (cached on disk)."""
    cache = _NLI.setdefault("cache", ee.DiskCache("nli"))
    keys = [ee.h(NLI_MODEL, p, hypothesis) for p in premises]
    miss = [i for i, k in enumerate(keys) if cache.get(k) is None]
    if miss:
        if "m" not in _NLI:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            _NLI["t"] = AutoTokenizer.from_pretrained(NLI_MODEL)
            _NLI["m"] = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).eval()
        import torch

        tok, m = _NLI["t"], _NLI["m"]
        ent = m.config.label2id.get("entailment", 0)
        with torch.no_grad():
            enc = tok([premises[i] for i in miss], [hypothesis] * len(miss), truncation=True, padding=True,
                      return_tensors="pt", max_length=256)
            p = torch.softmax(m(**enc).logits, -1)[:, ent].tolist()
        for i, v in zip(miss, p):
            cache.put(keys[i], v)
        cache.save()
    return [cache.get(k) for k in keys]


NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"


def nli_filter(runs, chunks, tau):
    """Keep ranked passages 1-3 whose NLI entailment of 'This passage answers: <question>' is >= tau."""
    out = []
    for r in runs:
        top = r["ranked"][:3]
        probs = nli_probs([chunks[i].text for i, _, _ in top], f"This passage answers the question: {r['q_en']}")
        out.append({**r, "ranked": [x for x, p in zip(top, probs) if p >= tau]})
    return out


def scored(name, items, runs, chunks, snippets):
    pol = ee.tune(items, runs, chunks, snippets, True, True)
    rows = ee.grade(items, runs, chunks, pol, snippets)
    res = {"variant": name, "policy": {"t": pol.t, "s_min": pol.s_min}}
    for sp in ("tune", "dev"):
        res[sp] = ee.card([r for r in rows if r["split"] == sp])
    res["select"] = ee.card([r for r in rows if r["split"] in ee.SELECT])
    res["objective"] = list(ee.objective(res["select"]))
    head = json.dumps({k: res[k] for k in ("variant", "policy", "objective")}, default=ee._jsonable)
    print(head, "\n  tune", res["tune"],
          "\n  dev ", res["dev"], flush=True)
    return res, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="base,rrf,bm25,entity,diag")
    args = ap.parse_args()
    only = set(args.only.split(","))
    items = [json.loads(x) for x in (ee.EVAL / "eval_set.jsonl").read_text(encoding="utf-8").splitlines() if x]
    items += [json.loads(x) for x in (ee.EVAL / "dev_set.jsonl").read_text(encoding="utf-8").splitlines() if x]
    items = [it for it in items if it["split"] in ee.SELECT]  # held-out items are not loaded at all
    corpus = ee.load()
    snippets = ee.evidence_snippets(corpus)
    ab = json.loads((ee.EVAL / "ablation.json").read_text(encoding="utf-8"))
    fc = dict(ab["final_config"])
    fc["rerankers"] = tuple(fc.get("rerankers") or ())
    base = ee.Config(**{k: v for k, v in fc.items() if k in ee.Config.__dataclass_fields__})
    models, cache, results = ee.Models(), {}, []
    runs = ee.run_config(base, corpus, items, models, cache)
    chunks = cache[(base.chunking, base.normalize, base.bm25_k1, base.bm25_b)][0]
    if "base" in only:
        results.append(scored("current shipped config, gate re-tuned on tune+dev", items, runs, chunks, snippets)[0])
    if "tuneonly" in only:
        sel, ee.SELECT = ee.SELECT, ("tune",)
        results.append(scored("current shipped config, gate tuned on tune only", items, runs, chunks, snippets)[0])
        ee.SELECT = sel
    if "nli" in only:
        for tau in (0.05, 0.2, 0.5):
            results.append(scored(f"+ NLI entailment filter (mDeBERTa, top 3, p >= {tau})", items,
                                  nli_filter(runs, chunks, tau), chunks, snippets)[0])
    if "entity" in only:
        results.append(scored("+ key name/number filter on ranked passages", items,
                              with_entity_filter(runs, chunks), chunks, snippets)[0])
    if "rrf" in only:
        for k in (10, 20, 30, 100):
            cfg = replace(base, rrf_k=k)
            r2 = ee.run_config(cfg, corpus, items, models, cache)
            results.append(scored(f"RRF k={k}", items, r2, chunks, snippets)[0])
    if "bm25" in only:
        for k1 in (0.9, 1.2, 1.5, 2.0):
            for b in (0.3, 0.75, 0.9):
                if (k1, b) == (1.5, 0.75):
                    continue
                cfg = replace(base, bm25_k1=k1, bm25_b=b)
                r2 = ee.run_config(cfg, corpus, items, models, cache)
                results.append(scored(f"BM25 k1={k1} b={b}", items, r2, chunks, snippets)[0])
    if "combo" in only:
        for k in (10, 20):
            cfg = replace(base, bm25_k1=1.2, rrf_k=k)
            r2 = ee.run_config(cfg, corpus, items, models, cache)
            results.append(scored(f"BM25 k1=1.2 b=0.75 + RRF k={k}", items, r2, chunks, snippets)[0])
    if "supo" in only:
        cfg = replace(base, bm25_k1=1.2, rrf_k=20, support_original=True)
        r2 = ee.run_config(cfg, corpus, items, models, cache)
        results.append(scored("BM25 k1=1.2 + RRF k=20 + support also on the original query", items, r2, chunks,
                              snippets)[0])
    if "diag" in only:
        pol = ee.tune(items, runs, chunks, snippets, True, True)
        rows = ee.grade(items, runs, chunks, pol, snippets)
        diag = []
        for it, r, row in zip(items, runs, rows):
            if it["lang"] == "pt" and it["type"] == "answerable" and not row["ok"]:
                diag.append({"id": it["id"], "q": it["question"], "q_en": r["q_en"],
                             "gold": it.get("refs") or it.get("evidence"),
                             "answered": row["answered"], "top": row["top"], "support": row["support"],
                             "ranked": [(chunks[i].ref or chunks[i].chunk_id, round(s, 3), round(sp, 2))
                                        for i, s, sp in r["ranked"]]})
        text = json.dumps(diag, indent=1, ensure_ascii=False, default=ee._jsonable)
        (ee.EVAL / "round2_pt_misses.json").write_text(text, encoding="utf-8")
        print(f"{len(diag)} PT misses written")
    models.save()
    if results:
        (ee.EVAL / "round2.json").write_text(json.dumps(results, indent=1, default=ee._jsonable), encoding="utf-8")


if __name__ == "__main__":
    main()
