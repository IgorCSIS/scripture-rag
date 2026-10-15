"""Ablation study on the 248 item eval set with a strict tune / held-out split.

Every stage is measured by switching it on in sequence (greedy ablation):
chunking, archaic normalization, query translation, query expansion, the
embedder, the hybrid weight, the reranker, and the answer gate (global vs
per-language thresholds, with and without the lexical support check).
Thresholds are always chosen on the tune split and reported on both splits.

Retrieval and reranking are computed once per configuration and cached on
disk; the gate is then evaluated offline, which makes threshold sweeps cheap.

Usage::

    python scripts/elite_eval.py            # full study
    python scripts/elite_eval.py --quick    # skip bge-m3 and bge-reranker

Writes ``docs/eval/ablation.json``, ``docs/eval/ABLATIONS.md`` and
``docs/benchmarks/ablation_*.png``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stress_test import load_corpus  # noqa: E402

from scripture_rag.bm25 import BM25Index  # noqa: E402
from scripture_rag.chunker import Chunk, Chunker  # noqa: E402
from scripture_rag.gate import IdfTable, support  # noqa: E402
from scripture_rag.ingest import Document  # noqa: E402
from scripture_rag.normalize import default_normalizer, expand_query  # noqa: E402
from scripture_rag.sanitize import clean_query  # noqa: E402
from scripture_rag.translate import detect_language  # noqa: E402
from scripture_rag.verses import BIBLE_BOOKS, ReferenceAligner, Verse, parse_numbered, parse_suras, verse_chunks  # noqa: E402

EVAL = Path("docs/eval")
BENCH = Path("docs/benchmarks")
CACHE = Path("data/cache/elite")
REF_URL = "https://www.gutenberg.org/cache/epub/10/pg10.txt"
REF_PATH = Path("data/ref/kjv_pg10.txt")
LANGS = ("en", "pt", "es")
REFUSE_TYPES = {"adversarial"}
# Splits used for every choice: the tune split plus the synthetic dev set (scripts/build_dev_set.py).
SELECT = ("tune", "dev")
SHOW_HELDOUT = False  # set by --final; held-out is scored once, on the final configuration


def norm(t: str) -> str:
    """Lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", t.lower()).strip()


class DiskCache:
    """Pickle-backed dict for expensive model outputs."""

    def __init__(self, name: str) -> None:
        CACHE.mkdir(parents=True, exist_ok=True)
        self.path = CACHE / f"{name}.pkl"
        self.d: dict[str, Any] = pickle.loads(self.path.read_bytes()) if self.path.exists() else {}
        self.dirty = 0

    def get(self, k: str) -> Any:
        return self.d.get(k)

    def put(self, k: str, v: Any) -> None:
        self.d[k] = v
        self.dirty += 1
        if self.dirty >= 2000:
            self.save()

    def save(self) -> None:
        if self.dirty:
            self.path.write_bytes(pickle.dumps(self.d))
            self.dirty = 0


def h(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:24]


# ---------------------------------------------------------------- corpus

def reference() -> list[Verse]:
    """Numbered KJV used to align the unnumbered 1611 facsimile (Genesis to Joshua)."""
    if not REF_PATH.exists():
        import requests

        REF_PATH.parent.mkdir(parents=True, exist_ok=True)
        REF_PATH.write_bytes(requests.get(REF_URL, timeout=120).content)
    return [v for v in parse_numbered(REF_PATH.read_text(encoding="utf-8")) if v.book in BIBLE_BOOKS[:6]]


@dataclass
class Corpus:
    """Chunk sets and evidence maps."""

    docs: list[Document]
    kjv_verses: list[Verse]
    koran_verses: list[Verse]

    def chunks(self, kind: str) -> list[Chunk]:
        out: list[Chunk] = []
        win = Chunker(200, 40)
        for d in self.docs:
            if kind == "verse" and d.book_id == "kjv_1611":
                out += verse_chunks(d.book_id, self.kjv_verses, 80)
            elif kind == "verse" and d.book_id == "koran_rodwell":
                out += verse_chunks(d.book_id, self.koran_verses, 80)
                # Rodwell's footnotes are dropped by the verse parser; keep them as plain windows.
                notes = Document("koran_rodwell_notes", d.title, d.pages)
                out += [replace(c, book_id="koran_rodwell", chunk_id=f"koran_rodwell-n{c.chunk_id[-4:]}")
                        for c in win.split(notes)]
            else:
                out += win.split(d)
        return out


def load() -> Corpus:
    docs, _ = load_corpus(50)
    kjv = next(d for d in docs if d.book_id == "kjv_1611")
    koran = next(d for d in docs if d.book_id == "koran_rodwell")
    key = h("align", *kjv.pages[:3], str(len(kjv.pages)))
    p = CACHE / f"align-{key}.pkl"
    CACHE.mkdir(parents=True, exist_ok=True)
    if p.exists():
        kv = pickle.loads(p.read_bytes())
    else:
        kv = ReferenceAligner(reference(), default_normalizer()).align(kjv)
        p.write_bytes(pickle.dumps(kv))
    return Corpus(docs, kv, parse_suras(koran))


def evidence_snippets(corpus: Corpus) -> dict[str, str]:
    """Gold ref -> first words of its 1611 text (or of the sentence that holds it)."""
    out: dict[str, str] = {}
    last = ""
    for v in corpus.kjv_verses:
        if v.text:
            last = " ".join(norm(v.text).split()[:7])
        if v.ref not in out and last:
            out[v.ref] = last
    return out


_RANGE_RE = re.compile(r"^(.*) (\d+):(\d+)(?:-(\d+))?$")


def ref_covers(chunk_ref: str, gold: str) -> bool:
    m, g = _RANGE_RE.match(chunk_ref or ""), _RANGE_RE.match(gold)
    if not m or not g or m.group(1) != g.group(1) or m.group(2) != g.group(2):
        return False
    lo, hi = int(m.group(3)), int(m.group(4) or m.group(3))
    return lo <= int(g.group(3)) <= hi


_GOLD: dict[str, set[str]] = {}
_WORD_RE = re.compile(r"[a-z]{4,}")


def gold_words(ref: str) -> set[str]:
    """Content words of the gold verse in the numbered reference KJV (modern spelling)."""
    if not _GOLD:
        for v in reference():
            _GOLD[v.ref] = set(_WORD_RE.findall(v.text.lower()))
    return _GOLD.get(ref, set())


def has_evidence(item: dict[str, Any], chunks: list[Chunk], snippets: dict[str, str]) -> bool:
    """True when the cited passages hold the gold answer.

    Accepted evidence: a gold phrase; a chunk whose verse range covers the gold ref; the first words
    of the aligned gold verse; or at least 60% of the gold verse's content words (from the numbered
    reference edition, compared after archaic normalization). The last rule makes grading robust to
    OCR noise and to alignment errors in the chunk's own label.
    """
    text = " ".join(norm(c.text) for c in chunks)
    if any(e in text for e in item.get("evidence") or []):
        return True
    normz = default_normalizer()
    for r in item.get("refs") or []:
        if any(ref_covers(c.ref, r) for c in chunks):
            return True
        s = snippets.get(r)
        if s and s in text:
            return True
        g = gold_words(r)
        if g:
            for c in chunks:
                words = set(_WORD_RE.findall(normz.text(c.text).lower()))
                if len(g & words) >= 0.6 * len(g):
                    return True
    return False


# ---------------------------------------------------------------- models

class Models:
    """Lazy model holders with disk caches."""

    def __init__(self) -> None:
        self.emb: dict[str, Any] = {}
        self.rr: dict[str, Any] = {}
        self.mt = None
        self.qcache = DiskCache("qvec")
        self.rcache = DiskCache("rerank")
        self.tcache = DiskCache("translate")

    def embedder(self, name: str) -> Any:
        if name not in self.emb:
            from scripture_rag.embed import SentenceTransformerEmbedder

            self.emb[name] = SentenceTransformerEmbedder(name)
        return self.emb[name]

    def passages(self, name: str, texts: list[str]) -> np.ndarray:
        p = CACHE / f"pvec-{h(name, *texts)}.npy"
        if p.exists():
            return np.load(p)
        t0 = time.perf_counter()
        m = self.embedder(name).encode(texts)
        print(f"  embedded {len(texts)} passages with {name} in {time.perf_counter() - t0:.0f} s", flush=True)
        np.save(p, m)
        return m

    def query(self, name: str, q: str) -> np.ndarray:
        k = h(name, q)
        v = self.qcache.get(k)
        if v is None:
            v = self.embedder(name).encode([q], is_query=True)[0]
            self.qcache.put(k, v)
        return v

    def translate(self, q: str) -> str:
        v = self.tcache.get(q)
        if v is None:
            if self.mt is None:
                from scripture_rag.translate import QueryTranslator

                self.mt = QueryTranslator()
            v = self.mt.to_english(q)
            self.tcache.put(q, v)
        return v

    def rerank(self, name: str, q: str, texts: list[str]) -> list[float]:
        keys = [h(name, q, t) for t in texts]
        miss = [i for i, k in enumerate(keys) if self.rcache.get(k) is None]
        if miss:
            if name not in self.rr:
                from scripture_rag.rerank import CrossEncoderReranker

                self.rr[name] = CrossEncoderReranker(name, window_words=60)
            r = self.rr[name]
            from scripture_rag.rerank import windows

            spans = [windows(texts[i], 60, 30) for i in miss]
            flat = r.score(q, [w for s in spans for w in s])
            pos = 0
            for i, s in zip(miss, spans):
                self.rcache.put(keys[i], max(flat[pos : pos + len(s)]))
                pos += len(s)
        return [self.rcache.get(k) for k in keys]

    def save(self) -> None:
        for c in (self.qcache, self.rcache, self.tcache):
            c.save()


# ---------------------------------------------------------------- retrieval

@dataclass(frozen=True)
class Config:
    """One retrieval configuration."""

    chunking: str = "window"
    normalize: bool = False
    translate: bool = False
    expand: bool = False
    embedder: str = "intfloat/multilingual-e5-small"
    dense_weight: float = 0.5
    reranker: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    candidates: int = 30
    bm25_original: bool = False
    rerankers: tuple[str, ...] = ()
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    rrf_k: int = 60
    support_original: bool = False

    def label(self) -> str:
        return (f"{self.chunking}|norm={int(self.normalize)}|mt={int(self.translate)}|exp={int(self.expand)}|"
                f"{self.embedder.split('/')[-1]}|w={self.dense_weight}|"
                + "+".join(r.split('/')[-1] for r in (self.rerankers or (self.reranker,)))
                + f"|k={self.candidates}|bmo={int(self.bm25_original)}"
                + ("" if (self.bm25_k1, self.bm25_b, self.rrf_k) == (1.5, 0.75, 60)
                   else f"|bm25={self.bm25_k1},{self.bm25_b}|rrf={self.rrf_k}")
                + ("|supo=1" if self.support_original else ""))


def wrrf(lists: list[tuple[list[int], float]], k: int = 60) -> list[int]:
    s: dict[int, float] = {}
    for ids, w in lists:
        for r, i in enumerate(ids, 1):
            s[i] = s.get(i, 0.0) + w / (k + r)
    return [i for i, _ in sorted(s.items(), key=lambda kv: -kv[1])]


def run_config(cfg: Config, corpus: Corpus, items: list[dict[str, Any]], models: Models,
               cache: dict[str, Any]) -> list[dict[str, Any]]:
    """Per item: ranked (chunk index, rerank score, support) and stage timings."""
    key = (cfg.chunking, cfg.normalize, cfg.bm25_k1, cfg.bm25_b)
    if key not in cache:
        chunks = corpus.chunks(cfg.chunking)
        normz = default_normalizer()
        # Every book is normalized, as in RagPipeline and the browser (site/js/rag.js).
        index_texts = [normz.text(c.text) if cfg.normalize else c.text for c in chunks]
        bm = BM25Index(cfg.bm25_k1, cfg.bm25_b)
        bm.add([str(i) for i in range(len(chunks))], index_texts)
        cache[key] = (chunks, index_texts, bm, IdfTable(index_texts))
    chunks, index_texts, bm, idf = cache[key]
    mkey = ((cfg.chunking, cfg.normalize), cfg.embedder)
    if mkey not in cache:
        cache[mkey] = models.passages(cfg.embedder, index_texts)
    mat = cache[mkey]
    out = []
    for it in items:
        q = clean_query(it["question"])
        lang = detect_language(q)
        t0 = time.perf_counter()
        q_en = models.translate(q) if cfg.translate and lang != "en" else q
        t1 = time.perf_counter()
        q_bm = expand_query(q_en) if cfg.expand else q_en
        lists: list[tuple[list[int], float]] = []
        bm_ids = [int(i) for i, _ in bm.search(q_bm, cfg.candidates)]
        lists.append((bm_ids, 1 - cfg.dense_weight))
        if cfg.bm25_original and q_en != q:
            lists.append(([int(i) for i, _ in bm.search(q, cfg.candidates)], 1 - cfg.dense_weight))
        for qq in dict.fromkeys([q, q_en]):
            sims = mat.astype(np.float64) @ models.query(cfg.embedder, qq).astype(np.float64)
            lists.append(([int(i) for i in np.argsort(-sims, kind="stable")[: cfg.candidates]], cfg.dense_weight))
        cand = wrrf(lists, cfg.rrf_k)[: cfg.candidates]
        t2 = time.perf_counter()
        texts = [chunks[i].text for i in cand]
        per = [models.rerank(m, q_en, texts) for m in (cfg.rerankers or (cfg.reranker,))]
        scores = [float(np.mean(v)) for v in zip(*per)]
        t3 = time.perf_counter()
        order = sorted(range(len(cand)), key=lambda j: -scores[j])[:5]
        def sup(i: int, q: str = q, q_en: str = q_en) -> float:
            v = support(q_en, index_texts[i], idf)
            # MT can leave names or whole questions untranslated; also credit words of the original.
            return max(v, support(q, index_texts[i], idf)) if cfg.support_original and q_en != q else v

        ranked = [(cand[j], float(scores[j]), sup(cand[j])) for j in order]
        out.append({"lang": lang, "q_en": q_en, "ranked": ranked, "cand_texts": texts,
                    "t_translate": t1 - t0, "t_retrieve": t2 - t1, "t_rerank": t3 - t2})
    models.save()
    return out


# ---------------------------------------------------------------- gate and grading

@dataclass(frozen=True)
class Policy:
    t: dict[str, float]
    s_min: float = 0.0
    top_n: int = 3


def grade(items: list[dict[str, Any]], runs: list[dict[str, Any]], chunks: list[Chunk], pol: Policy,
          snippets: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for it, r in zip(items, runs):
        t = pol.t.get(r["lang"], pol.t.get("en", 0.5))
        ok_ranked = [(i, s, sp) for i, s, sp in r["ranked"] if s >= t and sp >= pol.s_min][: pol.top_n]
        top = r["ranked"][0] if r["ranked"] else (None, 0.0, 0.0)
        answered = bool(r["ranked"]) and top[1] >= t and top[2] >= pol.s_min and bool(ok_ranked)
        ctx = [chunks[i] for i, _, _ in ok_ranked] if answered else []
        ev = has_evidence(it, ctx, snippets) if answered else False
        right_book = any(c.book_id == it["book"] for c in ctx)
        e = it["expect"]
        if e == "answer":
            ok = answered and ev and right_book
        elif e == "refuse":
            ok = not answered
        elif e == "refuse_or_correct":
            ok = (not answered) or ev
        elif e == "answer_no_obey":
            ok = answered and ev
        else:  # no_obey: extractive output can never follow injected instructions
            ok = True
        unsupported = answered and (it["type"] in REFUSE_TYPES or (it["type"] in {"answerable", "trick"} and not ev))
        rows.append({"id": it["id"], "type": it["type"], "lang": it["lang"], "split": it["split"], "ok": ok,
                     "answered": answered, "evidence": ev, "unsupported": unsupported, "top": round(top[1], 4),
                     "support": round(top[2], 3), "cited": [c.ref or f"{c.book_id} p.{c.page_start}" for c in ctx]})
    return rows


def card(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(sel: list[dict[str, Any]], key: str = "ok") -> float:
        return round(sum(r[key] for r in sel) / len(sel), 3) if sel else float("nan")

    ans = [r for r in rows if r["type"] == "answerable"]
    adv = [r for r in rows if r["type"] == "adversarial"]
    answered = [r for r in rows if r["answered"]]
    refused = [r for r in rows if not r["answered"]]
    return {
        "n": len(rows),
        "answer_accuracy": rate(ans),
        **{f"acc_{lg}": rate([r for r in ans if r["lang"] == lg]) for lg in LANGS},
        "refusal_recall": rate([dict(r, ok=not r["answered"]) for r in adv]),
        "refusal_precision": (round(sum(r["type"] != "answerable" for r in refused) / len(refused), 3)
                              if refused else 1.0),
        "adversarial_answered": sum(r["answered"] for r in adv),
        "unsupported_answers": sum(r["unsupported"] for r in rows),
        "trick_pass": rate([r for r in rows if r["type"] == "trick"]),
        "injection_pass": rate([r for r in rows if r["type"] == "injection"]),
        "answered": len(answered),
    }


def objective(c: dict[str, Any]) -> tuple:
    """No answers to unanswerable questions first; then accuracy (plus macro over languages) with each
    wrong-evidence answer costing 0.02; then refusal recall."""
    macro = np.nanmean([c[f"acc_{lg}"] for lg in LANGS])
    wrong = c["unsupported_answers"] - c["adversarial_answered"]
    return (-c["adversarial_answered"], round(c["answer_accuracy"] + macro - 0.02 * wrong, 4), c["refusal_recall"])


GRID = [round(x, 2) for x in np.arange(0.02, 0.99, 0.02)]
SUP_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def tune(items, runs, chunks, snippets, per_lang: bool, use_support: bool) -> Policy:
    """Pick thresholds on the tune split only."""
    tix = [k for k, it in enumerate(items) if it["split"] in SELECT]
    ti, tr = [items[k] for k in tix], [runs[k] for k in tix]
    best: tuple | None = None
    for s in SUP_GRID if use_support else [0.0]:
        if per_lang:
            t: dict[str, float] = {}
            for lg in LANGS:
                sel = [k for k, r in enumerate(tr) if r["lang"] == lg]
                si, sr = [ti[k] for k in sel], [tr[k] for k in sel]
                cand = max(GRID, key=lambda x: (objective(card(grade(si, sr, chunks, Policy({lg: x}, s),
                                                                     snippets))), x))
                t[lg] = cand
            pol = Policy(t, s)
        else:
            x = max(GRID, key=lambda x: (objective(card(grade(ti, tr, chunks, Policy({lg: x for lg in LANGS}, s),
                                                               snippets))), x))
            pol = Policy({lg: x for lg in LANGS}, s)
        score = objective(card(grade(ti, tr, chunks, pol, snippets)))
        if best is None or score > best[0]:
            best = (score, pol)
    assert best is not None
    return best[1]


def evaluate(name: str, cfg: Config, corpus, items, models, cache, snippets, per_lang=False, use_support=False):
    t0 = time.perf_counter()
    runs = run_config(cfg, corpus, items, models, cache)
    chunks = cache[(cfg.chunking, cfg.normalize, cfg.bm25_k1, cfg.bm25_b)][0]
    pol = tune(items, runs, chunks, snippets, per_lang, use_support)
    rows = grade(items, runs, chunks, pol, snippets)
    res = {"step": name, "config": cfg.label(), "policy": {"t": pol.t, "s_min": pol.s_min},
           "tune": card([r for r in rows if r["split"] == "tune"]),
           "dev": card([r for r in rows if r["split"] == "dev"]),
           "select": card([r for r in rows if r["split"] in SELECT]),
           "heldout": card([r for r in rows if r["split"] == "heldout"]) if SHOW_HELDOUT else None,
           "stage_ms": {k: round(1000 * float(np.median([r[k] for r in runs])), 2)
                        for k in ("t_translate", "t_retrieve", "t_rerank")},
           "wall_s": round(time.perf_counter() - t0, 1)}
    print(json.dumps({k: res[k] for k in ("step", "config", "policy")}), flush=True)
    print("   tune   ", res["tune"], "\n   dev    ", res["dev"], flush=True)
    if SHOW_HELDOUT:
        print("   heldout", res["heldout"], flush=True)
    return res, rows, runs


def _jsonable(o: Any) -> Any:
    """numpy scalars to Python for json.dumps."""
    return o.item() if hasattr(o, "item") else str(o)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--final", action="store_true", help="also score the held-out split (do this once)")
    ap.add_argument("--no-dev", action="store_true", help="select on the tune split only")
    args = ap.parse_args()
    global SHOW_HELDOUT, SELECT
    SHOW_HELDOUT = args.final
    if args.no_dev:
        SELECT = ("tune",)
    items_all = [json.loads(x) for x in (EVAL / "eval_set.jsonl").read_text(encoding="utf-8").splitlines() if x]
    if "dev" in SELECT:
        items_all += [json.loads(x) for x in (EVAL / "dev_set.jsonl").read_text(encoding="utf-8").splitlines() if x]
    corpus = load()
    snippets = evidence_snippets(corpus)
    full = " ".join(norm(" ".join(d.pages)) for d in corpus.docs)
    items, dropped = [], []
    for it in items_all:
        gold_ok = True
        if it.get("refs") and it["expect"] in {"answer", "answer_no_obey"}:
            gold_ok = all(r in snippets for r in it["refs"])
        if it.get("evidence") and it["expect"] == "answer" and not it.get("refs"):
            gold_ok = any(e in full for e in it["evidence"])
        (items if gold_ok else dropped).append(it)
    print(f"{len(items)} items kept, {len(dropped)} dropped (gold not verifiable in the PDF text): "
          f"{[d['id'] for d in dropped]}", flush=True)
    lang_acc = float(np.mean([detect_language(it["question"]) == it["lang"] for it in items]))
    models, cache, steps = Models(), {}, []
    cfg = Config()
    plan = [
        ("A0 baseline (window 200, raw, e5-small, mmarco, global threshold)", {}),
        ("A1 + verse-aware chunker", {"chunking": "verse"}),
        ("A2 + archaic normalization", {"normalize": True}),
        ("A3 + PT/ES query translation", {"translate": True}),
        ("A4 + archaic query expansion", {"expand": True}),
    ]
    best_rows = None
    for name, change in plan:
        cfg = replace(cfg, **change)
        res, rows, _ = evaluate(name, cfg, corpus, items, models, cache, snippets)
        steps.append(res)
    # Embedder comparison on the current pipeline.
    emb_res = {}
    embedders = ["intfloat/multilingual-e5-small", "intfloat/multilingual-e5-base"]
    for emb in embedders + ([] if args.quick else ["BAAI/bge-m3"]):
        res, _, _ = evaluate(f"A5 embedder {emb.split('/')[-1]}", replace(cfg, embedder=emb), corpus, items, models,
                             cache, snippets)
        emb_res[emb] = res
        steps.append(res)
    best_emb = max(emb_res, key=lambda e: objective(emb_res[e]["select"]))
    cfg = replace(cfg, embedder=best_emb)
    w_res = {}
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        res, _, _ = evaluate(f"A6 dense weight {w}", replace(cfg, dense_weight=w), corpus, items, models, cache,
                             snippets)
        w_res[w] = res
        steps.append(res)
    cfg = replace(cfg, dense_weight=max(w_res, key=lambda w: objective(w_res[w]["select"])))
    rr_res = {}
    for rr in ["cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"] + ([] if args.quick else ["BAAI/bge-reranker-v2-m3"]):
        res, _, _ = evaluate(f"A7 reranker {rr.split('/')[-1]}", replace(cfg, reranker=rr), corpus, items, models,
                             cache, snippets)
        rr_res[rr] = res
        steps.append(res)
    cfg = replace(cfg, reranker=max(rr_res, key=lambda r: objective(rr_res[r]["select"])))
    res, _, _ = evaluate("A8 per-language thresholds", cfg, corpus, items, models, cache, snippets, per_lang=True)
    steps.append(res)
    res, best_rows, _runs = evaluate("A9 per-language thresholds + lexical support gate", cfg, corpus, items, models,
                                    cache, snippets, per_lang=True, use_support=True)
    steps.append(res)
    # Later stages: each is kept only if it wins on the tune split.
    later = [
        ("A10 + 50 reranker candidates", {"candidates": 50}, True),
        ("A11 + BM25 on the original PT/ES query", {"bm25_original": True}, True),
        ("A12 reranker ensemble mMiniLM + ms-marco-MiniLM-L-6 (English only; not shipped, see SCORECARD)",
         {"rerankers": (cfg.reranker, "cross-encoder/ms-marco-MiniLM-L-6-v2")}, False),
        ("A13 + BM25 k1=1.2 (grid in round2.json)", {"bm25_k1": 1.2}, True),
        ("A14 + RRF k=20 (grid in round2.json)", {"rrf_k": 20}, True),
    ]
    for name, change, may_select in later:
        r2, rows2, runs2 = evaluate(name, replace(cfg, **change), corpus, items, models, cache, snippets,
                                    per_lang=True, use_support=True)
        steps.append(r2)
        if may_select and objective(r2["select"]) >= objective(res["select"]):
            cfg, res, best_rows, _runs = replace(cfg, **change), r2, rows2, runs2
    out = {"items": len(items), "dropped": [d["id"] for d in dropped], "language_detection_accuracy": lang_acc,
           "final_config": cfg.__dict__,
           "final_label": cfg.label(), "final_policy": res["policy"], "steps": steps, "final_rows": best_rows}
    text = json.dumps(out, indent=2, ensure_ascii=False, default=_jsonable)
    (EVAL / "ablation.json").write_text(text, encoding="utf-8")
    print("FINAL", cfg.label(), res["policy"], flush=True)


if __name__ == "__main__":
    main()
