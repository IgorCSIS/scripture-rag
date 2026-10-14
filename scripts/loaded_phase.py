"""Grounding test: the same questions with and without the source text loaded.

The rule is grounding, not topic: a question about the Talmud or a Quran
commentary must abstain while only the Bible and the Koran are loaded, and must
answer with citations once a translation of that source is loaded. Questions
whose specific source is still absent (for example a tractate that is not in
the loaded volume) must keep abstaining.

Usage::

    python scripts/loaded_phase.py [--threshold T]

Writes ``docs/eval/loaded_phase_results.json`` and ``.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stress_test import EVAL_DIR, build, load_corpus, norm  # noqa: E402

from scripture_rag.config import Settings  # noqa: E402
from scripture_rag.pipeline import RagPipeline  # noqa: E402
from scripture_rag.validate import CitationValidator  # noqa: E402


def run(pipe: RagPipeline, items: list[dict[str, Any]], loaded: bool) -> list[dict[str, Any]]:
    """Ask every item under one condition and grade it.

    Parameters
    ----------
    pipe : RagPipeline
        Pipeline for this condition.
    items : list[dict]
        Phase items.
    loaded : bool
        Whether the extra sources are indexed.

    Returns
    -------
    list[dict]
        Graded rows.
    """
    validator = CitationValidator()
    rows = []
    for it in items:
        ans = pipe.ask(it["question"])
        cited = " ".join(norm(c.text) for c in ans.citations)
        expect = it["expect_loaded"] if loaded else "refuse"
        ev = it["evidence_loaded"]
        has_ev = any(e in cited for e in ev)
        right_book = any(c.book_id == it["book_loaded"] for c in ans.citations)
        valid = ans.refused or validator.validate(ans.text, set(pipe.chunks)).ok
        if expect == "answer":
            ok = not ans.refused and has_ev and right_book and valid
        elif expect == "refuse_or_correct":
            ok = ans.refused or (has_ev and right_book and valid)
        else:
            ok = ans.refused
        rows.append({"id": it["id"], "lang": it["lang"], "question": it["question"], "expect": expect,
                     "ok": bool(ok), "refused": ans.refused, "top_score": round(ans.top_score, 4),
                     "citations_valid": bool(valid), "evidence_found": has_ev,
                     "citations": [f"{c.book_id} p.{c.page_start} #{c.chunk_id}" for c in ans.citations],
                     "answer": ans.text[:400], "note": it.get("note", "")})
    return rows


def to_markdown(conds: dict[str, list[dict[str, Any]]], threshold: float, evidence: list[dict[str, Any]]) -> str:
    """Render both conditions side by side."""
    lines = ["# Grounding phase: source not loaded vs loaded", "",
             f"Same questions, same models, same threshold ({threshold}). No topic rules: the only "
             "difference between the two runs is which documents are indexed.", "", "## Added sources", ""]
    lines += [f"- `{e['book_id']}`: {e['url']} (possible-copyright-status: "
              f"{e.get('possible-copyright-status', 'n/a')}, date: {e.get('date', 'n/a')})" for e in evidence]
    lines += ["", "## Summary", "", "| Condition | Pass | Answered | Abstained |", "|---|---|---|---|"]
    for name, rows in conds.items():
        lines.append(f"| {name} | {sum(r['ok'] for r in rows)}/{len(rows)} | "
                     f"{sum(not r['refused'] for r in rows)} | {sum(r['refused'] for r in rows)} |")
    lines += ["", "## Per question", "", "| ID | Lang | Question | Not loaded | Loaded | Loaded citation |",
              "|---|---|---|---|---|---|"]
    a, b = conds["not_loaded"], conds["loaded"]
    for x, y in zip(a, b):
        def cell(r: dict[str, Any]) -> str:
            return ("abstain" if r["refused"] else "answer") + (" (pass)" if r["ok"] else " (FAIL)")
        lines.append(f"| {x['id']} | {x['lang']} | {x['question']} | {cell(x)} | {cell(y)} | "
                     f"{', '.join(y['citations'][:2]) or '-'} |")
    lines += ["", "## Loaded answers", ""]
    for y in b:
        if not y["refused"]:
            lines += [f"- **{y['id']}** {y['question']}", f"  - {y['answer'][:300]!r}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run both conditions."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=None, help="default: the stress test calibration")
    ap.add_argument("--max-mb", type=int, default=50)
    args = ap.parse_args()
    prev = json.loads((EVAL_DIR / "stress_results.json").read_text(encoding="utf-8"))
    t = args.threshold if args.threshold is not None else prev["calibration"]["threshold"]
    s = prev["settings"]
    settings = Settings(embed_model=s["embed_model"], rerank_model=s["rerank_model"], chunk_tokens=s["chunk_tokens"],
                        chunk_overlap=s["chunk_overlap"], refusal_threshold=t)
    items = [json.loads(x) for x in (EVAL_DIR / "loaded_phase_set.jsonl").read_text(encoding="utf-8").splitlines()
             if x]
    base, _ = load_corpus(args.max_mb)
    extra, evidence = load_corpus(args.max_mb, "corpus_extra.json", injection=False)
    conds = {"not_loaded": run(build(settings, base), items, False),
             "loaded": run(build(settings, base + extra), items, True)}
    out = {"threshold": t, "corpus_added": evidence, "conditions": conds}
    (EVAL_DIR / "loaded_phase_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), "utf-8")
    (EVAL_DIR / "loaded_phase_results.md").write_text(to_markdown(conds, t, evidence), "utf-8")
    for name, rows in conds.items():
        print(name, f"{sum(r['ok'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
