"""Rewrite measured numbers in README.md and docs/COMPLEXITY.md from the committed benchmark JSON.

Usage::

    python scripts/update_doc_numbers.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

B = Path("docs/benchmarks")
EV = Path("docs/eval")


def f2(x: float) -> str:
    return f"{x:.2f}"


def sub_row(text: str, label: str, cells: list[str], keep: int = 0, width: int | None = None,
            numeric_from: int | None = None) -> str:
    """Replace cells after the first `keep` of rows `| label | ... |` with `width` cells (default keep + len(cells)).

    With `numeric_from`, only rows whose cells from that index on are all numbers are touched.
    """
    pat = re.compile(r"^\| " + re.escape(label) + r" \|(.*)\|$", re.M)
    want = width if width is not None else keep + len(cells)

    def rep(m: re.Match[str]) -> str:
        old = [c.strip() for c in m.group(1).split("|")]
        if len(old) != want:
            return m.group(0)
        if numeric_from is not None and not all(re.fullmatch(r"-?[0-9.]+(?: \(.*\))?", c) for c in old[numeric_from:]):
            return m.group(0)
        return "| " + label + " | " + " | ".join(old[:keep] + cells) + " |"

    return pat.sub(rep, text)


def set_cell(text: str, label: str, idx: int, value: str, width: int) -> str:
    """Set cell `idx` (0 = first cell after the label) of rows `| label | ... |` that have `width` cells."""
    pat = re.compile(r"^\| " + re.escape(label) + r" \|(.*)\|$", re.M)

    def rep(m: re.Match[str]) -> str:
        old = [c.strip() for c in m.group(1).split("|")]
        if len(old) != width or not re.match(r"-?[0-9]", old[idx]):
            return m.group(0)
        old[idx] = value
        return "| " + label + " | " + " | ".join(old) + " |"

    return pat.sub(rep, text)


def main() -> None:
    res = json.loads((B / "results.json").read_text())["loglog_slopes"]
    lib = json.loads((B / "library_stages.json").read_text())
    ls = json.loads((B / "library_slopes.json").read_text())
    ann = json.loads((B / "browser_ann.json").read_text())
    abl = json.loads((EV / "ablation.json").read_text())
    final = next(s for s in abl["steps"] if s["config"] == abl["final_label"] and s["policy"] == abl["final_policy"])
    h = final["heldout"]
    core = {"Chunking": "chunking", "Brute-force search": "brute_force_search", "HNSW search": "hnsw_search",
            "BM25 query": "bm25_query", "k-means": "kmeans"}
    libk = {"Outliers (z-score)": "outliers", "Book auto-tags": "auto_tags", "Manifest CSV": "manifest_csv",
            "Role-scoped answer": "role_scoped_answer", "Compare by tradition": "compare_by_tradition"}
    iak = {"filterHits": "filterHits", "clusterHits": "clusterHits", "rankRelated": "rankRelated",
           "Book selector (selectBooks)": "book_selector"}
    pca = f2(res["pca_2d"])

    t = Path("README.md").read_text(encoding="utf-8")
    c = Path("docs/COMPLEXITY.md").read_text(encoding="utf-8")
    for lab, k in core.items():
        t = set_cell(t, lab, 1, f2(res[k]), 2)
        c = set_cell(c, lab, 1, f2(res[k]), 3)
    t = set_cell(t, "PCA to 2D", 1, f"{pca} (solver switching, see docs)", 2)
    c = set_cell(c, "PCA 2D", 1, pca, 3)
    for lab, k in {**libk, **iak}.items():
        t = set_cell(t, lab, 1, f2(ls[k]), 2)
    for lab, k in libk.items():
        vals = lib["stages"][k]
        t = sub_row(t, lab, [f2(v) for v in vals], keep=0, numeric_from=0)
        c = sub_row(c, lab, [f"{v:.3f}" for v in vals] + [f2(ls[k])], keep=1, numeric_from=1)
    for lab, k in iak.items():
        vals = lib["ia"]["stages"][k]
        t = sub_row(t, lab, [f"{v:.3f}" for v in vals] + [f2(ls[k])], keep=0, numeric_from=0)
        c = sub_row(c, lab, [f"{v:.3f}" for v in vals] + [f2(ls[k])], keep=1, numeric_from=1)
    for row in ann:
        t = sub_row(t, f"{row['n']:,}", [f2(row["brute_p50_ms"]), f2(row["hnsw_p50_ms"]), f2(row["hnsw_build_s"]),
                                         f2(row["recall_at_10"])], numeric_from=0)
    t = re.sub(r"^\| Answer accuracy \(en / pt / es\) \| [0-9.]+ \([0-9. /]+\); target 0.90, not met \|$",
               f"| Answer accuracy (en / pt / es) | {h['answer_accuracy']:.3f} ({h['acc_en']:.3f} / "
               f"{h['acc_pt']:.3f} / {h['acc_es']:.3f}); target 0.90, not met |", t, flags=re.M)
    wrong = h["unsupported_answers"] - h["adversarial_answered"]
    t = re.sub(r"^\| Refusal recall / precision \| [0-9.]+ / [0-9.]+; .*\|$",
               f"| Refusal recall / precision | {h['refusal_recall']:.3f} / {h['refusal_precision']:.3f}; "
               f"{h['adversarial_answered']} adversarial answered, {wrong} wrong-evidence answers; "
               f"injection pass {h['injection_pass']:.3f} |", t, flags=re.M)
    Path("README.md").write_text(t, encoding="utf-8")
    Path("docs/COMPLEXITY.md").write_text(c, encoding="utf-8")


if __name__ == "__main__":
    main()
