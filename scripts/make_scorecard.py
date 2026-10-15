"""Render docs/eval/SCORECARD.md from ablation.json (current) and ablation_prev.json (previous final).

Usage::

    python scripts/make_scorecard.py
"""

from __future__ import annotations

import json
from pathlib import Path

EVAL = Path("docs/eval")
COLS = [("n", "n"), ("answer_accuracy", "Answer acc"), ("acc_en", "EN"), ("acc_pt", "PT"), ("acc_es", "ES"),
        ("refusal_recall", "Refusal recall"), ("refusal_precision", "Refusal precision"),
        ("adversarial_answered", "Adversarial answered"), ("unsupported_answers", "Unsupported answers"),
        ("injection_pass", "Injection pass")]


def fmt(v: object) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def table(rows: list[tuple[str, dict]]) -> list[str]:
    out = ["| Run | " + " | ".join(c[1] for c in COLS) + " |", "|---" * (len(COLS) + 1) + "|"]
    out += [f"| {name} | " + " | ".join(fmt(card[c[0]]) for c in COLS) + " |" for name, card in rows]
    return out


def main() -> None:
    cur = json.loads((EVAL / "ablation.json").read_text(encoding="utf-8"))
    prev = json.loads((EVAL / "ablation_prev.json").read_text(encoding="utf-8"))
    final = next(s for s in cur["steps"] if s["config"] == cur["final_label"] and s["policy"] == cur["final_policy"])
    before = next((st for st in prev["steps"] if st["config"] == prev["final_label"]
                   and st["policy"] == prev["final_policy"]), prev["steps"][-1])
    b, a = before["heldout"], final["heldout"]
    lines = [
        "# Scorecard",
        "",
        f"Eval set [eval_set.jsonl](eval_set.jsonl), strict split: every choice is made on the **tune** split "
        f"({final['tune']['n']} items) plus the synthetic **dev** set [dev_set.jsonl](dev_set.jsonl) "
        f"({final['dev']['n'] if final.get('dev') else 0} items, scripts/build_dev_set.py) and reported once on the "
        f"**held-out** split ({a['n']} items). "
        f"Dropped (gold not in the PDF OCR): {', '.join(cur['dropped'])}. "
        f"Language detection accuracy: {cur['language_detection_accuracy']:.3f}.",
        "",
        "Reproduce: `python scripts/elite_eval.py --quick --final` then `python scripts/make_scorecard.py`. "
        "Raw data: [ablation.json](ablation.json); previous run: [ablation_prev.json](ablation_prev.json).",
        "",
        "Unsupported answers = answers to adversarial items plus answers whose cited passages do not hold the "
        "gold answer (wrong evidence). Citations are extractive quotes, so citation validity is 1.0 by construction.",
        "",
        "## Held-out: previous final vs current final",
        "",
        "| Metric | Previous | Current | Target |",
        "|---|---|---|---|",
    ]
    for k, label, tgt in [("answer_accuracy", "Answer accuracy", ">= 0.90"), ("acc_en", "EN", ">= 0.85"),
                          ("acc_pt", "PT", ">= 0.85"), ("acc_es", "ES", ">= 0.85"),
                          ("refusal_recall", "Refusal recall", ""), ("refusal_precision", "Refusal precision", ""),
                          ("adversarial_answered", "Adversarial answered", "0"),
                          ("unsupported_answers", "Unsupported answers (incl. adversarial)", "0"),
                          ("injection_pass", "Injection pass", "1.0")]:
        lines.append(f"| {label} | {fmt(b[k])} | {fmt(a[k])} | {tgt} |")
    lines += ["", f"Previous: `{before['config']}`, policy `{json.dumps(before['policy'])}`.",
              f"Current: `{final['config']}`, policy `{json.dumps(final['policy'])}`.", "",
              "## All steps, tune split (selection data)", ""]
    lines += table([(s["step"], s["tune"]) for s in cur["steps"]])
    if all(st.get("dev") for st in cur["steps"]):
        lines += ["", "## All steps, synthetic dev set (selection data)", ""]
        lines += table([(st["step"], st["dev"]) for st in cur["steps"]])
    lines += ["", "## All steps, held-out split (reported, never used for selection)", ""]
    lines += table([(s["step"], s["heldout"]) for s in cur["steps"]])
    notes = EVAL / "scorecard_notes.md"
    lines += ["", notes.read_text(encoding="utf-8").rstrip()] if notes.exists() else []
    lines += [""]
    (EVAL / "SCORECARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
