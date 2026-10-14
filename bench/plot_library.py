"""Plot docs/benchmarks/library_stages.json into library_stages.png and ia_stages.png; print slopes."""
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

d = json.load(open("docs/benchmarks/library_stages.json"))
slopes = {}
for key, sub, xl, title in [("library_stages", d, "passages n", "Browser library stages, 8 books"),
                            ("ia_stages", d["ia"], "hits r (books B for the selector)", "Internet Archive panel stages")]:
    n = np.array(sub["sizes"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, v in sub["stages"].items():
        v = np.maximum(np.array(v), 1e-4)
        s = float(np.polyfit(np.log(n), np.log(v), 1)[0])
        slopes[k] = round(s, 2)
        ax.loglog(n, v, "o-", label=f"{k} (slope {s:.2f})")
    ax.set_xlabel(xl)
    ax.set_ylabel("median ms")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"docs/benchmarks/{key}.png", dpi=120)
json.dump(slopes, open("docs/benchmarks/library_slopes.json", "w"), indent=2)
print(json.dumps(slopes))
