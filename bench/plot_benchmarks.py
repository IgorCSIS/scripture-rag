"""Re-plot every benchmark PNG from the committed JSON files (no re-timing).

Inputs: docs/benchmarks/results.json, library_stages.json, browser_ann.json.
Outputs: PNGs in docs/benchmarks/ and site/assets/benchmarks/, plus library_slopes.json.
Run from the repo root: python bench/plot_benchmarks.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

B = Path("docs/benchmarks")
OUT = [B, Path("site/assets/benchmarks")]
DIM = 384
COLORS = ["#3b5bdb", "#e8590c", "#0ca678", "#ae3ec9", "#f59f00", "#1098ad", "#495057"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                     "axes.titleweight": "bold", "figure.facecolor": "white", "axes.facecolor": "white"})

TITLES = {"chunking": "Chunking", "brute_force_search": "Brute-force search", "hnsw_search": "HNSW search",
          "bm25_query": "BM25 query", "kmeans": "k-means", "pca_2d": "PCA to 2D"}
XLAB = {"chunking": "n (characters, scaled with chunk count)"}
PRED = {"chunking": ("O(N)", lambda n: n), "brute_force_search": ("O(n d)", lambda n: n),
        "hnsw_search": ("O(d log n)", np.log2), "bm25_query": ("O(postings), about O(n)", lambda n: n),
        "kmeans": ("O(n K i d)", lambda n: n), "pca_2d": ("O(n d^2)", lambda n: n)}


def plain(v, _pos=None):
    """Plain tick label: 1,000 instead of 10^3, 0.5 instead of 5x10^-1."""
    if v >= 1:
        return f"{v:,.0f}"
    return f"{v:.4f}".rstrip("0").rstrip(".")


def loglog_axes(ax, xs):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.autoscale_view()
    ax.set_xticks(xs)
    ax.xaxis.set_major_formatter(FuncFormatter(plain))
    ax.xaxis.set_minor_formatter(NullFormatter())
    lo, hi = ax.get_ylim()
    subs = (1.0, 2.0, 5.0) if np.log10(hi / lo) <= 3.5 else (1.0,)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=subs))
    ax.yaxis.set_major_formatter(FuncFormatter(plain))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", alpha=0.3)


def save(fig, name):
    fig.tight_layout()
    for d in OUT:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.png", dpi=120)
    plt.close(fig)


def complexity():
    r = json.loads((B / "results.json").read_text())
    n = np.asarray(r["sizes"], float)
    for name, secs in r["seconds"].items():
        t = np.asarray(secs) * 1000
        label, f = PRED[name]
        ref = f(n)
        c = float(np.dot(ref, t) / np.dot(ref, ref))
        slope = r["loglog_slopes"][name]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, log in zip(axes, (False, True)):
            ax.plot(n, t, "o-", color=COLORS[0], label="measured median")
            ax.plot(n, c * ref, "--", color=COLORS[1], label=f"fit c * {label}")
            ax.set_xlabel(XLAB.get(name, "n (chunks)"))
            ax.set_ylabel("median latency (ms)")
            if log:
                loglog_axes(ax, n)
                ax.tick_params(axis="x", labelrotation=30)
                ax.set_title(f"{TITLES[name]}, log-log, slope {slope:.2f}")
            else:
                ax.xaxis.set_major_formatter(FuncFormatter(plain))
                ax.grid(True, alpha=0.3)
                ax.set_title(f"{TITLES[name]}, median latency vs n")
            ax.legend(loc="upper left", fontsize=9)
        save(fig, name)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, secs) in enumerate(r["seconds"].items()):
        ax.plot(n, np.asarray(secs) * 1000, "o-", color=COLORS[i], label=f"{TITLES[name]} (slope {r['loglog_slopes'][name]:.2f})")
    loglog_axes(ax, n)
    ax.set_xlabel("n (chunks; characters for chunking)")
    ax.set_ylabel("median latency (ms)")
    ax.set_title(f"Python runtime vs n, log-log (d = {DIM})")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9, frameon=False)
    save(fig, "overview")


def library():
    d = json.loads((B / "library_stages.json").read_text())
    slopes = {}
    for key, sub, xl, title in [
        ("library_stages", d, "n (passages, 8 books)", "Browser library stages, log-log"),
        ("ia_stages", d["ia"], "r (search hits; B = r books for the selector)", "Internet Archive panel stages, log-log"),
    ]:
        n = np.asarray(sub["sizes"], float)
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for i, (k, v) in enumerate(sub["stages"].items()):
            v = np.maximum(np.asarray(v, float), 1e-4)
            s = float(np.polyfit(np.log(n), np.log(v), 1)[0])
            slopes[k] = round(s, 2)
            ax.plot(n, v, "o-", color=COLORS[i], label=f"{k} (slope {s:.2f})")
        loglog_axes(ax, n)
        ax.set_xlabel(xl)
        ax.set_ylabel("median latency (ms)")
        ax.set_title(title)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9, frameon=False)
        save(fig, key)
    (B / "library_slopes.json").write_text(json.dumps(slopes, indent=2))


def browser_ann():
    rows = json.loads((B / "browser_ann.json").read_text())
    n = np.asarray([r["n"] for r in rows], float)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(n, [r["brute_p50_ms"] for r in rows], "o-", color=COLORS[1], label="brute force p50")
    ax.plot(n, [r["hnsw_p50_ms"] for r in rows], "o-", color=COLORS[0], label="HNSW (ann.js) p50")
    for r in rows:
        ax.annotate(f"recall@10 {r['recall_at_10']:.2f}", (r["n"], r["hnsw_p50_ms"]), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=8, color=COLORS[0])
    ax.set_yscale("log")
    ax.set_ylim(0.1, 100)
    loglog_axes(ax, n)
    ax.set_xlabel(f"n (passages, d = {DIM} unit vectors)")
    ax.set_ylabel("p50 latency (ms)")
    ax.set_title("Browser vector search, brute force vs HNSW, log-log")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(B / "browser_ann.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    complexity()
    library()
    browser_ann()
