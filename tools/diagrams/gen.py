"""Generate all scripture-rag SVG diagrams in one style."""
from pathlib import Path
from xml.sax.saxutils import escape as esc

REPO = Path(__file__).resolve().parents[2]
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
W = 960
PAL = {
    "input": ("#e7f5ff", "#1c7ed6"), "query": ("#f3f0ff", "#7048e8"), "retrieve": ("#e6fcf5", "#0ca678"),
    "rank": ("#fff4e6", "#e8590c"), "gate": ("#fff9db", "#e67700"), "out": ("#ebfbee", "#2f9e44"),
    "refuse": ("#fff5f5", "#e03131"), "store": ("#f1f3f5", "#495057"), "ui": ("#e7f5ff", "#1971c2"),
}
TXT, SUB = "#1f2328", "#4a525c"


def tw(s, fs, bold=False):
    return len(s) * fs * (0.6 if bold else 0.54)


class D:
    def __init__(self, title, subtitle, h):
        self.title, self.subtitle, self.h, self.els, self.boxes = title, subtitle, h, [], []

    def node(self, x, y, w, h, kind, title, lines=()):
        f, s = PAL[kind]
        self.boxes.append((x, y, w, h, title, lines))
        self.els.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{f}" stroke="{s}" stroke-width="1.6"/>')
        self.els.append(f'<rect x="{x}" y="{y+10}" width="4" height="{h-20}" rx="2" fill="{s}"/>')
        n = 1 + len(lines)
        top = y + h / 2 - (n - 1) * 10 + 5
        self.els.append(f'<text x="{x+w/2}" y="{top}" text-anchor="middle" font-size="15" font-weight="650" fill="{TXT}">{esc(title)}</text>')
        for i, l in enumerate(lines):
            self.els.append(f'<text x="{x+w/2}" y="{top+20*(i+1)}" text-anchor="middle" font-size="13" fill="{SUB}">{esc(l)}</text>')
        return (x, y, w, h)

    def arrow(self, pts, label=None, lpos=None, color="#57606a", dash=False, anchor="middle"):
        d = "M" + " L".join(f"{x},{y}" for x, y in pts)
        da = ' stroke-dasharray="5 4"' if dash else ""
        self.els.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.8"{da} marker-end="url(#ah)"/>')
        if label:
            lx, ly = lpos
            w = tw(label, 13) + 12
            ax = {"middle": lx - w / 2, "start": lx - 6, "end": lx - w + 6}[anchor]
            self.els.append(f'<rect x="{ax}" y="{ly-14}" width="{w}" height="20" rx="6" fill="#ffffff" stroke="#d0d7de"/>')
            self.els.append(f'<text x="{lx}" y="{ly+1}" text-anchor="{anchor}" font-size="13" fill="{TXT}">{esc(label)}</text>')

    def lane(self, x, y, w, h, label):
        self.els.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="#f6f8fa" stroke="#d0d7de" stroke-dasharray="6 4"/>')
        self.els.append(f'<text x="{x+16}" y="{y+24}" font-size="14" font-weight="700" fill="{SUB}" letter-spacing="0.4">{esc(label.upper())}</text>')

    def text(self, x, y, s, fs=13, fill=SUB, anchor="start", weight="400"):
        self.els.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{fs}" font-weight="{weight}" fill="{fill}">{esc(s)}</text>')

    def svg(self):
        for (x, y, w, h, t, lines) in self.boxes:
            need = max([tw(t, 15, True)] + [tw(l, 13) for l in lines]) + 20
            print(f"OVERFLOW {self.title}: {t!r} {lines} needs {need:.0f} > {w}") if need > w else None
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {self.h}" width="{W}" height="{self.h}" '
                f'role="img" aria-labelledby="t d" font-family="{FONT}">'
                f'<title id="t">{esc(self.title)}</title><desc id="d">{esc(self.subtitle)}</desc>'
                '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                '<path d="M0,0 L10,5 L0,10 z" fill="#57606a"/></marker></defs>'
                f'<rect x="1" y="1" width="{W-2}" height="{self.h-2}" rx="18" fill="#ffffff" stroke="#d0d7de" stroke-width="1.5"/>'
                f'<text x="32" y="42" font-size="20" font-weight="700" fill="{TXT}">{esc(self.title)}</text>'
                f'<text x="32" y="66" font-size="14" fill="{SUB}">{esc(self.subtitle)}</text>')
        return head + "".join(self.els) + "</svg>\n"


def row(d, y, h, specs, x0=32, x1=W - 32, gap=36, labels=None):
    n = len(specs)
    w = (x1 - x0 - gap * (n - 1)) / n
    out = []
    for i, (kind, t, lines) in enumerate(specs):
        x = x0 + i * (w + gap)
        out.append(d.node(round(x), y, round(w), h, kind, t, lines))
        if i:
            px, py, pw, ph = out[i - 1]
            lab = labels[i - 1] if labels else None
            d.arrow([(px + pw, y + h / 2), (round(x) - 2, y + h / 2)])
    return out


def wrap(d, a, b, label=None):
    """Connector from the end of one row to the start of the next."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    mid = (ay + ah + by) / 2
    d.arrow([(ax + aw / 2, ay + ah), (ax + aw / 2, mid), (bx + bw / 2, mid), (bx + bw / 2, by - 2)])
    if label:
        d.text((ax + aw / 2 + bx + bw / 2) / 2, mid - 6, label, anchor="middle")


def save(d, path):
    p = REPO / path
    p.write_text(d.svg(), encoding="utf-8")
    print("wrote", p)


# ---------------------------------------------------------------- query flow (shared by retrieval + data_flow)
def query_flow(d, y0, with_title=True):
    H = 76
    r1 = row(d, y0, H, [("query", "Question", ["English or Spanish"]), ("query", "clean_query", ["tidy the raw text"]),
                        ("query", "Detect language", ["en / es; other: notice"]), ("query", "Translate to EN", ["MarianMT, ES only"])])
    y1 = y0 + H + 56
    r2 = row(d, y1, H, [("query", "Normalize + expand", ["archaic forms, synonyms"]), ("retrieve", "BM25", ["translated query", "+ original query"]),
                        ("retrieve", "Dense search", ["multilingual-e5-small"]), ("rank", "Weighted RRF", ["fuse both rankings"])])
    wrap(d, r1[3], r2[0])
    y2 = y1 + H + 56
    r3 = row(d, y2, H, [("rank", "Rerank top 50", ["cross-encoder", "mmarco-mMiniLMv2"]),
                        ("gate", "Language gate", ["score >= t[lang]"]), ("gate", "Support gate", ["lexical support >= 0.3"]),
                        ("out", "Cited verse quotes", ["top 3, [book p.N #id]"])])
    wrap(d, r2[3], r3[0])
    for i, g in ((1, r3[1]), (2, r3[2])):
        pass
    # yes labels on gate arrows
    for a in (r3[1], r3[2]):
        x, y, w, h = a
        d.arrow([(x + w, y + h / 2), (x + w + 34, y + h / 2)], label="yes", lpos=(x + w + 17, y + h / 2 - 12))
    y3 = y2 + H + 60
    gx = r3[1][0]
    gw = r3[2][0] + r3[2][2] - gx
    ref = d.node(gx, y3, gw, 64, "refuse", "Refuse", ["\u201cI could not find this in the loaded texts.\u201d"])
    for a in (r3[1], r3[2]):
        x, y, w, h = a
        d.arrow([(x + w / 2, y + h), (x + w / 2, y3 - 2)], label="no", lpos=(x + w / 2 + 22, y + h + 30))
    return y3 + 64


def retrieval():
    d = D("Answering a question", "Extractive only: every answer is a quoted passage with a citation, or a refusal.", 560)
    query_flow(d, 96)
    save(d, "site/assets/retrieval.svg")


def pipeline():
    d = D("Ingest pipeline", "Each book is checked, parsed page by page, normalized, chunked by verse and indexed.", 372)
    H = 76
    r1 = row(d, 96, H, [("input", "Upload checks", ["magic bytes, size", "and page caps"]), ("input", "Parse pages", ["pypdf / pdf.js", "with a timeout"]),
                        ("input", "Strip control chars", ["escape all output"])])
    r2 = row(d, 96 + H + 56, H, [("retrieve", "Normalize archaic", ["thee, saith, -eth ..."]), ("retrieve", "Verse chunks", ["about 80 words,", "1 verse overlap"]),
                                 ("store", "Embed + index", ["BM25 + e5 vectors,", "HNSW above 20k"])])
    wrap(d, r1[2], r2[0])
    d.text(W / 2, 350, "No verse numbers found: 200-token windows with 40-token overlap instead.", anchor="middle")
    save(d, "site/assets/pipeline.svg")


def data_flow():
    d = D("Data flow", "Ingest path (top) builds the indexes; query path (bottom) reads them.", 800)
    H = 76
    d.lane(20, 84, W - 40, 150, "Ingest")
    row(d, 124, H, [("input", "Sanitize + parse", ["PDF / TXT or IA item,", "text by page"]),
                    ("retrieve", "Normalize", ["archaic spelling"]), ("retrieve", "Verse chunk", ["80 words, 1 verse", "overlap"]),
                    ("store", "Indexes", ["BM25 + vectors", "IndexedDB"])], x0=40, x1=W - 40, gap=30)
    d.lane(20, 250, W - 40, 530, "Query")
    query_flow(d, 290)
    save(d, "docs/diagrams/data_flow.svg")


def state():
    d = D("Query lifecycle", "States of one question, from Idle to a saved answer or refusal.", 500)
    H = 70
    y1, y2, y3 = 110, 250, 400
    r = row(d, y1, H, [("ui", "Idle", ["waiting"]), ("query", "Preparing", ["clean, detect,", "translate"]), ("retrieve", "Retrieving", ["BM25 + e5, RRF"]),
                      ("rank", "Reranking", ["top 50 to 5"]), ("gate", "Gating", ["score >= t[lang]"])], gap=34)
    X = [b[0] for b in r]; w = r[0][2]
    cx = [x + w / 2 for x in X]
    d.text((X[0] + w + X[1]) / 2, y1 + H / 2 - 10, "ask", anchor="middle", fill=TXT, fs=13)
    books = d.node(X[0], y2, w, H, "store", "Books changed", ["re-index"])
    saved = d.node(X[1], y2, w, H, "store", "Saved", ["IndexedDB history"])
    ref = d.node(X[3], y2, w, H, "refuse", "Refused", ["no answer"])
    sup = d.node(X[4], y2, w, H, "gate", "Support check", ["support >= 0.3"])
    ans = d.node(X[4], y3, w, H, "out", "Answered", ["quote + validate"])
    d.arrow([(cx[0], y2), (cx[0], y1 + H + 2)], label="done", lpos=(cx[0] + 34, y1 + H + 36))
    d.arrow([(X[1] + 20, y2), (X[0] + w - 20, y1 + H + 2)], label="next", lpos=(X[1] + 4, y1 + H + 40))
    d.arrow([(cx[4], y1 + H), (cx[4], y2 - 2)], label="score ok", lpos=(cx[4] - 46, y1 + H + 36))
    d.arrow([(X[4] + 24, y1 + H), (X[3] + w - 24, y2 - 2)], label="score low", lpos=(X[3] + w - 30, y1 + H + 36))
    d.arrow([(X[4], y2 + H / 2), (X[3] + w + 2, y2 + H / 2)], label="low", lpos=(X[4] - 17, y2 + H / 2 - 14))
    d.arrow([(cx[4], y2 + H), (cx[4], y3 - 2)], label="supported", lpos=(cx[4] - 52, y2 + H + 40))
    d.arrow([(X[3], y2 + H / 2), (X[1] + w + 2, y2 + H / 2)])
    d.arrow([(X[4], y3 + H / 2), (cx[1], y3 + H / 2), (cx[1], y2 + H + 2)])
    return d


def architecture_docs():
    d = D("Architecture", "Two implementations of one pipeline. Browser and Python make identical decisions (parity tests).", 600)
    lx, lw = 20, W - 40
    d.lane(lx, 84, lw, 230, "Browser app (site/js), runs offline")
    H = 76
    b = row(d, 124, H, [("ui", "UI", ["Ask, Compare, Library", "Theme map, History"]), ("input", "Ingest", ["pdf.js, sanitize,", "verse chunker"]),
                        ("rank", "rag.js pipeline", ["translate, hybrid,", "rerank, gates"]), ("store", "IndexedDB", ["books, chunks,", "history, meta"])], x0=40, x1=W - 40, gap=30)
    b2 = row(d, 124 + H + 20, H, [("input", "IA panel", ["filters, license,", "clusters, related"]), ("query", "models.js", ["transformers.js ONNX:", "e5, MarianMT, mMiniLM"]),
                                   ("retrieve", "Indexes", ["BM25, brute force;", "HNSW above 20k"]), ("retrieve", "Theme map", ["k-means, PCA,", "outliers, TF-IDF"])], x0=40, x1=W - 40, gap=30)
    d.lane(lx, 360, lw, 220, "Python package (src/scripture_rag)")
    p = row(d, 400, H, [("ui", "Apps", ["Gradio, Streamlit,", "CLI demo"]), ("input", "Ingest", ["pypdf, sanitize,", "verses, normalize"]),
                        ("rank", "RagPipeline", ["translate, hybrid,", "rerank, gate"]), ("out", "Extractive answer", ["quotes + citation", "validator"])], x0=40, x1=W - 40, gap=30)
    p2 = row(d, 400 + H + 20, H, [("input", "Internet Archive", ["ia_search, ia_import"]), ("query", "Models", ["sentence-transformers,", "MarianMT"]),
                                   ("retrieve", "Indexes", ["BM25, NumPy, FAISS"]), ("store", "Eval harness", ["held-out scorecard,", "parity export"])], x0=40, x1=W - 40, gap=30)
    for i, j in ((0, 0), (1, 1), (2, 2), (3, 3)):
        pass
    d.arrow([(W / 2, 316), (W / 2, 358)], dash=True)
    d.arrow([(W / 2 + 20, 358), (W / 2 + 20, 316)], dash=True, label="parity: same chunks, decisions, citations", lpos=(W / 2 + 40, 342), anchor="start")
    return d


def architecture_site():
    d = D("System overview", "Everything runs on your device. Nothing is generated: answers are quoted passages.", 480)
    H = 80
    r1 = row(d, 96, H, [("input", "Your books", ["PDF / TXT or", "Internet Archive"]), ("input", "Ingest", ["checks, pages,", "verse chunks"]),
                        ("store", "Indexes", ["BM25 + e5 vectors", "in IndexedDB"])])
    r2 = row(d, 96 + H + 60, H, [("query", "Question", ["cleaned, detected,", "translated to EN"]), ("rank", "Hybrid + rerank", ["RRF, top 50,", "cross-encoder"]),
                                  ("gate", "Two gates", ["score >= t[lang],", "support >= 0.3"])])
    r2b = r2
    d.arrow([(r1[2][0] + r1[2][2] / 2, 96 + H), (r1[2][0] + r1[2][2] / 2, 96 + H + 30), (r2[1][0] + r2[1][2] / 2, 96 + H + 30), (r2[1][0] + r2[1][2] / 2, r2[1][1] - 2)])
    y3 = r2[0][1] + H + 50
    gx, gy, gw, gh = r2[2]
    ans = d.node(r2[1][0], y3, r2[1][2], 64, "out", "Cited quotes", ["[book p.N #id]"])
    ref = d.node(gx, y3, gw, 64, "refuse", "Refusal", ["not in loaded texts"])
    d.arrow([(gx + gw / 2, gy + gh), (gx + gw / 2, y3 - 2)], label="no", lpos=(gx + gw / 2 + 20, gy + gh + 28))
    d.arrow([(gx + 10, gy + gh), (ans[0] + ans[2] - 10, y3 - 2)], label="yes", lpos=(gx - 14, gy + gh + 28))
    return d


def clustering():
    d = D("Unsupervised theme map", "Clusters are statistical similarity, not doctrinal claims.", 330)
    H = 76
    r1 = row(d, 96, H, [("store", "Passage vectors", ["e5 embeddings"]), ("retrieve", "k-means", ["K by silhouette"]),
                        ("retrieve", "PCA to 2D", ["map layout"]), ("out", "TF-IDF labels", ["class-based keywords"])])
    r2 = row(d, 96 + H + 56, H, [("gate", "Outlier flags", ["centroid z-score > 2.5"]), ("ui", "Theme map", ["re-clusters on change"]),
                                 ("query", "Similar passages", ["nearest, other books"])])
    wrap(d, r1[3], r2[0])
    return d


if __name__ == "__main__":
    retrieval(); pipeline(); data_flow()
    save(state(), "docs/diagrams/state.svg")
    save(architecture_docs(), "docs/diagrams/architecture.svg")
    save(architecture_site(), "site/assets/architecture.svg")
    save(clustering(), "site/assets/clustering.svg")
