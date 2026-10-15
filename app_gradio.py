"""Local Gradio app: library panel plus Ask, Sources, Theme Map and Similar Passages tabs.

Run with ``python app_gradio.py`` (binds to 127.0.0.1 only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np

from scripture_rag.config import Settings
from scripture_rag.discover import book_tags, cluster_hits, library_centroid, rank_related, related_query
from scripture_rag.ia_import import fetch_metadata, import_item
from scripture_rag.ia_search import SORTS, TRADITIONS, Filters, IASearch
from scripture_rag.ingest import parse_bytes
from scripture_rag.library import ROLES, ask_by_role, compare, manifest, manifest_export, outliers, set_role
from scripture_rag.pipeline import Answer, RagPipeline
from scripture_rag.sanitize import UploadError, escape_for_display
from scripture_rag.similar import similar_passages
from scripture_rag.topics import TopicModeler, export_theme_map

PIPELINE = RagPipeline(Settings.from_env())
MAP_PATH = Path("site/theme_map.html")


def library_md() -> str:
    """Render the loaded book list.

    Returns
    -------
    str
        Markdown list of books with license evidence.
    """
    if not PIPELINE.documents:
        return "_No books loaded yet._"
    rows = [
        f"- **{escape_for_display(d.title)}** (`{d.book_id}`, {d.role}, {len(d.pages)} pages)"
        + (f"  \n  license: {escape_for_display(d.license_evidence)}" if d.license_evidence else "")
        for d in PIPELINE.documents.values()
    ]
    return "\n".join(rows)


def upload(files: list[str] | None) -> str:
    """Load uploaded files into the pipeline.

    Parameters
    ----------
    files : list[str] or None
        Temp paths from Gradio (random names, removed when the session ends).

    Returns
    -------
    str
        Status and updated library.
    """
    msgs = []
    s = PIPELINE.settings
    for f in files or []:
        p = Path(f)
        try:
            doc = parse_bytes(p.name, p.read_bytes(), s.max_upload_mb, s.max_pages, s.parse_timeout_s)
            msgs.append(f"Loaded {escape_for_display(p.name)}: {PIPELINE.add_document(doc)} passages")
        except UploadError as exc:
            msgs.append(f"Rejected {escape_for_display(p.name)}: {exc}")
    return "\n\n".join(msgs) + "\n\n" + library_md()


IA = IASearch()


def _hits_md(hits: list[Any], scores: list[float] | None = None) -> str:
    """Markdown table of search hits with license fields and preview links."""
    rows = ["| Title | Creator | Date | Copyright status | License | Preview |", "|---|---|---|---|---|---|"]
    for k, h in enumerate(hits):
        extra = f" ({scores[k]:.2f})" if scores else ""
        rows.append(f"| {escape_for_display(h.title[:80])}{extra} | {escape_for_display(h.creator[:40])} | {h.date} | "
                    f"{escape_for_display(h.copyright_status) or 'n/a'} | {escape_for_display(h.license_url) or 'n/a'} "
                    f"| [{h.identifier}]({h.preview_url}) |")
    return "\n".join(rows)


def ia_search(query: str, language: str, tradition: str, creator: str, collection: str, year_from: float | None,
              year_to: float | None, pdf_only: bool, public_domain: bool, sort: str, group: bool) -> tuple[str, Any]:
    """Search Internet Archive texts with filters; optionally cluster results into labeled groups.

    Returns
    -------
    tuple[str, gr.update]
        Markdown results and identifier choices.
    """
    f = Filters(language=None if language == "any" else language, tradition=None if tradition == "any" else tradition,
                creator=creator or "", collection=collection or "", year_from=int(year_from) if year_from else None,
                year_to=int(year_to) if year_to else None, pdf_only=pdf_only, public_domain=public_domain, sort=sort)
    try:
        hits = IA.search(query or "", rows=30, filters=f)
    except Exception as exc:  # network and validation errors are shown to the user
        return f"Error: {escape_for_display(str(exc))}", gr.update(choices=[])
    if not hits:
        return "No results.", gr.update(choices=[])
    if group and len(hits) >= 4:
        body = "\n\n".join(f"**{escape_for_display(g.label)}** ({len(g.hits)})\n\n{_hits_md(g.hits)}"
                            for g in cluster_hits(hits, PIPELINE.embedder))
    else:
        body = _hits_md(hits)
    return body, gr.update(choices=[h.identifier for h in hits], value=hits[0].identifier)


def ia_related() -> tuple[str, Any]:
    """Recommend Internet Archive books near the centroid of the loaded library."""
    ids, mat = PIPELINE.embeddings()
    if not ids:
        return "Load a book first.", gr.update()
    q = related_query([PIPELINE.chunks[i].text for i in ids[:2000]])
    try:
        hits = IA.search(q, rows=40, filters=Filters(public_domain=True))
    except Exception as exc:
        return f"Error: {escape_for_display(str(exc))}", gr.update()
    loaded = {d.source.rsplit("/", 1)[-1] for d in PIPELINE.documents.values() if d.source}
    ranked = rank_related(hits, PIPELINE.embedder, library_centroid(mat), loaded)
    if not ranked:
        return "No related books found.", gr.update()
    return (f"Related to your library (query: `{escape_for_display(q)}`)\n\n"
            + _hits_md([h for h, _ in ranked], [s for _, s in ranked]),
            gr.update(choices=[h.identifier for h, _ in ranked], value=ranked[0][0].identifier))


def set_book_role(book_id: str, role: str) -> str:
    """User override of a book's role tag."""
    if book_id in PIPELINE.documents:
        set_role(PIPELINE.documents[book_id], role)
    return library_md()


def compare_md(question: str, by: str) -> str:
    """Compare mode: one grounded answer per book or tradition, each with its own citations."""
    parts = []
    for group, ans in compare(PIPELINE, question, by).items():
        body = escape_for_display(ans.text) if not ans.refused else "_Not found in this group._"
        parts.append(f"### {escape_for_display(group)}\n\n{body}")
    return "\n\n".join(parts) or "Load books first."


def by_role_md(question: str) -> str:
    """Answers kept apart: primary texts first, commentary second."""
    out = []
    for role, ans in ask_by_role(PIPELINE, question).items():
        out.append(f"### {role.title()}\n\n" + (escape_for_display(ans.text) if not ans.refused else "_Not found._"))
    return "\n\n".join(out) or "Load books first."


def manifest_md() -> tuple[str, str]:
    """Library manifest table and a JSON export file path."""
    rows = manifest(PIPELINE)
    if not rows:
        return "_No books loaded._", None
    head = "| Book | Title | Role | Tradition | Date | Source | License | Passages |\n|---|---|---|---|---|---|---|---|"
    body = "\n".join(f"| {r['book_id']} | {escape_for_display(str(r['title']))} | {r['role']} | {r['tradition']} | "
                     f"{r['date']} | {escape_for_display(str(r['source_url']))} | "
                     f"{escape_for_display(str(r['license_status']))} | {r['passages']} |" for r in rows)
    path = MAP_PATH.with_name("library_manifest.json")
    path.write_text(manifest_export(PIPELINE), encoding="utf-8")
    return f"{head}\n{body}", str(path)


def book_choices() -> Any:
    """Refresh the book filter."""
    return gr.update(choices=list(PIPELINE.documents))


def ia_preview(identifier: str) -> str:
    """Show license evidence for an Internet Archive item before import.

    Parameters
    ----------
    identifier : str
        IA identifier.

    Returns
    -------
    str
        Markdown with license fields.
    """
    try:
        item = fetch_metadata(identifier.strip())
    except Exception as exc:  # network and validation errors are shown to the user
        return f"Error: {escape_for_display(str(exc))}"
    fields = "\n".join(f"- {k}: {escape_for_display(v)}" for k, v in item.license_fields.items()) or "- none listed"
    return (f"**{escape_for_display(item.title)}**, file: `{item.file_name}`\n\n{fields}\n\n"
            "These fields are uploader supplied evidence, not proof. Tick the box and import only if you accept.")


def ia_import(identifier: str, confirmed: bool) -> str:
    """Import an IA item after explicit confirmation.

    Parameters
    ----------
    identifier : str
        IA identifier.
    confirmed : bool
        User accepted the license evidence.

    Returns
    -------
    str
        Status and library.
    """
    try:
        doc = import_item(identifier.strip(), lambda _item: confirmed)
    except Exception as exc:
        return f"Error: {escape_for_display(str(exc))}"
    if doc is None:
        return "Import cancelled: confirm the license evidence first."
    return f"Imported {PIPELINE.add_document(doc)} passages.\n\n" + library_md()


def sources_md(ans: Answer) -> str:
    """Render cited passages with pages (FR-08).

    Parameters
    ----------
    ans : Answer
        Pipeline answer.

    Returns
    -------
    str
        Markdown with one collapsible block per citation.
    """
    if ans.refused:
        return f"No sources. Reason: {escape_for_display(ans.reason)} (top score {ans.top_score:.2f})"
    return "\n\n".join(
        f"<details><summary>{c.book_id} [{PIPELINE.documents[c.book_id].role}]"
        f"{', ' + c.ref if c.ref else ''}, PDF page {c.page_start}"
        f"{'' if c.page_end == c.page_start else '-' + str(c.page_end)}, #{c.chunk_id}</summary>\n\n"
        f"{escape_for_display(c.text)}\n\n</details>"
        for c in ans.citations
    )


def ask(question: str, books: list[str] | None = None) -> tuple[str, str]:
    """Answer a question.

    Parameters
    ----------
    question : str
        User question.

    Returns
    -------
    tuple[str, str]
        Escaped answer and sources markdown.
    """
    try:
        ans = PIPELINE.ask(question, set(books or []) or None)
    except Exception as exc:  # for example a model download failure
        return f"Error: {escape_for_display(str(exc))}", ""
    return escape_for_display(ans.text), sources_md(ans)


def theme_map(books: list[str] | None = None) -> str:
    """Cluster all passages and export the theme map.

    Returns
    -------
    str
        Markdown with topic keywords and the HTML path.
    """
    ids, mat = PIPELINE.embeddings(set(books or []) or None)
    if len(ids) < 3:
        return "Load at least a few passages first."
    texts = [PIPELINE.chunks[i].text for i in ids]
    res = TopicModeler().fit(mat, texts)
    owners = [PIPELINE.chunks[i].book_id for i in ids]
    export_theme_map(res, owners, texts, MAP_PATH)
    topics = "\n".join(f"- T{k}: {', '.join(v)}" for k, v in res.keywords.items())
    tags = "\n".join(f"- `{b}`: {', '.join(t)}" for b, t in book_tags(owners, res).items())
    topics += f"\n\n**Book tags**\n\n{tags}"
    flagged = outliers(mat, res.labels)[:10]
    if flagged:
        topics += "\n\n**Outliers** (far from their theme centroid)\n\n" + "\n".join(
            f"- z={o.z:.1f} {c}: {escape_for_display(c.text[:120])}"
            for o in flagged for c in [PIPELINE.chunks[ids[o.index]]])
    return (f"{len(res.keywords)} themes, silhouette {res.silhouette:.3f}. Map written to `{MAP_PATH}`.\n\n{topics}\n\n"
            "_Clusters are statistical similarity, not doctrinal claims._")


def similar(question: str, books: list[str] | None = None) -> str:
    """Find the best passage for a phrase, then its neighbors in other books.

    Parameters
    ----------
    question : str
        Phrase or question.

    Returns
    -------
    str
        Markdown list of neighbors.
    """
    hits = PIPELINE.retriever.retrieve(question, 1)
    if not hits:
        return "No matching passage."
    ids, mat = PIPELINE.embeddings(set(books or []) or None)
    anchor = PIPELINE.chunks[hits[0]]
    if anchor.chunk_id not in ids:
        row = PIPELINE.vectors.matrix[PIPELINE.vectors.ids.index(anchor.chunk_id)]
        ids, mat = [*ids, anchor.chunk_id], np.vstack([mat, row])
    res = similar_passages(anchor.chunk_id, ids, mat, PIPELINE.chunks)
    lines = [f"**Anchor** {anchor}: {escape_for_display(anchor.text[:200])}"]
    lines += [f"- {score:.2f} {c}: {escape_for_display(c.text[:200])}" for c, score in res]
    return "\n".join(lines)


def build() -> gr.Blocks:
    """Assemble the UI.

    Returns
    -------
    gr.Blocks
        Gradio app.
    """
    with gr.Blocks(title="Scripture RAG") as demo:
        gr.Markdown("# Scripture RAG\nAnswers only from your loaded texts, with citations.")
        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(file_count="multiple", file_types=[".pdf", ".txt"], type="filepath")
                lib = gr.Markdown(library_md())
                files.upload(upload, files, lib)
                with gr.Accordion("Search the Internet Archive", open=False):
                    ia_q = gr.Textbox(label="Search words")
                    with gr.Row():
                        ia_lang = gr.Dropdown(["any", "en", "pt", "es"], value="any", label="Language")
                        ia_trad = gr.Dropdown(["any", *TRADITIONS], value="any", label="Tradition / subject")
                        ia_sort = gr.Dropdown(list(SORTS), value="relevance", label="Sort")
                    with gr.Row():
                        ia_creator = gr.Textbox(label="Creator")
                        ia_coll = gr.Textbox(label="Collection")
                    with gr.Row():
                        ia_from = gr.Number(label="Year from", precision=0)
                        ia_to = gr.Number(label="Year to", precision=0)
                    with gr.Row():
                        ia_pdf = gr.Checkbox(value=True, label="PDF only")
                        ia_pd = gr.Checkbox(value=True, label="Public domain only")
                        ia_group = gr.Checkbox(value=True, label="Group results by topic")
                    ia_go = gr.Button("Search")
                    ia_rel = gr.Button("Related to my library")
                    ia_res = gr.Markdown()
                ia_id = gr.Dropdown(label="Internet Archive identifier", choices=[], allow_custom_value=True)
                ia_inputs = [ia_q, ia_lang, ia_trad, ia_creator, ia_coll, ia_from, ia_to,
                             ia_pdf, ia_pd, ia_sort, ia_group]
                ia_go.click(ia_search, ia_inputs, [ia_res, ia_id])
                ia_q.submit(ia_search, ia_inputs, [ia_res, ia_id])
                ia_rel.click(ia_related, None, [ia_res, ia_id])
                book_filter = gr.Dropdown(label="Search only these books (Ask, Theme Map, Similar)", multiselect=True,
                                          choices=list(PIPELINE.documents))
                lib.change(book_choices, None, book_filter)
                ia_info = gr.Markdown()
                gr.Button("Check license").click(ia_preview, ia_id, ia_info)
                ok = gr.Checkbox(label="I reviewed the license evidence")
                gr.Button("Import").click(ia_import, [ia_id, ok], lib)
            with gr.Column(scale=3), gr.Tabs():
                with gr.Tab("Ask"):
                    q = gr.Textbox(label="Question (EN, PT, ES)")
                    a = gr.Markdown()
                with gr.Tab("Sources"):
                    src = gr.Markdown()
                q.submit(ask, [q, book_filter], [a, src])
                with gr.Tab("Theme Map"):
                    tm = gr.Markdown()
                    gr.Button("Build theme map").click(theme_map, book_filter, tm)
                with gr.Tab("Compare"):
                    cq = gr.Textbox(label="Question")
                    cby = gr.Radio(["book", "tradition"], value="tradition", label="Group by")
                    co = gr.Markdown()
                    cq.submit(compare_md, [cq, cby], co)
                    gr.Button("Primary vs commentary").click(by_role_md, cq, co)
                with gr.Tab("Library manifest"):
                    mo = gr.Markdown()
                    mf = gr.File(label="manifest.json")
                    gr.Button("Refresh manifest").click(manifest_md, None, [mo, mf])
                    with gr.Row():
                        rb = gr.Dropdown(label="Book", choices=list(PIPELINE.documents))
                        rr = gr.Dropdown(list(ROLES), value="primary", label="Role")
                    gr.Button("Set role").click(set_book_role, [rb, rr], lib)
                    lib.change(book_choices, None, rb)
                with gr.Tab("Similar Passages"):
                    sq = gr.Textbox(label="Phrase")
                    so = gr.Markdown()
                    sq.submit(similar, [sq, book_filter], so)
    return demo


if __name__ == "__main__":
    build().launch(server_name="127.0.0.1")

