"""Local Gradio app: library panel plus Ask, Sources, Theme Map and Similar Passages tabs.

Run with ``python app_gradio.py`` (binds to 127.0.0.1 only).
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from scripture_rag.config import Settings
from scripture_rag.ia_import import fetch_metadata, import_item
from scripture_rag.ingest import parse_bytes
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
        f"- **{escape_for_display(d.title)}** (`{d.book_id}`, {len(d.pages)} pages)"
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
        f"<details><summary>{c.book_id}, PDF page {c.page_start}"
        f"{'' if c.page_end == c.page_start else '-' + str(c.page_end)}, #{c.chunk_id}</summary>\n\n"
        f"{escape_for_display(c.text)}\n\n</details>"
        for c in ans.citations
    )


def ask(question: str) -> tuple[str, str]:
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
        ans = PIPELINE.ask(question)
    except Exception as exc:  # for example Ollama not running
        return f"Error: {escape_for_display(str(exc))}", ""
    return escape_for_display(ans.text), sources_md(ans)


def theme_map() -> str:
    """Cluster all passages and export the theme map.

    Returns
    -------
    str
        Markdown with topic keywords and the HTML path.
    """
    ids, mat = PIPELINE.embeddings()
    if len(ids) < 3:
        return "Load at least a few passages first."
    texts = [PIPELINE.chunks[i].text for i in ids]
    res = TopicModeler().fit(mat, texts)
    export_theme_map(res, [PIPELINE.chunks[i].book_id for i in ids], texts, MAP_PATH)
    topics = "\n".join(f"- T{k}: {', '.join(v)}" for k, v in res.keywords.items())
    return (f"{len(res.keywords)} themes, silhouette {res.silhouette:.3f}. Map written to `{MAP_PATH}`.\n\n{topics}\n\n"
            "_Clusters are statistical similarity, not doctrinal claims._")


def similar(question: str) -> str:
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
    ids, mat = PIPELINE.embeddings()
    anchor = PIPELINE.chunks[hits[0]]
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
                ia_id = gr.Textbox(label="Internet Archive identifier")
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
                q.submit(ask, q, [a, src])
                with gr.Tab("Theme Map"):
                    tm = gr.Markdown()
                    gr.Button("Build theme map").click(theme_map, None, tm)
                with gr.Tab("Similar Passages"):
                    sq = gr.Textbox(label="Phrase")
                    so = gr.Markdown()
                    sq.submit(similar, sq, so)
    return demo


if __name__ == "__main__":
    build().launch(server_name="127.0.0.1")

