"""Hosted demo in extractive mode: cited passages only, no LLM.

Run with ``streamlit run app_streamlit.py``.
"""

from __future__ import annotations

import streamlit as st

from scripture_rag.config import Settings
from scripture_rag.ingest import parse_bytes
from scripture_rag.pipeline import RagPipeline
from scripture_rag.sanitize import UploadError


@st.cache_resource
def get_pipeline() -> RagPipeline:
    """Create one pipeline per session in extractive mode.

    Returns
    -------
    RagPipeline
        Pipeline without an LLM.
    """
    base = Settings.from_env()
    return RagPipeline(Settings(mode="extractive", embed_model=base.embed_model, rerank_model=base.rerank_model,
                                refusal_threshold=base.refusal_threshold))


def main() -> None:
    """Render the page."""
    st.set_page_config(page_title="Scripture RAG demo")
    st.title("Scripture RAG")
    st.caption("Extractive demo: shows cited passages only. Uploads stay in memory for this session.")
    pipe = get_pipeline()
    for f in st.file_uploader("PDF or TXT", type=["pdf", "txt"], accept_multiple_files=True) or []:
        if f.name in st.session_state.setdefault("loaded", set()):
            continue
        try:
            n = pipe.add_document(parse_bytes(f.name, f.getvalue()))
            st.session_state["loaded"].add(f.name)
            st.success(f"Loaded {f.name}: {n} passages")
        except UploadError as exc:
            st.error(f"Rejected {f.name}: {exc}")
    q = st.text_input("Question")
    if q:
        ans = pipe.ask(q)
        if ans.refused:
            st.warning(ans.text)
        for c in ans.citations:
            # st.text never renders HTML or Markdown, so document text is displayed literally.
            with st.expander(f"{c.book_id}, PDF page {c.page_start}, #{c.chunk_id}"):
                st.text(c.text)


if __name__ == "__main__":
    main()
