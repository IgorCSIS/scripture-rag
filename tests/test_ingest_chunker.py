"""FR-03 parsing and page-aware chunking tests."""

import pymupdf
import pytest

from scripture_rag.chunker import Chunker
from scripture_rag.ingest import Document, parse_bytes, split_txt_pages
from scripture_rag.sanitize import UploadError


def _pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_txt_pages_form_feed_and_virtual() -> None:
    assert split_txt_pages("a\fb\fc") == ["a", "b", "c"]
    assert len(split_txt_pages("x" * 6001)) == 3


def test_pdf_three_pages() -> None:
    doc = parse_bytes("three.pdf", _pdf(["alpha one", "beta two", "gamma three"]))
    assert len(doc.pages) == 3 and "beta" in doc.pages[1]


def test_pdf_page_cap() -> None:
    with pytest.raises(UploadError, match="limit"):
        parse_bytes("many.pdf", _pdf(["a", "b", "c"]), max_pages=2)


def test_txt_page_cap_and_bad_pdf() -> None:
    with pytest.raises(UploadError):
        parse_bytes("t.txt", b"a\fb\fc", max_pages=2)
    with pytest.raises(UploadError, match="open"):
        parse_bytes("broken.pdf", b"%PDF-garbage")


def test_three_page_fixture_page_ranges() -> None:
    pages = [" ".join(f"p{p}w{i}" for i in range(10)) for p in (1, 2, 3)]
    chunks = Chunker(target_tokens=8, overlap=2).split(Document("b", "B", pages))
    assert chunks[0].page_start == 1 and chunks[0].page_end == 1
    spanning = [c for c in chunks if c.page_start != c.page_end]
    assert spanning and all(c.page_end == c.page_start + 1 for c in spanning)
    assert chunks[-1].page_end == 3
    for c in chunks:
        first = c.text.split()[0]
        assert first.startswith(f"p{c.page_start}")
        assert c.text.split()[-1].startswith(f"p{c.page_end}")


def test_overlap() -> None:
    doc = Document("b", "B", [" ".join(f"w{i}" for i in range(30))])
    chunks = Chunker(target_tokens=10, overlap=3).split(doc)
    assert chunks[0].text.split()[-3:] == chunks[1].text.split()[:3]
    assert chunks[-1].text.split()[-1] == "w29"
    assert str(chunks[0]) == "[b p.1 #b-0000]"


def test_chunker_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError):
        Chunker(10, 10)


def test_empty_document() -> None:
    assert Chunker().split(Document("e", "E", [""])) == []
