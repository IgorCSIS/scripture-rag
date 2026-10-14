"""Parse PDF and TXT files into page-level documents."""

from __future__ import annotations

import concurrent.futures as cf
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from .sanitize import UploadError, safe_book_id, strip_control_chars, validate_upload

VIRTUAL_PAGE_CHARS: Final[int] = 3000


@dataclass
class Document:
    """A loaded book split into pages.

    Parameters
    ----------
    book_id : str
        Citation-safe identifier.
    title : str
        Display title.
    pages : list[str]
        Page texts; index 0 is page 1.
    source : str
        Where the text came from (file name or IA identifier).
    license_evidence : str
        Free text license evidence, if any.
    """

    book_id: str
    title: str
    pages: list[str] = field(default_factory=list)
    source: str = ""
    license_evidence: str = ""

    def __str__(self) -> str:
        """Return a short summary."""
        return f"Document({self.book_id}, {len(self.pages)} pages)"


def split_txt_pages(text: str) -> list[str]:
    """Split plain text into pages.

    Parameters
    ----------
    text : str
        Full text.

    Returns
    -------
    list[str]
        Pages split on form feed if present, otherwise virtual pages of
        ``VIRTUAL_PAGE_CHARS`` characters.
    """
    if "\f" in text:
        return [p for p in text.split("\f")]
    return [text[i : i + VIRTUAL_PAGE_CHARS] for i in range(0, max(len(text), 1), VIRTUAL_PAGE_CHARS)]


def _pdf_pages(data: bytes, max_pages: int) -> list[str]:
    """Extract per-page text from PDF bytes with PyMuPDF.

    Parameters
    ----------
    data : bytes
        PDF content.
    max_pages : int
        Page cap.

    Returns
    -------
    list[str]
        Text of each page.

    Raises
    ------
    UploadError
        If the PDF has too many pages or cannot be opened.
    """
    import pymupdf  # imported lazily so TXT-only use needs no PDF library

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several error types
        raise UploadError("Could not open PDF") from exc
    with doc:
        if doc.page_count > max_pages:
            raise UploadError(f"PDF has {doc.page_count} pages; limit is {max_pages}")
        # get_text only extracts text; links, scripts and annotations are ignored.
        return [page.get_text("text") for page in doc]


def parse_bytes(
    filename: str,
    data: bytes,
    max_mb: int = 50,
    max_pages: int = 2000,
    timeout_s: float = 60.0,
) -> Document:
    """Validate and parse an uploaded file.

    Parameters
    ----------
    filename : str
        Original file name.
    data : bytes
        File content.
    max_mb : int
        Size cap in megabytes.
    max_pages : int
        Page cap.
    timeout_s : float
        Parse time limit in seconds.

    Returns
    -------
    Document
        Parsed, sanitized document.

    Raises
    ------
    UploadError
        On validation failure, page cap or timeout.
    """
    kind = validate_upload(filename, data, max_mb)
    if kind == "pdf":
        pool = cf.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(_pdf_pages, data, max_pages)
        try:
            pages = future.result(timeout=timeout_s)
        except cf.TimeoutError as exc:
            raise UploadError(f"Parsing took longer than {timeout_s:.0f} s") from exc
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
    else:
        pages = split_txt_pages(data.decode("utf-8"))
        if len(pages) > max_pages:
            raise UploadError(f"Text has {len(pages)} pages; limit is {max_pages}")
    pages = [strip_control_chars(p) for p in pages]
    return Document(book_id=safe_book_id(filename), title=Path(filename).stem, pages=pages, source=filename)


def load_path(path: str | Path, **limits: float) -> Document:
    """Load a PDF or TXT file from disk.

    Parameters
    ----------
    path : str or Path
        File path.
    **limits : float
        Optional ``max_mb``, ``max_pages``, ``timeout_s`` overrides.

    Returns
    -------
    Document
        Parsed document.
    """
    p = Path(path)
    return parse_bytes(p.name, p.read_bytes(), **limits)  # type: ignore[arg-type]
