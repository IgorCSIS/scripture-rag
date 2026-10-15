"""Upload validation and text sanitization (FR-01, security controls)."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Final

ALLOWED_EXTENSIONS: Final[frozenset[str]] = frozenset({".pdf", ".txt"})
PDF_MAGIC: Final[bytes] = b"%PDF-"
# Control characters except tab, newline, carriage return and form feed.
_CONTROL_RE: Final[re.Pattern[str]] = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f]")


class UploadError(ValueError):
    """Raised when an uploaded file fails validation."""


def validate_upload(filename: str, data: bytes, max_mb: int = 50) -> str:
    """Check extension, size and magic bytes of an upload.

    Parameters
    ----------
    filename : str
        Original file name supplied by the user.
    data : bytes
        Raw file content.
    max_mb : int
        Maximum allowed size in megabytes.

    Returns
    -------
    str
        ``"pdf"`` or ``"txt"``.

    Raises
    ------
    UploadError
        If the type is not allowed, the content does not match the extension,
        or the file is too large or empty.
    """
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadError(f"Unsupported file type {ext or '(none)'}; use .pdf or .txt")
    if len(data) > max_mb * 1024 * 1024:
        raise UploadError(f"File is larger than {max_mb} MB")
    if not data:
        raise UploadError("File is empty")
    if ext == ".pdf":
        if not data.startswith(PDF_MAGIC):
            raise UploadError("File has a .pdf extension but is not a PDF")
        return "pdf"
    if data.startswith(PDF_MAGIC):
        raise UploadError("File has a .txt extension but is a PDF")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UploadError("Text file is not valid UTF-8") from exc
    return "txt"


def strip_control_chars(text: str) -> str:
    """Remove control characters that could confuse rendering or prompts.

    Parameters
    ----------
    text : str
        Raw extracted text.

    Returns
    -------
    str
        Text without control characters (tabs, newlines and form feeds kept).
    """
    return _CONTROL_RE.sub("", text)


def escape_for_display(text: str) -> str:
    """HTML-escape document text before it is rendered in a UI.

    Parameters
    ----------
    text : str
        Untrusted text from a document.

    Returns
    -------
    str
        Escaped text safe to embed in HTML or Markdown.
    """
    return html.escape(text, quote=True)


def safe_book_id(name: str) -> str:
    """Derive a citation-safe book id from a file name.

    Parameters
    ----------
    name : str
        File name or title.

    Returns
    -------
    str
        Lowercase id containing only letters, digits, dots, dashes and underscores.
    """
    stem = Path(name).stem.lower()
    cleaned = re.sub(r"[^a-z0-9._-]+", "_", stem).strip("._-")
    return cleaned or "book"


# Leading role tags and instruction preambles that carry no content to retrieve on.
_ROLE_RE: Final[re.Pattern[str]] = re.compile(r"^\s*(?:system|assistant|user|developer|admin)\s*[:>\]-]+\s*", re.I)
_PREAMBLE_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:(?:developer|debug|admin|god|jailbreak|dan)\s+mode\s+(?:is\s+)?(?:enabled|on|activated)"
    r"|ignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above)\s+(?:instructions|rules|prompts?)"
    r"|you\s+are\s+now\s+[^.?!]{0,60}"
    r"|new\s+instructions?)\s*[.!:;,]+\s*",
    re.I,
)


def clean_query(question: str) -> str:
    """Strip injected role tags and instruction preambles from the front of a query.

    Retrieval should run on the content of the question, not on text such as
    "SYSTEM: developer mode enabled." that only tries to steer a model. The
    original question is returned unchanged when nothing would be left.
    O(|q|).
    """
    q = question
    for _ in range(4):
        new = _PREAMBLE_RE.sub("", _ROLE_RE.sub("", q))
        if new == q:
            break
        q = new
    return q.strip() or question.strip()
