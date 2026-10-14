"""Minimal text-only PDF writer for test fixtures (no third-party dependency)."""

from __future__ import annotations

import textwrap


def _escape(text: str) -> str:
    """Escape PDF string delimiters.

    Parameters
    ----------
    text : str
        Raw line.

    Returns
    -------
    str
        Line safe inside a PDF literal string.
    """
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def text_pdf(pages: list[str], width: int = 90) -> bytes:
    """Build a valid PDF with one page per string, Helvetica 11 pt, ASCII text.

    Parameters
    ----------
    pages : list[str]
        Page texts; long lines are wrapped.
    width : int
        Characters per line.

    Returns
    -------
    bytes
        PDF file content.
    """
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", ""]
    kids = []
    font = 3 + 2 * len(pages)
    for i, text in enumerate(pages):
        page_id, content_id = 3 + 2 * i, 4 + 2 * i
        kids.append(f"{page_id} 0 R")
        lines = textwrap.wrap(text, width) or [""]
        ops = " ".join(f"({_escape(line)}) Tj T*" for line in lines)
        stream = f"BT /F1 11 Tf 14 TL 50 750 Td {ops} ET"
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font} 0 R >> >> >>"
        )
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >>"
    objs.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out, offsets = b"%PDF-1.4\n", []
    for n, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out
