"""Internet Archive import with license evidence and explicit confirmation (FR-02)."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

import requests

from .ingest import Document, parse_bytes

METADATA_URL: Final[str] = "https://archive.org/metadata/{id}"
DOWNLOAD_URL: Final[str] = "https://archive.org/download/{id}/{file}"
_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9._-]{1,100}$")
LICENSE_FIELDS: Final[tuple[str, ...]] = ("licenseurl", "rights", "possible-copyright-status", "date", "publicdate")


@dataclass
class IAItem:
    """Metadata summary for one Internet Archive item.

    Parameters
    ----------
    identifier : str
        IA identifier.
    title : str
        Item title.
    license_fields : dict[str, str]
        License evidence fields (uploader supplied, not proof).
    file_name : str or None
        Chosen file, PDF preferred, else ``*_djvu.txt``.
    """

    identifier: str
    title: str
    license_fields: dict[str, str]
    file_name: str | None

    def __str__(self) -> str:
        """Return a short summary."""
        return f"IAItem({self.identifier}, file={self.file_name})"

    def evidence_text(self) -> str:
        """Format license evidence on one line.

        Returns
        -------
        str
            ``field=value`` pairs.
        """
        return "; ".join(f"{k}={v}" for k, v in self.license_fields.items())


def choose_file(files: list[dict[str, Any]]) -> str | None:
    """Pick the best text-bearing file.

    Parameters
    ----------
    files : list[dict]
        ``files`` list from the metadata API.

    Returns
    -------
    str or None
        PDF name (keeps page numbers) else a ``_djvu.txt`` name, else None.
    """
    names = [str(f.get("name", "")) for f in files]
    for n in names:
        if n.lower().endswith(".pdf"):
            return n
    for n in names:
        if n.lower().endswith("_djvu.txt"):
            return n
    return None


def fetch_metadata(identifier: str, session: requests.Session | None = None) -> IAItem:
    """Read item metadata.

    Parameters
    ----------
    identifier : str
        IA identifier.
    session : requests.Session or None
        HTTP session (injectable for tests).

    Returns
    -------
    IAItem
        Summary with license evidence.

    Raises
    ------
    ValueError
        If the identifier is malformed or the item does not exist.
    """
    if not _ID_RE.match(identifier):
        raise ValueError("Invalid Internet Archive identifier")
    http = session or requests.Session()
    resp = http.get(METADATA_URL.format(id=identifier), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not data or "metadata" not in data:
        raise ValueError(f"No item named {identifier}")
    meta = data["metadata"]
    fields = {k: str(meta[k]) for k in LICENSE_FIELDS if k in meta}
    return IAItem(identifier, str(meta.get("title", identifier)), fields, choose_file(data.get("files", [])))


def import_item(
    identifier: str,
    confirm: Callable[[IAItem], bool],
    session: requests.Session | None = None,
    max_mb: int = 50,
) -> Document | None:
    """Fetch metadata, ask for confirmation, then download and parse.

    Parameters
    ----------
    identifier : str
        IA identifier.
    confirm : Callable[[IAItem], bool]
        Shows license evidence to the user; returns True to proceed.
    session : requests.Session or None
        HTTP session.
    max_mb : int
        Download size cap.

    Returns
    -------
    Document or None
        Parsed document, or None when the user declines.

    Raises
    ------
    ValueError
        If no usable file exists.
    """
    http = session or requests.Session()
    item = fetch_metadata(identifier, http)
    if item.file_name is None:
        raise ValueError("Item has no PDF or _djvu.txt file")
    if not confirm(item):
        return None
    resp = http.get(DOWNLOAD_URL.format(id=identifier, file=item.file_name), timeout=120)
    resp.raise_for_status()
    name = item.file_name if item.file_name.lower().endswith(".pdf") else f"{identifier}.txt"
    doc = parse_bytes(name, resp.content, max_mb=max_mb)
    doc.title = item.title
    doc.source = f"archive.org/details/{identifier}"
    doc.license_evidence = item.evidence_text()
    return doc
