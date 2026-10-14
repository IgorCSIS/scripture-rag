"""Page-aware sliding window chunker (FR-03)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from .ingest import Document

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage.

    Parameters
    ----------
    chunk_id : str
        Unique id, ``{book_id}-{index:04d}``.
    book_id : str
        Owning book.
    text : str
        Passage text.
    page_start : int
        First page (1-based PDF page index) the chunk touches.
    page_end : int
        Last page the chunk touches.
    char_start : int
        Start offset within the page-joined document text.
    char_end : int
        End offset (exclusive).
    ref : str
        Verse reference such as ``Exodus 21:23-25``; empty for plain windows.
    """

    chunk_id: str
    book_id: str
    text: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    ref: str = ""

    def __str__(self) -> str:
        """Return the citation label."""
        return f"[{self.book_id} p.{self.page_start} #{self.chunk_id}]"

    @property
    def label(self) -> str:
        """Human citation: ``Exodus 21:23-25, p.61`` or ``p.61``."""
        return f"{self.ref}, p.{self.page_start}" if self.ref else f"p.{self.page_start}"


class Chunker:
    """Split documents into overlapping token windows that remember pages.

    Tokens are whitespace-delimited words, a simple and language-neutral
    approximation of model tokens.

    Parameters
    ----------
    target_tokens : int
        Tokens per chunk.
    overlap : int
        Tokens shared between consecutive chunks.
    """

    def __init__(self, target_tokens: int = 400, overlap: int = 60) -> None:
        """Create a chunker.

        Raises
        ------
        ValueError
            If overlap is not smaller than the target size.
        """
        if not 0 <= overlap < target_tokens:
            raise ValueError("overlap must be in [0, target_tokens)")
        self._target = target_tokens
        self._overlap = overlap

    def __str__(self) -> str:
        """Return a short summary."""
        return f"Chunker(target={self._target}, overlap={self._overlap})"

    def split(self, doc: Document) -> list[Chunk]:
        """Chunk a document.

        Parameters
        ----------
        doc : Document
            Parsed document.

        Returns
        -------
        list[Chunk]
            Chunks in reading order.
        """
        # One linear pass collects (start, end, page) for every token.
        tokens: list[tuple[int, int, int]] = []
        joined_parts: list[str] = []
        offset = 0
        for page_no, page in enumerate(doc.pages, start=1):
            for m in _TOKEN_RE.finditer(page):
                tokens.append((offset + m.start(), offset + m.end(), page_no))
            joined_parts.append(page)
            offset += len(page) + 1  # one separator char between pages
        joined = "\n".join(joined_parts)
        chunks: list[Chunk] = []
        step = self._target - self._overlap
        for idx, start in enumerate(range(0, len(tokens), step)):
            window = tokens[start : start + self._target]
            c0, c1 = window[0][0], window[-1][1]
            text = " ".join(joined[c0:c1].split())
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.book_id}-{idx:04d}",
                    book_id=doc.book_id,
                    text=text,
                    page_start=window[0][2],
                    page_end=window[-1][2],
                    char_start=c0,
                    char_end=c1,
                )
            )
            if start + self._target >= len(tokens):
                break
        return chunks
