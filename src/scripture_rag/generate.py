"""Extractive answer generator: quotes retrieved passages with citations (no LLM)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .chunker import Chunk
from .config import REFUSAL_TEXT
from .validate import split_sentences


def citation(chunk: Chunk) -> str:
    """Format the citation tag for a chunk.

    Parameters
    ----------
    chunk : Chunk
        Source chunk.

    Returns
    -------
    str
        ``[book_id p.N #chunk_id]``.
    """
    return f"[{chunk.book_id} p.{chunk.page_start} #{chunk.chunk_id}]"


class Generator(ABC):
    """Interface: turn a question and context into a cited answer."""

    @abstractmethod
    def answer(self, question: str, context: list[Chunk]) -> str:
        """Produce an answer.

        Parameters
        ----------
        question : str
            User question.
        context : list[Chunk]
            Reranked passages.

        Returns
        -------
        str
            Answer with citations, or the refusal text.
        """

    def __str__(self) -> str:
        """Return the class name."""
        return type(self).__name__


class ExtractiveGenerator(Generator):
    """Return the top passages verbatim, each followed by its citation.

    Parameters
    ----------
    max_passages : int
        Passages to quote.
    """

    def __init__(self, max_passages: int = 3) -> None:
        """Create the generator."""
        self._max = max_passages

    def answer(self, question: str, context: list[Chunk]) -> str:
        """Quote passages (see ``Generator.answer``)."""
        if not context:
            return REFUSAL_TEXT
        lines = []
        for c in context[: self._max]:
            tag = citation(c)
            sentences = []
            for sent in split_sentences(c.text.replace("\n", " ")):
                if sent[-1] not in ".!?":
                    sent += "."
                sentences.append(f"{sent} {tag}")
            # Every quoted sentence carries the citation of its passage.
            lines.append(" ".join(sentences))
        return "\n".join(lines)
