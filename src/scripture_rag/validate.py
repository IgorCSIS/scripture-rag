"""Citation validation with a regular expression (FR-06).

The citation language ``[book_id p.N #chunk_id]`` is regular, so a single
compiled pattern (a finite automaton) decides membership.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Final

CITATION_RE: Final[re.Pattern[str]] = re.compile(r"\[([a-z0-9._-]+) p\.(\d+) #([a-z0-9._-]+)\]")
_SENTENCE_RE: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?\]])\s+(?=[A-Z0-9\"'(\u00bf\u00a1\u00c0-\u00dd])")


@dataclass
class ValidationResult:
    """Outcome of citation checking.

    Parameters
    ----------
    ok : bool
        True when every sentence is cited and every id is known.
    cited_ids : list[str]
        Chunk ids cited, in order of first appearance.
    errors : list[str]
        Human readable problems.
    """

    ok: bool
    cited_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Return a short summary."""
        return f"ValidationResult(ok={self.ok}, cited={len(self.cited_ids)})"


def split_sentences(text: str) -> list[str]:
    """Split an answer into sentences.

    Parameters
    ----------
    text : str
        Answer text.

    Returns
    -------
    list[str]
        Non-empty sentences.
    """
    parts: list[str] = []
    for line in text.splitlines():
        parts.extend(s.strip() for s in _SENTENCE_RE.split(line))
    return [p for p in parts if p]


class CitationValidator:
    """Check that an answer cites only provided chunks, sentence by sentence."""

    def __str__(self) -> str:
        """Return the class name."""
        return "CitationValidator"

    def validate(self, answer: str, context_ids: set[str]) -> ValidationResult:
        """Validate an answer.

        Parameters
        ----------
        answer : str
            Model output.
        context_ids : set[str]
            Chunk ids that were given to the model.

        Returns
        -------
        ValidationResult
            Result with errors listed.
        """
        errors: list[str] = []
        cited: list[str] = []
        sentences = split_sentences(answer)
        if not sentences:
            errors.append("empty answer")
        for sent in sentences:
            found = CITATION_RE.findall(sent)
            if not found:
                errors.append(f"missing citation: {sent[:60]}")
            for _book, _page, cid in found:
                if cid not in context_ids:
                    errors.append(f"unknown chunk id: {cid}")
                elif cid not in cited:
                    cited.append(cid)
        return ValidationResult(ok=not errors, cited_ids=cited, errors=errors)
