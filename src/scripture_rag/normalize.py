"""Early Modern English spelling normalization for indexing (not for display).

Follows the VARD idea (Baron and Rayson, 2008): keep the original text and
add a modern form for retrieval. A token that is not in the modern vocabulary
is rewritten with the first sequence of spelling rules (breadth first, depth
<= 3) that produces an in-vocabulary word; otherwise it is kept as is.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Callable, Iterable
from functools import lru_cache
from pathlib import Path
from typing import Final

_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z]+")
VOCAB_PATH: Final[Path] = Path(__file__).with_name("data") / "modern_vocab.txt"

# Modern query words mapped to the archaic or translation-specific word used in the texts.
QUERY_SYNONYMS: Final[dict[str, str]] = {
    "donkey": "ass", "donkeys": "asses", "quran": "koran", "qur": "koran", "saturday": "sabbath",
    "you": "thee", "your": "thy", "clothes": "raiment", "clothing": "raiment", "boat": "ark",
}


def _rules() -> list[Callable[[str], str | None]]:
    """Return the rewrite rules, each mapping a word to a variant or None."""

    def sub(pattern: str, repl: str) -> Callable[[str], str | None]:
        rx = re.compile(pattern)

        def rule(w: str) -> str | None:
            out = rx.sub(repl, w, count=1)
            return out if out != w else None

        return rule

    return [
        sub(r"e$", ""),                     # goe -> go, olde -> old
        sub(r"([bcdfgklmnprst])\1$", r"\1"),  # sonn -> son, shall stays (in vocab)
        sub(r"([bcdfgklmnprst])\1", r"\1"),   # doubled consonant inside
        sub(r"ie$", "y"),                   # citie -> city, drie -> dry
        sub(r"ee", "ea"),                   # yeere -> yeare -> year
        sub(r"ee", "e"),                    # cleere -> clere
        sub(r"^v", "u"),                    # vnto -> unto
        sub(r"(?<=[aeiou])u(?=[aeiou])", "v"),  # giue -> give, haue -> have
        sub(r"^i(?=[aeiou])", "j"),         # iudge -> judge
        sub(r"^l(?=[aeiou])", "j"),         # OCR: lacob -> jacob, ludges -> judges
        sub(r"^l(?=[aeiou])", "i"),         # OCR: lsaac -> isaac
        sub(r"es$", "s"),                   # yeares -> years, starres -> stares -> stars
        sub(r"oo", "o"),                    # foorth -> forth
        sub(r"neth$", "nth"),               # moneth -> month
        sub(r"eth$", "es"),                 # goeth -> goes
        sub(r"eth$", "s"),                  # maketh -> makes
        sub(r"y$", "ie"),                   # reverse for completeness
        sub(r"ck$", "c"),                   # musick -> music
        sub(r"our$", "or"),                 # rare, kept for OCR variants
        sub(r"ie(?=[a-z])", "y"),           # dyed/died style
        sub(r"(?<=[a-z])y(?=[a-z])", "i"),  # thyrd -> third
    ]


class ArchaicNormalizer:
    """Rewrite archaic or OCR-damaged words into modern forms.

    Parameters
    ----------
    vocab : Iterable[str]
        Modern word list (lowercase).
    max_depth : int
        Maximum number of chained rules.
    """

    def __init__(self, vocab: Iterable[str], max_depth: int = 3) -> None:
        """Store the vocabulary and rules."""
        self._vocab = frozenset(vocab)
        self._rules = _rules()
        self._depth = max_depth
        self._cache: dict[str, str] = {}

    def __str__(self) -> str:
        """Return a short summary."""
        return f"ArchaicNormalizer(vocab={len(self._vocab)})"

    @classmethod
    def default(cls) -> ArchaicNormalizer:
        """Load the bundled modern vocabulary.

        Returns
        -------
        ArchaicNormalizer
            Normalizer backed by ``data/modern_vocab.txt``.
        """
        return cls(VOCAB_PATH.read_text(encoding="utf-8").split())

    def word(self, w: str) -> str:
        """Normalize one lowercase word.

        Parameters
        ----------
        w : str
            Word.

        Returns
        -------
        str
            Modern form, or the input when no rule chain reaches the vocabulary.
        """
        if len(w) < 3:
            return w
        if w in self._vocab:
            # Archaic verb endings map to the modern form when it is known: goeth -> goes.
            if w.endswith("eth") and len(w) >= 5:
                for cand in (w[:-3] + "es", w[:-3] + "s"):
                    if cand in self._vocab:
                        return cand
            return w
        if w in self._cache:
            return self._cache[w]
        found = w
        queue, seen = deque([(w, 0)]), {w}
        while queue:
            cur, depth = queue.popleft()
            if depth == self._depth:
                continue
            for rule in self._rules:
                nxt = rule(cur)
                if nxt is None or nxt in seen or len(nxt) < 2:
                    continue
                if nxt in self._vocab:
                    found = nxt
                    queue.clear()
                    break
                seen.add(nxt)
                queue.append((nxt, depth + 1))
        self._cache[w] = found
        return found

    def text(self, text: str) -> str:
        """Normalize every word in a text, lowercased.

        Parameters
        ----------
        text : str
            Original passage.

        Returns
        -------
        str
            Modernized text for indexing.
        """
        return _WORD_RE.sub(lambda m: self.word(m.group(0).lower()), text)


def expand_query(query: str) -> str:
    """Append archaic equivalents of modern query words.

    Parameters
    ----------
    query : str
        User question (English).

    Returns
    -------
    str
        Query with synonyms appended.
    """
    extra = [QUERY_SYNONYMS[w] for w in re.findall(r"[a-z]+", query.lower()) if w in QUERY_SYNONYMS]
    return query if not extra else f"{query} {' '.join(extra)}"


@lru_cache(maxsize=1)
def default_normalizer() -> ArchaicNormalizer:
    """Shared normalizer instance.

    Returns
    -------
    ArchaicNormalizer
        Cached default normalizer.
    """
    return ArchaicNormalizer.default()


def build_vocab(texts: Iterable[str], min_count: int = 2) -> list[str]:
    """Build a modern vocabulary from reference texts.

    Parameters
    ----------
    texts : Iterable[str]
        Modern-spelling texts.
    min_count : int
        Minimum frequency to keep a word (filters OCR noise).

    Returns
    -------
    list[str]
        Sorted words.
    """
    counts: dict[str, int] = {}
    for t in texts:
        for w in _WORD_RE.findall(t.lower()):
            counts[w] = counts.get(w, 0) + 1
    return sorted(w for w, c in counts.items() if c >= min_count)
