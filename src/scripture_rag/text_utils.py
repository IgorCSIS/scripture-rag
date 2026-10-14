"""Shared tokenization helpers."""

from __future__ import annotations

import re
import unicodedata
from typing import Final

_WORD_RE: Final[re.Pattern[str]] = re.compile(r"\w+", re.UNICODE)

# Small multilingual stopword list (EN, PT, ES) for lexical scoring.
STOPWORDS: Final[frozenset[str]] = frozenset(
    """a an and are as at be by for from has have he her his i in is it its of on or
    that the their them they this to was were what when where which who whom why will
    with you your our we us my me does did do how not no shall unto thee thou o e el la los las de del
    que en un una por para con se do da dos das um uma os ao com qual quem como""".split()
)


def normalize(text: str) -> str:
    """Lowercase and strip accents.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Normalized text.
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    """Split text into normalized word tokens.

    Parameters
    ----------
    text : str
        Input text.
    drop_stopwords : bool
        Remove common function words when True.

    Returns
    -------
    list[str]
        Tokens.
    """
    words = _WORD_RE.findall(normalize(text))
    if drop_stopwords:
        return [w for w in words if w not in STOPWORDS]
    return words
