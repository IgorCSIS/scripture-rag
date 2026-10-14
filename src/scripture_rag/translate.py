"""Language detection and PT/ES to English query translation."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Final

_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[a-záéíóúâêôãõçñü]+")
_HINTS: Final[dict[str, frozenset[str]]] = {
    "pt": frozenset("o a os as um uma que não em do da dos das no na nos nas com para por como quem qual "
                    "quando onde porque é são foi está seu sua ao à também mais muito você".split()),
    "es": frozenset("el la los las un una que no en del al con para por como quién quien cuál cuándo dónde "
                    "porque es son fue está su sus también más muy usted qué".split()),
    "en": frozenset("the a an of to in is are was what who which how why when where does did do and with "
                    "for from by that this his her their".split()),
}
_MARKERS: Final[dict[str, str]] = {"ç": "pt", "ã": "pt", "õ": "pt", "ê": "pt", "ñ": "es", "¿": "es", "¡": "es"}
MT_MODEL: Final[str] = "Helsinki-NLP/opus-mt-ROMANCE-en"


def detect_language(text: str) -> str:
    """Guess EN, PT or ES from function words and diacritics.

    Parameters
    ----------
    text : str
        Question.

    Returns
    -------
    str
        ``en``, ``pt`` or ``es``.
    """
    low = text.lower()
    scores = {lang: 0.0 for lang in _HINTS}
    for ch, lang in _MARKERS.items():
        if ch in low:
            scores[lang] += 2
    for w in _WORD_RE.findall(low):
        for lang, hints in _HINTS.items():
            if w in hints:
                scores[lang] += 1
    return max(scores, key=lambda k: (scores[k], k == "en"))


class QueryTranslator:
    """Translate PT/ES questions to English with a local MarianMT model.

    Parameters
    ----------
    model_name : str
        Hugging Face model id.
    """

    def __init__(self, model_name: str = MT_MODEL) -> None:
        """Load the model (downloads on first use)."""
        from transformers import MarianMTModel, MarianTokenizer

        self._tok = MarianTokenizer.from_pretrained(model_name)
        self._model = MarianMTModel.from_pretrained(model_name)

    def __str__(self) -> str:
        """Return the class name."""
        return "QueryTranslator"

    @lru_cache(maxsize=1024)  # noqa: B019 (one long-lived instance per process)
    def to_english(self, text: str) -> str:
        """Translate one question.

        Parameters
        ----------
        text : str
            PT or ES question.

        Returns
        -------
        str
            English translation.
        """
        batch = self._tok([text], return_tensors="pt", padding=True)
        out = self._model.generate(**batch, max_new_tokens=96, num_beams=4)
        return self._tok.batch_decode(out, skip_special_tokens=True)[0]
