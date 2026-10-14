"""Rerankers that score (question, passage) pairs on a 0 to 1 scale (FR-05)."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from .chunker import Chunk
from .text_utils import tokenize


def windows(text: str, size: int, stride: int) -> list[str]:
    """Split a passage into overlapping word windows.

    Parameters
    ----------
    text : str
        Passage.
    size : int
        Words per window.
    stride : int
        Words between window starts.

    Returns
    -------
    list[str]
        At least one window.
    """
    words = text.split()
    if len(words) <= size:
        return [text]
    return [" ".join(words[i : i + size]) for i in range(0, len(words) - size + stride, stride)]


class Reranker(ABC):
    """Interface for pairwise relevance scoring.

    When ``window_words`` is set, each passage is scored as the maximum over
    short overlapping windows, so one relevant verse inside a long, noisy OCR
    chunk is not diluted by its neighbors.
    """

    window_words: int | None = None

    @abstractmethod
    def score(self, query: str, passages: list[str]) -> list[float]:
        """Score passages for one query.

        Parameters
        ----------
        query : str
            Question.
        passages : list[str]
            Candidate texts.

        Returns
        -------
        list[float]
            Relevance in [0, 1], one per passage.
        """

    def rerank(self, query: str, chunks: list[Chunk], top_n: int = 5) -> list[tuple[Chunk, float]]:
        """Sort chunks by relevance.

        Parameters
        ----------
        query : str
            Question.
        chunks : list[Chunk]
            Candidates.
        top_n : int
            Number to keep.

        Returns
        -------
        list[tuple[Chunk, float]]
            Best chunks with scores, descending.
        """
        if not chunks:
            return []
        if self.window_words:
            size = self.window_words
            spans = [windows(c.text, size, size // 2) for c in chunks]
            flat = self.score(query, [w for span in spans for w in span])
            scores, pos = [], 0
            for span in spans:
                scores.append(max(flat[pos : pos + len(span)]))
                pos += len(span)
        else:
            scores = self.score(query, [c.text for c in chunks])
        ranked = sorted(zip(chunks, scores), key=lambda cs: cs[1], reverse=True)
        return ranked[:top_n]

    def __str__(self) -> str:
        """Return the class name."""
        return type(self).__name__


class LexicalOverlapReranker(Reranker):
    """Offline reranker: share of distinct query content words found in the passage.

    A transparent baseline for tests and the no-download mode.
    """

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Score by query term coverage (see ``Reranker.score``)."""
        q_terms = set(tokenize(query))
        if not q_terms:
            return [0.0 for _ in passages]
        out = []
        for text in passages:
            p_terms = set(tokenize(text))
            out.append(len(q_terms & p_terms) / len(q_terms))
        return out


MULTILINGUAL_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class CrossEncoderReranker(Reranker):
    """Cross-encoder reranker with a sigmoid to map logits to [0, 1].

    ``cross-encoder/ms-marco-MiniLM-L-6-v2`` is English only; use
    ``MULTILINGUAL_RERANKER`` for PT and ES questions.

    Parameters
    ----------
    model_name : str
        Hugging Face model id.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", window_words: int | None = 60) -> None:
        """Load the model (downloads on first use).

        Parameters
        ----------
        model_name : str
            Hugging Face model id.
        window_words : int or None
            Window size for max-window scoring; None scores whole passages.
        """
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)
        self.window_words = window_words

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Score pairs with the cross-encoder (see ``Reranker.score``)."""
        import torch

        # Identity keeps raw logits whatever activation the model config sets, so the
        # sigmoid below is applied exactly once.
        logits = self._model.predict(
            [(query, p) for p in passages], batch_size=16, show_progress_bar=False, activation_fn=torch.nn.Identity()
        )
        return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]


def make_reranker(name: str) -> Reranker:
    """Build a reranker from a config name.

    Parameters
    ----------
    name : str
        ``lexical`` or a cross-encoder model id.

    Returns
    -------
    Reranker
        Ready reranker.
    """
    if name == "lexical":
        return LexicalOverlapReranker()
    return CrossEncoderReranker(name)
