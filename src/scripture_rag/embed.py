"""Text embedders behind one interface."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Final

import numpy as np

from .text_utils import tokenize

E5_MODELS: Final[frozenset[str]] = frozenset({"intfloat/multilingual-e5-small", "intfloat/multilingual-e5-base"})


class Embedder(ABC):
    """Interface: map texts to L2-normalized float32 vectors."""

    dim: int

    @abstractmethod
    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Embed texts.

        Parameters
        ----------
        texts : list[str]
            Inputs.
        is_query : bool
            True for questions, False for passages.

        Returns
        -------
        np.ndarray
            Array of shape (len(texts), dim), rows of unit length.
        """

    def __str__(self) -> str:
        """Return the class name and dimension."""
        return f"{type(self).__name__}(dim={self.dim})"


class HashingEmbedder(Embedder):
    """Deterministic offline embedder using the hashing trick.

    Used in tests and as a no-download fallback. It captures word overlap,
    not meaning.

    Parameters
    ----------
    dim : int
        Output dimension.
    """

    def __init__(self, dim: int = 256) -> None:
        """Create the embedder."""
        self.dim = dim

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Embed texts by hashing tokens into buckets.

        Parameters
        ----------
        texts : list[str]
            Inputs.
        is_query : bool
            Ignored; kept for interface parity.

        Returns
        -------
        np.ndarray
            Unit vectors of shape (len(texts), dim).
        """
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in tokenize(text):
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
                out[row, h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)


class SentenceTransformerEmbedder(Embedder):
    """Embedder backed by a sentence-transformers model.

    Parameters
    ----------
    model_name : str
        Hugging Face model id.
    device : str or None
        Torch device, for example ``cuda``; auto when None.
    """

    def __init__(self, model_name: str, device: str | None = None) -> None:
        """Load the model (downloads on first use)."""
        from sentence_transformers import SentenceTransformer

        self._name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        get_dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Embed texts, adding E5 prefixes when the model needs them.

        Parameters
        ----------
        texts : list[str]
            Inputs.
        is_query : bool
            True for questions.

        Returns
        -------
        np.ndarray
            Unit vectors.
        """
        if self._name in E5_MODELS:
            prefix = "query: " if is_query else "passage: "
            texts = [prefix + t for t in texts]
        vecs = self._model.encode(
            texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)


def make_embedder(name: str) -> Embedder:
    """Build an embedder from a config name.

    Parameters
    ----------
    name : str
        ``hashing`` or a sentence-transformers model id.

    Returns
    -------
    Embedder
        Ready embedder.
    """
    if name == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(name)
