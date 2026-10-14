"""Runtime configuration read from environment variables.

No secrets live in code. Copy ``.env.example`` to ``.env`` to override defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Final
from urllib.parse import urlparse

REFUSAL_TEXT: Final[str] = "I could not find this in the loaded texts."
LOCAL_HOSTS: Final[frozenset[str]] = frozenset({"localhost", "127.0.0.1", "::1"})


def _env(name: str, default: str) -> str:
    """Read one environment variable.

    Parameters
    ----------
    name : str
        Variable name.
    default : str
        Value used when the variable is unset or empty.

    Returns
    -------
    str
        The configured value.
    """
    value = os.environ.get(name, "").strip()
    return value or default


@dataclass(frozen=True)
class Settings:
    """Immutable application settings.

    Parameters
    ----------
    mode : str
        ``local`` (Ollama generation) or ``extractive`` (no LLM).
    embed_model : str
        Sentence-transformers model id, or ``hashing`` for the offline embedder.
    rerank_model : str
        Cross-encoder model id, or ``lexical`` for the offline reranker.
    refusal_threshold : float
        Minimum top rerank score (0 to 1) required to answer.
    ollama_url : str
        Ollama base URL; must point to localhost.
    ollama_model : str
        Primary Ollama model tag.
    ollama_fallback_model : str
        Model tag used when the primary is not pulled.
    max_upload_mb : int
        Upload size cap in megabytes.
    max_pages : int
        Page cap per document.
    parse_timeout_s : float
        Wall clock limit for parsing one file.
    chunk_tokens : int
        Target chunk size in tokens.
    chunk_overlap : int
        Token overlap between neighbor chunks.
    candidates_k : int
        Candidates pulled from each retriever before fusion.
    top_n : int
        Passages kept after reranking.
    chunker : str
        ``window`` (token windows) or ``verse`` (chapter:verse and sura:ayah aware).
    normalize : bool
        Index 1611-style spellings in modern form.
    translate : bool
        Translate PT/ES questions to English before retrieval and reranking.
    expand : bool
        Add archaic synonyms to the lexical query.
    dense_weight : float
        Weight of the dense lists in weighted rank fusion (BM25 gets the rest).
    lang_thresholds : str
        Per-language gate thresholds, ``en=0.3,pt=0.4``; others use ``refusal_threshold``.
    min_support : float
        Minimum lexical support of the best passage (0 disables the check).
    reference_path : str
        Numbered reference text used to align unnumbered Bible editions.
    """

    mode: str = "extractive"
    embed_model: str = "hashing"
    rerank_model: str = "lexical"
    refusal_threshold: float = 0.5
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_fallback_model: str = "llama3.2:3b"
    max_upload_mb: int = 50
    max_pages: int = 2000
    parse_timeout_s: float = 60.0
    chunk_tokens: int = 400
    chunk_overlap: int = 60
    candidates_k: int = 50
    top_n: int = 5
    chunker: str = "window"
    normalize: bool = False
    translate: bool = False
    expand: bool = False
    dense_weight: float = 0.5
    lang_thresholds: str = ""
    min_support: float = 0.0
    reference_path: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate settings.

        Raises
        ------
        ValueError
            If the mode is unknown or the Ollama URL is not on localhost.
        """
        if self.chunker not in {"window", "verse"}:
            raise ValueError(f"CHUNKER must be 'window' or 'verse', got {self.chunker!r}")
        if self.mode not in {"local", "extractive"}:
            raise ValueError(f"MODE must be 'local' or 'extractive', got {self.mode!r}")
        host = urlparse(self.ollama_url).hostname or ""
        if host not in LOCAL_HOSTS:
            raise ValueError("OLLAMA_URL must point to localhost")
        if not 0 <= self.chunk_overlap < self.chunk_tokens:
            raise ValueError("CHUNK_OVERLAP must be in [0, CHUNK_TOKENS)")

    def __str__(self) -> str:
        """Return a short human readable summary."""
        return f"Settings(mode={self.mode}, embed={self.embed_model}, rerank={self.rerank_model})"

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables.

        Returns
        -------
        Settings
            Settings with environment overrides applied.
        """
        d = cls.__dataclass_fields__
        return cls(
            mode=_env("MODE", d["mode"].default),
            embed_model=_env("EMBED_MODEL", d["embed_model"].default),
            rerank_model=_env("RERANK_MODEL", d["rerank_model"].default),
            refusal_threshold=float(_env("REFUSAL_THRESHOLD", str(d["refusal_threshold"].default))),
            ollama_url=_env("OLLAMA_URL", d["ollama_url"].default),
            ollama_model=_env("OLLAMA_MODEL", d["ollama_model"].default),
            ollama_fallback_model=_env("OLLAMA_FALLBACK_MODEL", d["ollama_fallback_model"].default),
            max_upload_mb=int(_env("MAX_UPLOAD_MB", str(d["max_upload_mb"].default))),
            max_pages=int(_env("MAX_PAGES", str(d["max_pages"].default))),
            parse_timeout_s=float(_env("PARSE_TIMEOUT_S", str(d["parse_timeout_s"].default))),
            chunk_tokens=int(_env("CHUNK_TOKENS", str(d["chunk_tokens"].default))),
            chunk_overlap=int(_env("CHUNK_OVERLAP", str(d["chunk_overlap"].default))),
            chunker=_env("CHUNKER", d["chunker"].default),
            normalize=_env("NORMALIZE", "0") == "1",
            translate=_env("TRANSLATE", "0") == "1",
            expand=_env("EXPAND", "0") == "1",
            dense_weight=float(_env("DENSE_WEIGHT", str(d["dense_weight"].default))),
            lang_thresholds=_env("LANG_THRESHOLDS", ""),
            min_support=float(_env("MIN_SUPPORT", str(d["min_support"].default))),
            reference_path=_env("REFERENCE_PATH", ""),
        )

    def thresholds(self) -> dict[str, float]:
        """Parse ``lang_thresholds`` into a dict."""
        out = {}
        for part in filter(None, (p.strip() for p in self.lang_thresholds.split(","))):
            k, _, v = part.partition("=")
            out[k.strip()] = float(v)
        return out
