"""Runtime configuration read from environment variables.

No secrets live in code. Copy ``.env.example`` to ``.env`` to override defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Final

REFUSAL_TEXT: Final[str] = "I could not find this in the loaded texts."
PT_UNSUPPORTED_TEXT: Final[str] = "Portuguese questions are not supported yet; please ask in English or Spanish."


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
    embed_model : str
        Sentence-transformers model id, or ``hashing`` for the offline embedder.
    rerank_model : str
        Cross-encoder model id, or ``lexical`` for the offline reranker.
    refusal_threshold : float
        Minimum top rerank score (0 to 1) required to answer.
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
    bm25_k1 : float
        BM25 term-frequency saturation (chosen on tune + dev, see docs/eval/round2.json).
    portuguese : bool
        Answer Portuguese questions (off for launch: they get ``PT_UNSUPPORTED_TEXT``, no retrieval).
    rrf_k : int
        Rank fusion constant in ``w / (rrf_k + rank)``.
    reference_path : str
        Numbered reference text used to align unnumbered Bible editions.
    """

    embed_model: str = "hashing"
    rerank_model: str = "lexical"
    refusal_threshold: float = 0.5
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
    lang_thresholds: str = "en=0.12,pt=0.72,es=0.16"
    min_support: float = 0.3
    bm25_k1: float = 1.5
    rrf_k: int = 60
    portuguese: bool = False
    reference_path: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate settings.

        Raises
        ------
        ValueError
            If the chunker or chunk overlap is invalid.
        """
        if self.chunker not in {"window", "verse"}:
            raise ValueError(f"CHUNKER must be 'window' or 'verse', got {self.chunker!r}")
        if not 0 <= self.chunk_overlap < self.chunk_tokens:
            raise ValueError("CHUNK_OVERLAP must be in [0, CHUNK_TOKENS)")

    def __str__(self) -> str:
        """Return a short human readable summary."""
        return f"Settings(extractive, embed={self.embed_model}, rerank={self.rerank_model})"

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
            embed_model=_env("EMBED_MODEL", d["embed_model"].default),
            rerank_model=_env("RERANK_MODEL", d["rerank_model"].default),
            refusal_threshold=float(_env("REFUSAL_THRESHOLD", str(d["refusal_threshold"].default))),
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
            lang_thresholds=_env("LANG_THRESHOLDS", d["lang_thresholds"].default),
            min_support=float(_env("MIN_SUPPORT", str(d["min_support"].default))),
            portuguese=_env("PORTUGUESE", "0") == "1",
            reference_path=_env("REFERENCE_PATH", ""),
        )

    def thresholds(self) -> dict[str, float]:
        """Parse ``lang_thresholds`` into a dict."""
        out = {}
        for part in filter(None, (p.strip() for p in self.lang_thresholds.split(","))):
            k, _, v = part.partition("=")
            out[k.strip()] = float(v)
        return out
