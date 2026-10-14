"""Shared fixtures: fast pipeline with the offline embedder and reranker."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripture_rag.config import Settings
from scripture_rag.corpus import load_folder
from scripture_rag.embed import HashingEmbedder
from scripture_rag.pipeline import RagPipeline
from scripture_rag.rerank import LexicalOverlapReranker

FIXTURES = Path(__file__).parent / "fixtures"


def pytest_configure(config: pytest.Config) -> None:
    """Register the slow marker."""
    config.addinivalue_line("markers", "slow: needs model downloads")


@pytest.fixture()
def settings() -> Settings:
    """Small chunks so fixtures produce several passages."""
    return Settings(chunk_tokens=40, chunk_overlap=8, refusal_threshold=0.5)


@pytest.fixture()
def pipeline(settings: Settings) -> RagPipeline:
    """Pipeline loaded with the public-domain fixture books."""
    p = RagPipeline(settings, embedder=HashingEmbedder(), reranker=LexicalOverlapReranker())
    load_folder(p, FIXTURES / "books")
    return p
