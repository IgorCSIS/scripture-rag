"""End to end pipeline: ingest, index, retrieve, rerank, answer, validate."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bm25 import BM25Index
from .chunker import Chunk, Chunker
from .config import REFUSAL_TEXT, Settings
from .embed import Embedder, make_embedder
from .generate import ExtractiveGenerator, Generator, OllamaGenerator
from .hybrid import HybridRetriever
from .index import BruteForceIndex
from .ingest import Document
from .rerank import Reranker, make_reranker
from .validate import CitationValidator


@dataclass
class Answer:
    """Result of one question.

    Parameters
    ----------
    question : str
        The question.
    text : str
        Answer or refusal text.
    refused : bool
        True when the system declined to answer.
    top_score : float
        Best rerank score.
    citations : list[Chunk]
        Chunks cited in the answer.
    reason : str
        Why the system refused, if it did.
    """

    question: str
    text: str
    refused: bool
    top_score: float
    citations: list[Chunk] = field(default_factory=list)
    reason: str = ""

    def __str__(self) -> str:
        """Return the answer text."""
        return self.text


class RagPipeline:
    """Owns the indexes and runs the ask flow.

    Parameters
    ----------
    settings : Settings
        Configuration.
    embedder : Embedder or None
        Override the configured embedder (tests use a fake one).
    reranker : Reranker or None
        Override the configured reranker.
    generator : Generator or None
        Override the generator chosen by ``settings.mode``.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        generator: Generator | None = None,
    ) -> None:
        """Build components."""
        self.settings = settings or Settings.from_env()
        s = self.settings
        self.embedder = embedder or make_embedder(s.embed_model)
        self.reranker = reranker or make_reranker(s.rerank_model)
        if generator is not None:
            self.generator = generator
        elif s.mode == "local":
            self.generator = OllamaGenerator(s.ollama_url, s.ollama_model, s.ollama_fallback_model)
        else:
            self.generator = ExtractiveGenerator()
        self.chunker = Chunker(s.chunk_tokens, s.chunk_overlap)
        self.vectors = BruteForceIndex()
        self.bm25 = BM25Index()
        self.retriever = HybridRetriever(self.embedder, self.vectors, self.bm25)
        self.validator = CitationValidator()
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, Chunk] = {}

    def __str__(self) -> str:
        """Return a short summary."""
        return f"RagPipeline(books={len(self.documents)}, chunks={len(self.chunks)}, mode={self.settings.mode})"

    def add_document(self, doc: Document, vectors: np.ndarray | None = None) -> int:
        """Chunk, embed and index a document.

        Parameters
        ----------
        doc : Document
            Parsed document. A duplicate book id gets a numeric suffix.
        vectors : np.ndarray or None
            Precomputed passage embeddings (one row per chunk, same chunker
            settings and model); computed when None.

        Returns
        -------
        int
            Number of chunks added.
        """
        base, n = doc.book_id, 2
        while doc.book_id in self.documents:
            doc.book_id = f"{base}-{n}"
            n += 1
        chunks = self.chunker.split(doc)
        if not chunks:
            return 0
        self.documents[doc.book_id] = doc
        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        self.chunks.update(zip(ids, chunks))
        if vectors is None or len(vectors) != len(chunks):
            vectors = self.embedder.encode(texts)
        self.vectors.add(ids, vectors)
        self.bm25.add(ids, texts)
        return len(chunks)

    def embeddings(self, books: set[str] | None = None) -> tuple[list[str], np.ndarray]:
        """Return chunk ids and vectors, optionally only for some books.

        Parameters
        ----------
        books : set[str] or None
            Book ids to keep; None keeps all. O(n) filter.

        Returns
        -------
        tuple[list[str], np.ndarray]
            Ids and an (n, d) matrix.
        """
        if not books:
            return self.vectors.ids, self.vectors.matrix
        keep = [k for k, i in enumerate(self.vectors.ids) if self.chunks[i].book_id in books]
        return [self.vectors.ids[k] for k in keep], self.vectors.matrix[keep]

    def ask(self, question: str, books: set[str] | None = None) -> Answer:
        """Answer from loaded text only, or refuse.

        Parameters
        ----------
        question : str
            User question.
        books : set[str] or None
            Restrict evidence to these book ids (None searches all).

        Returns
        -------
        Answer
            Cited answer or refusal.
        """
        s = self.settings
        if not self.chunks or not question.strip():
            return Answer(question, REFUSAL_TEXT, True, 0.0, reason="no text loaded or empty question")
        if books:
            # Over-fetch, then keep candidates from the selected books.
            pool = self.retriever.retrieve(question, s.candidates_k * 4)
            cand_ids = [c for c in pool if self.chunks[c].book_id in books][: s.candidates_k]
        else:
            cand_ids = self.retriever.retrieve(question, s.candidates_k)
        ranked = self.reranker.rerank(question, [self.chunks[c] for c in cand_ids], s.top_n)
        top = ranked[0][1] if ranked else 0.0
        if top < s.refusal_threshold:
            return Answer(question, REFUSAL_TEXT, True, top, reason="retrieval gate: score below threshold")
        # Only passages that pass the gate are used as evidence.
        context = [c for c, score in ranked if score >= s.refusal_threshold]
        draft = self.generator.answer(question, context)
        if draft.strip() == REFUSAL_TEXT:
            return Answer(question, REFUSAL_TEXT, True, top, reason="generator refused")
        result = self.validator.validate(draft, {c.chunk_id for c in context})
        if not result.ok:
            return Answer(question, REFUSAL_TEXT, True, top, reason="citation check: " + "; ".join(result.errors[:3]))
        return Answer(question, draft, False, top, [self.chunks[c] for c in result.cited_ids])
