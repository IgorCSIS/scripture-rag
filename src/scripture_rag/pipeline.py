"""End to end pipeline: ingest, index, retrieve, rerank, answer, validate."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .bm25 import BM25Index
from .chunker import Chunk, Chunker
from .config import PT_UNSUPPORTED_TEXT, REFUSAL_TEXT, Settings
from .embed import Embedder, make_embedder
from .gate import GatePolicy, IdfTable, support
from .generate import ExtractiveGenerator, Generator
from .hybrid import HybridRetriever
from .index import BruteForceIndex
from .ingest import Document
from .normalize import default_normalizer, expand_query
from .rerank import Reranker, make_reranker
from .sanitize import clean_query
from .translate import detect_language
from .validate import CitationValidator
from .verses import BIBLE_BOOKS, ReferenceAligner, parse_numbered, parse_suras, verse_chunks


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
        Override the extractive generator.
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
        self.generator = generator or ExtractiveGenerator()
        self.chunker = Chunker(s.chunk_tokens, s.chunk_overlap)
        self.vectors = BruteForceIndex()
        self.bm25 = BM25Index(s.bm25_k1)
        self.retriever = HybridRetriever(self.embedder, self.vectors, self.bm25)
        self.validator = CitationValidator()
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, Chunk] = {}
        self.index_text: dict[str, str] = {}
        self.gate = GatePolicy(s.thresholds(), s.refusal_threshold, s.min_support)
        self._idf: IdfTable | None = None
        self._translator = None
        self._aligner: ReferenceAligner | None = None

    def __str__(self) -> str:
        """Return a short summary."""
        return f"RagPipeline(books={len(self.documents)}, chunks={len(self.chunks)})"

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
        chunks = self.split(doc)
        if not chunks:
            return 0
        self.documents[doc.book_id] = doc
        ids = [c.chunk_id for c in chunks]
        norm = default_normalizer() if self.settings.normalize else None
        texts = [norm.text(c.text) if norm else c.text for c in chunks]
        self.chunks.update(zip(ids, chunks))
        self.index_text.update(zip(ids, texts))
        self._idf = None
        if vectors is None or len(vectors) != len(chunks):
            vectors = self.embedder.encode(texts)
        self.vectors.add(ids, vectors)
        self.bm25.add(ids, texts)
        return len(chunks)

    def split(self, doc: Document) -> list[Chunk]:
        """Chunk with verse references when the layout allows it, else token windows.

        Numbered editions (``21:24`` markers) and suras with ``SURA`` headings are
        parsed directly; an unnumbered Bible is aligned to ``reference_path``.
        Text the verse parsers skip (notes, front matter) is kept as windows.
        """
        if self.settings.chunker != "verse":
            return self.chunker.split(doc)
        text = "\n".join(doc.pages)
        verses = []
        if len(re.findall(r"(?<![\d:])\d{1,3}:\d{1,3}\s", text)) > 50:
            verses = parse_numbered(text)
        elif len(re.findall(r"(?m)^\s*SURA\s", text)) >= 3:
            verses = parse_suras(doc)
        elif self.settings.reference_path:
            if self._aligner is None:
                ref = parse_numbered(open(self.settings.reference_path, encoding="utf-8").read(), BIBLE_BOOKS)
                self._aligner = ReferenceAligner(ref, default_normalizer())
            verses = self._aligner.align(doc)
            if len(verses) < max(50, 0.2 * max(1, len(text.split()) // 30)):
                verses = []  # not this work: alignment found too little
        if not verses:
            return self.chunker.split(doc)
        chunks = verse_chunks(doc.book_id, verses, 80)
        windows = self.chunker.split(Document(doc.book_id, doc.title, doc.pages))
        if verses[0].book == "Sura":  # keep translator's notes searchable too
            chunks += [Chunk(f"{doc.book_id}-n{i:04d}", doc.book_id, c.text, c.page_start, c.page_end,
                             c.char_start, c.char_end) for i, c in enumerate(windows)]
        return chunks

    def _english(self, question: str, lang: str) -> str:
        """Translate PT/ES questions when enabled."""
        if not self.settings.translate or lang == "en" or (lang == "pt" and not self.settings.portuguese):
            return question
        if self._translator is None:
            from .translate import QueryTranslator

            self._translator = QueryTranslator()
        return self._translator.to_english(question)

    def retrieve(self, question: str, q_en: str, k: int) -> list[str]:
        """Weighted rank fusion of BM25 (expanded English query, plus the original query when it was
        translated) and dense lists (original and English)."""
        s = self.settings
        q_bm = expand_query(q_en) if s.expand else q_en
        lists = [(self.retriever.bm25_only(q_bm, k), 1.0 - s.dense_weight)]
        if q_en != question:
            # Names and loan words survive in the original; guards against MT errors such as Gabaon -> Gabon.
            lists.append((self.retriever.bm25_only(question, k), 1.0 - s.dense_weight))
        for q in dict.fromkeys([question, q_en]):
            lists.append((self.retriever.vector_only(q, k), s.dense_weight))
        scores: dict[str, float] = {}
        for ids, w in lists:
            for r, cid in enumerate(ids, 1):
                scores[cid] = scores.get(cid, 0.0) + w / (s.rrf_k + r)
        return [c for c, _ in sorted(scores.items(), key=lambda kv: -kv[1])][:k]

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
        question = clean_query(question)
        lang = detect_language(question)
        if lang == "pt" and not s.portuguese:
            return Answer(question, PT_UNSUPPORTED_TEXT, True, 0.0, reason="unsupported language (pt)")
        q_en = self._english(question, lang)
        if books:
            # Over-fetch, then keep candidates from the selected books.
            pool = self.retrieve(question, q_en, s.candidates_k * 4)
            cand_ids = [c for c in pool if self.chunks[c].book_id in books][: s.candidates_k]
        else:
            cand_ids = self.retrieve(question, q_en, s.candidates_k)
        ranked = self.reranker.rerank(q_en, [self.chunks[c] for c in cand_ids], s.top_n)
        top = ranked[0][1] if ranked else 0.0
        if self._idf is None:
            self._idf = IdfTable(self.index_text.values())
        sup = [support(q_en, self.index_text[c.chunk_id], self._idf) for c, _ in ranked]
        t = self.gate.threshold(lang)
        if top < t:
            return Answer(question, REFUSAL_TEXT, True, top, reason=f"retrieval gate: score below threshold ({lang})")
        if not self.gate.allow(lang, top, sup[0]):
            return Answer(question, REFUSAL_TEXT, True, top,
                          reason="support check: passage does not cover the question")
        # Only passages that pass both checks are used as evidence.
        context = [c for (c, score), sp in zip(ranked, sup) if score >= t and sp >= self.gate.min_support][:3]
        draft = self.generator.answer(question, context)
        if draft.strip() == REFUSAL_TEXT:
            return Answer(question, REFUSAL_TEXT, True, top, reason="generator refused")
        result = self.validator.validate(draft, {c.chunk_id for c in context})
        if not result.ok:
            return Answer(question, REFUSAL_TEXT, True, top, reason="citation check: " + "; ".join(result.errors[:3]))
        return Answer(question, draft, False, top, [self.chunks[c] for c in result.cited_ids])
