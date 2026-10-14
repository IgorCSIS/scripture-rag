"""Verse references, the answer gate and per-language settings."""

from __future__ import annotations

from scripture_rag.chunker import Chunk
from scripture_rag.config import Settings
from scripture_rag.gate import GatePolicy, IdfTable, support
from scripture_rag.ingest import Document
from scripture_rag.pipeline import RagPipeline
from scripture_rag.verses import ReferenceAligner, Verse, parse_numbered, parse_suras, roman, span_ref, verse_chunks

REF = ("1:1 In the beginning God created the heaven and the earth. 1:2 And the earth was without form, and void. "
       "1:3 And God said, Let there be light: and there was light. 2:1 Thus the heavens and the earth were finished. "
       "1:1 Now these are the names of the children of Israel. 21:24 Eye for eye, tooth for tooth, hand for hand.")


def test_parse_numbered_books_and_refs() -> None:
    vs = parse_numbered(REF)
    assert [v.ref for v in vs][:4] == ["Genesis 1:1", "Genesis 1:2", "Genesis 1:3", "Genesis 2:1"]
    assert vs[-1].ref == "Exodus 21:24" and vs[-1].text.startswith("Eye for eye")


def test_roman_and_ocr_repairs() -> None:
    assert roman("XCV") == 95 and roman("XIV") == 14 and roman("U") == 2 and roman("lX") == 9 and roman("Q") == 0


def test_parse_suras_counts_lines_and_resyncs() -> None:
    page = ("The Koran 41\nSURA XCV. THE FIG\nMecca. 8 Verses\nIn the Name of God, the Compassionate, the Merciful\n"
            "I SWEAR by the FIG and by the olive,\nBy Mount Sinai,\nAnd by this inviolate soil !\n"
            "That of goodliest fabric we created man,\n5 Then brought him down to be the lowest of the low;\n"
            "Save who believe and do the things that are right, for theirs\nshall be a reward that faileth not.\n")
    vs = parse_suras(Document("k", "k", [page.replace("\\\n", "\n")]))
    refs = {v.ref: v.text for v in vs}
    assert refs["Sura 95:2"] == "By Mount Sinai,"
    assert refs["Sura 95:5"].startswith("Then brought him down")
    assert refs["Sura 95:6"].endswith("faileth not.")


def test_alignment_and_verse_chunks() -> None:
    ref = parse_numbered(REF)
    doc = Document("kjv", "KJV 1611", ["In the beginning God created the Heaven, and the Earth. And the earth was "
                                       "without forme, and voyd. And God said, Let there be light: and there was "
                                       "light."])
    vs = ReferenceAligner(ref).align(doc, min_score=0.3)
    assert [v.ref for v in vs][:3] == ["Genesis 1:1", "Genesis 1:2", "Genesis 1:3"]
    chunks = verse_chunks("kjv", vs, target_words=12, overlap_verses=0)
    assert chunks[0].ref.startswith("Genesis 1:1") and chunks[0].label.endswith("p.1")
    assert span_ref([Verse("Exodus", 21, 23, "", 1), Verse("Exodus", 21, 25, "", 1)]) == "Exodus 21:23-25"


def test_support_and_policy() -> None:
    idf = IdfTable(["eye for eye tooth for tooth", "the lord is my shepherd", "commentary of ibn kathir"])
    assert support("eye for eye and tooth for tooth", "eye for eye, tooth for tooth", idf) > 0.9
    assert support("Ibn Kathir on Alif Lam Mim", "Matth. vi. 26, carry not", idf) == 0.0
    pol = GatePolicy({"pt": 0.6}, default=0.3, min_support=0.2)
    assert pol.threshold("pt") == 0.6 and pol.threshold("en") == 0.3
    assert pol.allow("en", 0.4, 0.5) and not pol.allow("pt", 0.5, 0.5) and not pol.allow("en", 0.9, 0.1)


def test_settings_thresholds_and_pipeline_gate() -> None:
    s = Settings(lang_thresholds="en=0.2, pt=0.4", refusal_threshold=0.1, min_support=0.3, chunk_tokens=30,
                 chunk_overlap=5)
    assert s.thresholds() == {"en": 0.2, "pt": 0.4}
    pipe = RagPipeline(s)
    pipe.add_document(Document("b", "B", ["The Lord is my shepherd; I shall not want. He leadeth me beside the "
                                          "still waters."]))
    assert not pipe.ask("Who is my shepherd?").refused
    ans = pipe.ask("What does Ibn Kathir say about the farewell sermon?")
    assert ans.refused


def test_verse_chunker_in_pipeline() -> None:
    pipe = RagPipeline(Settings(chunker="verse", refusal_threshold=0.05, chunk_tokens=30, chunk_overlap=5))
    text = " ".join(f"{c}:{v} Verse number {v} of chapter {c} speaks of covenant and light." for c in (1, 2)
                    for v in range(1, 30))
    pipe.add_document(Document("bible", "Numbered", [text]))
    refs = {c.ref for c in pipe.chunks.values()}
    assert any(r.startswith("Genesis 1:") for r in refs) and any(r.startswith("Genesis 2:") for r in refs)
    assert all(isinstance(c, Chunk) for c in pipe.chunks.values())
