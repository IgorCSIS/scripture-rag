"""Verse-aware chunking with book chapter:verse and sura:ayah references (FR-17).

Three ways to recover verse references, chosen per document:

* ``parse_numbered``: editions that print ``21:24`` before each verse.
* ``parse_suras``: Quran translations with ``SURA <roman>`` headings, one verse
  per line and printed numbers every few verses (used to resynchronize).
* ``align_to_reference``: editions without verse numbers (the 1611 KJV
  facsimile) are aligned sentence by sentence to a numbered reference text of
  the same work, with an inverted index and a monotonic position prior.

Verses are then grouped into short chunks that never cross a chapter or sura,
so a citation reads ``Exodus 21:23-25``.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Final

from .chunker import Chunk
from .ingest import Document
from .normalize import ArchaicNormalizer
from .text_utils import tokenize

BIBLE_BOOKS: Final[tuple[str, ...]] = (
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua", "Judges", "Ruth", "1 Samuel",
    "2 Samuel", "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah", "Esther", "Job",
    "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon", "Isaiah", "Jeremiah", "Lamentations", "Ezekiel",
    "Daniel", "Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai",
    "Zechariah", "Malachi", "Matthew", "Mark", "Luke", "John", "Acts", "Romans", "1 Corinthians",
    "2 Corinthians", "Galatians", "Ephesians", "Philippians", "Colossians", "1 Thessalonians",
    "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews", "James", "1 Peter", "2 Peter",
    "1 John", "2 John", "3 John", "Jude", "Revelation",
)
_NUMBERED_RE: Final = re.compile(r"(?<![\d:])(\d{1,3}):(\d{1,3})\s")
_SURA_RE: Final = re.compile(r"^\s*S\s?U\s?R\s?A\S?\s+(?:\d\s+)?([IVXLCivxlcnUe]+)(?=[\W\d]|$)")
_OCR_ROMAN: Final = str.maketrans({"n": "I", "l": "I", "i": "I", "v": "V", "x": "X", "c": "C", "e": "C"})
_INLINE_NUM_RE: Final = re.compile(r"\s(\d{1,3})\s+(?=[A-Z\"'])")
_VERSE_COUNT_RE: Final = re.compile(r"(\d{1,3})\s+Verses")
_ROMAN: Final = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
_SENT_RE: Final = re.compile(r"(?<=[.;:?!,])\s+(?=[A-Z])")


def roman(s: str) -> int:
    """Convert a Roman numeral (``XCV``) to an int, repairing common OCR slips.

    ``U`` reads as ``II``, lowercase ``n``/``l``/``i`` as ``I`` and ``ex`` as ``CX``.
    """
    s = s.replace("U", "II").translate(_OCR_ROMAN)
    if not s or any(ch not in _ROMAN for ch in s):
        return 0
    total = 0
    for a, b in zip(s, s[1:] + " "):
        v = _ROMAN[a]
        total += -v if b in _ROMAN and _ROMAN[b] > v else v
    return total


@dataclass(frozen=True)
class Verse:
    """One verse with its location.

    Parameters
    ----------
    book : str
        Book name (``Exodus``) or ``Sura``.
    chapter : int
        Chapter or sura number.
    verse : int
        Verse or ayah number.
    text : str
        Verse text as printed in this edition.
    page : int
        1-based page of the document.
    """

    book: str
    chapter: int
    verse: int
    text: str
    page: int

    @property
    def ref(self) -> str:
        """``Exodus 21:24`` or ``Sura 95:4``."""
        return f"{self.book} {self.chapter}:{self.verse}"


def parse_numbered(text: str, books: tuple[str, ...] = BIBLE_BOOKS, page: int = 1) -> list[Verse]:
    """Parse a text that prints ``chapter:verse`` before each verse.

    A reset to ``1:1`` starts the next book in ``books``. O(n) in text length.
    """
    out: list[Verse] = []
    marks = list(_NUMBERED_RE.finditer(text))
    b = -1
    for m, nxt in zip(marks, marks[1:] + [None]):
        ch, vs = int(m.group(1)), int(m.group(2))
        if ch == 1 and vs == 1:
            b += 1
        if b < 0 or b >= len(books):
            continue
        body = text[m.end() : nxt.start() if nxt else len(text)]
        out.append(Verse(books[b], ch, vs, " ".join(body.split()), page))
    return out


def parse_suras(doc: Document) -> list[Verse]:
    """Recover sura:ayah for a Quran translation laid out one verse per line.

    A new verse starts on a line that begins with a capital letter (or with
    a printed verse number); lowercase lines continue the previous verse.
    Printed numbers resynchronize the count; footnotes (lines starting with
    ``*``, ``^`` or a superscript marker, and everything after them on the
    page) are skipped. O(n) in text length.
    """
    out: list[Verse] = []
    sura, n, cur, cur_page = 0, 0, [], 1

    def flush() -> None:
        if sura and cur and n:
            out.append(Verse("Sura", sura, n, " ".join(" ".join(cur).split()), cur_page))

    for page_no, page in enumerate(doc.pages, start=1):
        lines = page.split("\n")
        for k, line in enumerate(lines):
            s = line.strip()
            if not s or (k == 0 and len(s) < 40):  # running head
                continue
            m = _SURA_RE.match(s)
            if m:
                flush()
                sura, n, cur = roman(m.group(1)), 0, []
                sura = sura if 0 < sura <= 114 else 0
                continue
            if s[0] in "*^\u2020\u2021" or re.match(r"^\d\s?[A-Z][a-z]+\.", s) and n > 3:
                break  # footnote block until the end of the page
            if not sura or _VERSE_COUNT_RE.search(s) or s.startswith(("In the Name of God", "[")):
                continue
            # Printed numbers can also sit mid-line where OCR merged two lines.
            pieces, last = [], 0
            for m2 in _INLINE_NUM_RE.finditer(" " + s):
                v = int(m2.group(1))
                if v % 5 == 0 and abs(v - n) <= 15 and v > 0:
                    pieces.append((s[last : max(0, m2.start() - 1)], None))
                    last = m2.end() - 1
                    pieces.append(("", v))
            pieces.append((s[last:], None))
            for text, mark in pieces:
                text = text.strip()
                if mark is not None:
                    flush()
                    n, cur, cur_page = mark - 1, [], page_no
                    continue
                if not text:
                    continue
                num = re.match(r"^(\d{1,3})\s+(?=\S)", text)
                if num and abs(int(num.group(1)) - n) <= 15 and int(num.group(1)) % 5 == 0:
                    flush()
                    n, cur, cur_page = int(num.group(1)), [text[num.end():]], page_no
                elif text[0].isupper() or (n == 0 and text[0].isalpha()):
                    if cur or n:
                        flush()
                    n, cur, cur_page = n + 1, [text], page_no
                elif cur:
                    cur.append(text)
    flush()
    return out


class ReferenceAligner:
    """Assign verse references to unnumbered text by aligning to a numbered edition.

    Each sentence is matched to the reference verse with the largest share of
    IDF-weighted shared terms, with a bonus for verses just after the
    previous match (reading order is monotonic). Building is O(R) for R
    reference verses; each sentence costs O(sum of posting lengths), capped
    by skipping very common terms.
    """

    def __init__(self, reference: list[Verse], normalizer: ArchaicNormalizer | None = None) -> None:
        """Index the reference verses."""
        self.ref = reference
        self.norm = normalizer
        self.post: dict[str, list[int]] = defaultdict(list)
        df: Counter[str] = Counter()
        for i, v in enumerate(reference):
            terms = set(tokenize(v.text))
            df.update(terms)
            for t in terms:
                self.post[t].append(i)
        n = len(reference)
        self.idf = {t: math.log(n / c) for t, c in df.items()}
        self.max_df = max(50, n // 50)

    def match(self, sentence: str, prev: int) -> tuple[int, float]:
        """Best reference index and its score in [0, 1] for one sentence."""
        text = self.norm.text(sentence) if self.norm else sentence
        terms = set(tokenize(text))
        known = [t for t in terms if t in self.idf]
        denom = sum(self.idf[t] for t in known) or 1.0
        acc: dict[int, float] = defaultdict(float)
        for t in known:
            if len(self.post[t]) > self.max_df:
                continue
            for i in self.post[t]:
                acc[i] += self.idf[t]
        for t in known:  # common terms only count near the previous match
            if len(self.post[t]) > self.max_df:
                for i in acc:
                    if 0 <= i - prev <= 40 and t in set(tokenize(self.ref[i].text)):
                        acc[i] += self.idf[t]
        best, best_s = -1, 0.0
        for i, s in acc.items():
            score = s / denom + (0.25 if 0 <= i - prev <= 3 else 0.1 if 0 <= i - prev <= 40 else 0.0)
            if score > best_s:
                best, best_s = i, score
        return best, min(1.0, best_s)

    def align(self, doc: Document, min_score: float = 0.45) -> list[Verse]:
        """Split pages into sentences and tag each with a reference verse."""
        out: list[Verse] = []
        prev = -10
        for page_no, page in enumerate(doc.pages, start=1):
            flat = " ".join(page.split())
            for sent in _SENT_RE.split(flat):
                if len(sent.split()) < 3:
                    if out and out[-1].page == page_no:
                        last = out[-1]
                        out[-1] = Verse(last.book, last.chapter, last.verse, f"{last.text} {sent}", page_no)
                    continue
                i, s = self.match(sent, prev)
                if i < 0 or s < min_score:
                    continue
                r = self.ref[i]
                prev = i
                if out and out[-1].ref == r.ref:
                    last = out[-1]
                    out[-1] = Verse(r.book, r.chapter, r.verse, f"{last.text} {sent}", last.page)
                else:
                    out.append(Verse(r.book, r.chapter, r.verse, sent, page_no))
                # One printed sentence can hold several verses (Exodus 21:23-25).
                terms = set(tokenize(self.norm.text(sent) if self.norm else sent))
                while prev + 1 < len(self.ref) and self.contained(prev + 1, terms):
                    prev += 1
                    nxt = self.ref[prev]
                    if (nxt.book, nxt.chapter) != (r.book, r.chapter):
                        prev -= 1
                        break
                    out.append(Verse(nxt.book, nxt.chapter, nxt.verse, "", page_no))
        return out

    def contained(self, i: int, terms: set[str]) -> bool:
        """True when most of reference verse ``i`` (IDF weighted) is in ``terms``."""
        vt = [t for t in set(tokenize(self.ref[i].text)) if t in self.idf]
        total = sum(self.idf[t] for t in vt)
        return total > 0 and sum(self.idf[t] for t in vt if t in terms) / total >= 0.7


def span_ref(verses: list[Verse]) -> str:
    """``Exodus 21:23-25`` for consecutive verses of one chapter."""
    a, b = verses[0], verses[-1]
    return a.ref if a.verse == b.verse else f"{a.ref}-{b.verse}"


def verse_chunks(book_id: str, verses: list[Verse], target_words: int = 80, overlap_verses: int = 1) -> list[Chunk]:
    """Group verses into chunks that stay inside one chapter.

    O(V) for V verses. Each chunk carries ``ref`` for citations.
    """
    chunks: list[Chunk] = []
    groups: list[list[Verse]] = []
    for v in verses:
        if groups and (groups[-1][-1].book, groups[-1][-1].chapter) == (v.book, v.chapter):
            groups[-1].append(v)
        else:
            groups.append([v])
    for g in groups:
        i = 0
        while i < len(g):
            j, words = i, 0
            while j < len(g) and (words < target_words or j == i):
                words += len(g[j].text.split())
                j += 1
            part = g[i:j]
            idx = len(chunks)
            chunks.append(Chunk(f"{book_id}-{idx:04d}", book_id, " ".join(v.text for v in part if v.text),
                                part[0].page, part[-1].page, idx, idx + 1, span_ref(part)))
            if j >= len(g):
                break
            i = max(i + 1, j - overlap_verses)
    return chunks
