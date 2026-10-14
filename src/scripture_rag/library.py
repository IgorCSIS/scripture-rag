"""Library metadata: role tags, compare mode, outliers and the manifest (FR-20 to FR-23).

* ``suggest_role``: O(t) keyword scan of title and subject.
* ``compare``: one ``ask`` per group, so O(G * ask) for G books or traditions.
* ``outliers``: z-score of each passage's distance to its cluster centroid,
  O(n * d) after clustering.
* ``manifest``: O(B) rows for B books.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from .ingest import Document

if TYPE_CHECKING:
    from .pipeline import Answer, RagPipeline

ROLES: Final[tuple[str, ...]] = ("primary", "commentary")
_COMMENTARY_RE: Final = re.compile(
    r"\b(commentar\w*|tafsir|exegesis|notes on|annotat\w*|introduction to|history of|essays?|studies|"
    r"scholar\w*|criticism|critical|discourse|dictionary|concordance|handbook|lectures?|treatise|"
    r"companion|guide to|interpretation|theology|sermons?)\b", re.I)
_PRIMARY_RE: Final = re.compile(
    r"\b(bible|koran|qur.?an|alcoran|talmud|mishna\w*|torah|gospels?|psalms|dhammapada|upanishads?|"
    r"bhagavad\w*|tao te ching|sutras?|vedas?|scriptures?)\b", re.I)
TRADITION_WORDS: Final[dict[str, str]] = {
    "christianity": r"bible|gospel|testament|christian|psalm",
    "judaism": r"talmud|mishna|torah|midrash|jew|hebrew",
    "islam": r"koran|qur.?an|alcoran|islam|hadith|tafsir|sura|muhammad",
    "buddhism": r"buddh|dhamma|sutta|pali",
    "hinduism": r"veda|upanishad|bhagavad|hindu",
    "taoism": r"\btao\b|laozi|lao.tzu|chuang",
}


def suggest_role(title: str, subject: str = "") -> str:
    """Suggest ``primary`` or ``commentary`` from title and subject words.

    Commentary cues win over scripture names, because "A commentary on the
    Quran" names the Quran but is not the Quran.
    """
    text = f"{title} {subject}"
    if _COMMENTARY_RE.search(text):
        return "commentary"
    return "primary" if _PRIMARY_RE.search(text) or not text.strip() else "commentary"


def suggest_tradition(title: str, subject: str = "") -> str:
    """Best tradition label by keyword hits, or empty."""
    text = f"{title} {subject}".lower()
    scores = {k: len(re.findall(p, text)) for k, p in TRADITION_WORDS.items()}
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] else ""


def set_role(doc: Document, role: str) -> None:
    """User override of the suggested role."""
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    doc.role = role


def books_with(pipe: RagPipeline, role: str | None = None, tradition: str | None = None) -> set[str]:
    """Book ids matching role and tradition filters."""
    return {b for b, d in pipe.documents.items()
            if (role is None or d.role == role) and (tradition is None or d.tradition == tradition)}


def ask_by_role(pipe: RagPipeline, question: str) -> dict[str, Answer]:
    """Answer separately from primary texts and from commentary, never mixing them."""
    out = {}
    for role in ROLES:
        books = books_with(pipe, role)
        if books:
            out[role] = pipe.ask(question, books)
    return out


def compare(pipe: RagPipeline, question: str, by: str = "book") -> dict[str, Answer]:
    """One question, one grounded answer per book or per tradition.

    Parameters
    ----------
    pipe : RagPipeline
        Loaded pipeline.
    question : str
        Question.
    by : str
        ``book`` or ``tradition``.

    Returns
    -------
    dict[str, Answer]
        Group label to answer (each answer cites only its group).
    """
    groups: dict[str, set[str]] = {}
    for b, d in pipe.documents.items():
        key = b if by == "book" else (d.tradition or "unlabeled")
        groups.setdefault(key, set()).add(b)
    return {k: pipe.ask(question, books) for k, books in sorted(groups.items())}


@dataclass(frozen=True)
class Outlier:
    """A passage unusually far from its cluster centroid."""

    index: int
    cluster: int
    distance: float
    z: float


def outliers(vectors: np.ndarray, labels: np.ndarray, z_threshold: float = 2.5) -> list[Outlier]:
    """Flag passages whose distance to their centroid has z-score above a threshold.

    The z-score is computed within each cluster (clusters of size < 3 are
    skipped). O(n * d) time, O(k * d) extra memory.
    """
    found: list[Outlier] = []
    for c in sorted(set(labels.tolist())):
        idx = np.flatnonzero(labels == c)
        if c < 0 or len(idx) < 3:
            continue
        cen = vectors[idx].mean(axis=0)
        dist = np.linalg.norm(vectors[idx] - cen, axis=1)
        sd = float(dist.std())
        if sd == 0:
            continue
        z = (dist - dist.mean()) / sd
        found += [Outlier(int(i), int(c), float(d), float(s)) for i, d, s in zip(idx, dist, z) if s > z_threshold]
    return sorted(found, key=lambda o: -o.z)


MANIFEST_FIELDS: Final[tuple[str, ...]] = ("book_id", "title", "role", "tradition", "date", "source_url",
                                           "license_status", "pages", "passages")


def manifest(pipe: RagPipeline) -> list[dict[str, str | int]]:
    """One audit row per loaded document."""
    counts: dict[str, int] = {}
    for c in pipe.chunks.values():
        counts[c.book_id] = counts.get(c.book_id, 0) + 1
    rows = []
    for b, d in pipe.documents.items():
        src = d.source if not d.source or d.source.startswith("http") else f"https://{d.source}"
        rows.append({"book_id": b, "title": d.title, "role": d.role, "tradition": d.tradition, "date": d.date,
                     "source_url": src, "license_status": d.license_evidence or "not recorded",
                     "pages": len(d.pages), "passages": counts.get(b, 0)})
    return rows


def manifest_export(pipe: RagPipeline, fmt: str = "json") -> str:
    """Manifest as JSON or CSV text."""
    rows = manifest(pipe)
    if fmt == "json":
        return json.dumps(rows, indent=2, ensure_ascii=False)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(MANIFEST_FIELDS))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()
