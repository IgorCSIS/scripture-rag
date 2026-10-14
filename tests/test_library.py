"""Role tags, compare mode, outliers and the manifest."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripture_rag.config import Settings
from scripture_rag.ingest import Document
from scripture_rag.library import (
    ask_by_role,
    compare,
    manifest,
    manifest_export,
    outliers,
    set_role,
    suggest_role,
    suggest_tradition,
)
from scripture_rag.pipeline import RagPipeline


@pytest.mark.parametrize(("title", "subject", "role"), [
    ("The Koran, translated by J. M. Rodwell", "Koran", "primary"),
    ("A comprehensive commentary on the Quran", "Koran -- Commentaries", "commentary"),
    ("New edition of the Babylonian Talmud", "Talmud", "primary"),
    ("The history of the Talmud", "", "commentary"),
    ("Holy Bible", "", "primary"),
])
def test_suggest_role(title: str, subject: str, role: str) -> None:
    assert suggest_role(title, subject) == role


def test_suggest_tradition() -> None:
    assert suggest_tradition("The Koran", "Islam") == "islam"
    assert suggest_tradition("Babylonian Talmud") == "judaism"
    assert suggest_tradition("Cookbook") == ""


def _pipe() -> RagPipeline:
    pipe = RagPipeline(Settings(refusal_threshold=0.05, chunk_tokens=40, chunk_overlap=5))
    pipe.add_document(Document("bible", "Bible", ["The Lord is my shepherd; I shall not want. He maketh me to lie "
                                                  "down in green pastures."], "archive.org/details/x",
                               "possible-copyright-status=NOT_IN_COPYRIGHT", "primary", "christianity", "1611"))  # pragma: allowlist secret
    pipe.add_document(Document("notes", "Notes on the Psalms", ["The shepherd image in the psalm describes care "
                                                                "and guidance for the flock."], "", "",
                               "commentary", "christianity", "1880"))
    pipe.add_document(Document("koran", "Koran", ["Praise be to God, Lord of the worlds, the compassionate, the "
                                                  "merciful."], "", "", "primary", "islam"))
    return pipe


def test_role_separation_and_compare() -> None:
    pipe = _pipe()
    by_role = ask_by_role(pipe, "Who is my shepherd?")
    assert set(by_role) == {"primary", "commentary"}
    assert all(c.book_id != "notes" for c in by_role["primary"].citations)
    assert all(c.book_id == "notes" for c in by_role["commentary"].citations)
    cmp = compare(pipe, "Who is my shepherd?", by="tradition")
    assert set(cmp) == {"christianity", "islam"}
    assert all(c.book_id in {"bible", "notes"} for c in cmp["christianity"].citations)
    assert set(compare(pipe, "shepherd")) == {"bible", "notes", "koran"}
    with pytest.raises(ValueError):
        set_role(pipe.documents["bible"], "fiction")
    set_role(pipe.documents["bible"], "commentary")
    assert pipe.documents["bible"].role == "commentary"


def test_outliers_flags_far_point() -> None:
    rng = np.random.default_rng(0)
    vecs = np.vstack([rng.normal(0, 0.05, (30, 4)), [[3.0, 3.0, 3.0, 3.0]]])
    labels = np.zeros(31, dtype=int)
    out = outliers(vecs, labels, 2.5)
    assert out and out[0].index == 30 and out[0].z > 2.5
    assert outliers(vecs[:2], labels[:2]) == []


def test_manifest_rows_and_exports() -> None:
    pipe = _pipe()
    rows = manifest(pipe)
    bible = next(r for r in rows if r["book_id"] == "bible")
    assert bible["source_url"] == "https://archive.org/details/x" and bible["role"] == "primary"
    assert bible["license_status"].startswith("possible-copyright-status")
    assert next(r for r in rows if r["book_id"] == "koran")["license_status"] == "not recorded"
    assert len(json.loads(manifest_export(pipe))) == 3
    assert manifest_export(pipe, "csv").splitlines()[0].startswith("book_id,title,role")
