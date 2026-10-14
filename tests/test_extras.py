"""FR-02, FR-09, FR-10, FR-11 tests: IA import, topics, similar passages, eval."""

import json

import numpy as np
import pytest

from scripture_rag import eval as evalmod
from scripture_rag.ia_import import choose_file, fetch_metadata, import_item
from scripture_rag.similar import similar_passages
from scripture_rag.topics import TopicModeler, cluster_keywords, export_theme_map
from tests.conftest import FIXTURES


class _Resp:
    def __init__(self, payload=None, content=b"") -> None:
        self._payload, self.content = payload, content

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _Session:
    def __init__(self, meta) -> None:
        self.meta, self.urls = meta, []

    def get(self, url, timeout):
        self.urls.append(url)
        if "/metadata/" in url:
            return _Resp(self.meta)
        return _Resp(content=b"Page one text.\fPage two text.")


META = {
    "metadata": {"title": "Old Book", "licenseurl": "http://creativecommons.org/publicdomain/mark/1.0/",
                 "possible-copyright-status": "NOT_IN_COPYRIGHT", "date": "1881"},
    "files": [{"name": "oldbook_meta.xml"}, {"name": "oldbook_djvu.txt"}],
}


def test_choose_file_prefers_pdf() -> None:
    assert choose_file([{"name": "a_djvu.txt"}, {"name": "a.pdf"}]) == "a.pdf"
    assert choose_file([{"name": "a.epub"}]) is None


def test_ia_import_mocked() -> None:
    s = _Session(META)
    item = fetch_metadata("oldbook", s)
    assert item.license_fields["possible-copyright-status"] == "NOT_IN_COPYRIGHT"
    assert import_item("oldbook", lambda i: False, s) is None
    doc = import_item("oldbook", lambda i: True, s)
    assert doc is not None and len(doc.pages) == 2 and "NOT_IN_COPYRIGHT" in doc.license_evidence
    assert s.urls[-1] == "https://archive.org/download/oldbook/oldbook_djvu.txt"


def test_ia_errors() -> None:
    with pytest.raises(ValueError):
        fetch_metadata("bad id/../x")
    with pytest.raises(ValueError):
        fetch_metadata("missing", _Session({}))
    with pytest.raises(ValueError):
        import_item("nofile", lambda i: True, _Session({"metadata": {}, "files": []}))


def test_topics_deterministic(pipeline, tmp_path) -> None:
    ids, mat = pipeline.embeddings()
    texts = [pipeline.chunks[i].text for i in ids]
    r1 = TopicModeler(seed=1).fit(mat, texts)
    r2 = TopicModeler(seed=1).fit(mat, texts)
    assert np.array_equal(r1.labels, r2.labels) and r1.coords.shape == (len(ids), 2)
    assert all(r1.keywords.values())
    out = export_theme_map(r1, [pipeline.chunks[i].book_id for i in ids], texts, tmp_path / "map.html")
    assert "kjv_sample" in out.read_text()
    tiny = TopicModeler().fit(mat[:2], texts[:2])
    assert tiny.labels.tolist() == [0, 0]


def test_toy_clusters_separate() -> None:
    a = np.tile([1.0, 0.0], (5, 1)) + np.random.default_rng(0).normal(0, 0.01, (5, 2))
    b = np.tile([0.0, 1.0], (5, 1)) + np.random.default_rng(1).normal(0, 0.01, (5, 2))
    texts = ["water river sea"] * 5 + ["fire flame heat"] * 5
    r = TopicModeler(n_clusters=2).fit(np.vstack([a, b]), texts)
    assert len(set(r.labels[:5])) == 1 and r.labels[0] != r.labels[5] and r.silhouette > 0.9
    assert cluster_keywords(texts, np.array([-1] * 10)) == {}


def test_similar_other_books(pipeline) -> None:
    ids, mat = pipeline.embeddings()
    first = ids[0]
    res = similar_passages(first, ids, mat, pipeline.chunks, k=5)
    assert 0 < len(res) <= 5
    assert all(c.book_id != pipeline.chunks[first].book_id for c, _ in res)


def test_eval_cli(tmp_path) -> None:
    m = evalmod.main(["--corpus", str(FIXTURES / "books"), "--gold", str(FIXTURES / "gold_fixture.jsonl"),
                      "--out", str(tmp_path)])
    assert json.loads((tmp_path / "eval.json").read_text())["refusal_recall"] == m["refusal_recall"]
    assert m["refusal_recall"] >= 0.9 and m["answer_rate"] >= 0.8
    with pytest.raises(SystemExit):
        evalmod.main(["--corpus", str(tmp_path), "--gold", str(FIXTURES / "gold_fixture.jsonl")])
