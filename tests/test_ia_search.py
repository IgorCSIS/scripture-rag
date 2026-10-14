"""Internet Archive search: query building, rate limiting, backoff, cache, size cap."""

from __future__ import annotations

import os
from typing import Any

import pytest

from scripture_rag import ia_import
from scripture_rag.ia_search import IASearch, build_query


class FakeResp:
    def __init__(self, status: int = 200, data: Any = None, headers: dict[str, str] | None = None,
                 content: bytes = b"") -> None:
        self.status_code, self._data, self.headers, self.content = status, data, headers or {}, content

    def json(self) -> Any:
        return self._data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses: list[FakeResp]) -> None:
        self.responses, self.calls, self.headers = list(responses), [], {}

    def get(self, url: str, params: Any = None, timeout: float = 0) -> FakeResp:
        self.calls.append((url, params, timeout))
        return self.responses.pop(0)


DOCS = {"response": {"docs": [{"identifier": "koranrodw00rodwuoft", "title": "The Koran", "creator": ["Rodwell"],
                               "date": "1915-01-01T00:00:00Z", "possible-copyright-status": "NOT_IN_COPYRIGHT",
                               "item_size": 27000000}]}}


class Clock:
    def __init__(self) -> None:
        self.t, self.slept = 0.0, []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.t += s


def test_build_query_filters_and_sanitizes() -> None:
    q = build_query('koran "rodwell" (1861)', "pt")
    assert q.startswith("(koran rodwell 1861)") and "mediatype:texts" in q and "format:PDF" in q
    assert "language:(por OR Portuguese)" in q
    with pytest.raises(ValueError):
        build_query('  ""  ')


def test_search_parses_caches_and_spaces_requests() -> None:
    clock = Clock()
    sess = FakeSession([FakeResp(data=DOCS), FakeResp(data=DOCS)])
    ia = IASearch(sess, min_interval_s=1.0, clock=clock.now, sleep=clock.sleep)
    hits = ia.search("koran")
    assert hits[0].creator == "Rodwell" and hits[0].date == "1915-01-01" and hits[0].size_mb == 27.0
    assert hits[0].preview_url.endswith("/details/koranrodw00rodwuoft")
    assert ia.search("koran") == hits and len(sess.calls) == 1  # cached
    ia.search("koran", "en")
    assert len(sess.calls) == 2 and clock.slept == [1.0]
    assert sess.calls[0][2] == 20.0


def test_rate_limit_waits_between_calls() -> None:
    clock = Clock()
    sess = FakeSession([FakeResp(data=DOCS), FakeResp(data=DOCS)])
    ia = IASearch(sess, min_interval_s=1.0, clock=clock.now, sleep=clock.sleep)
    ia.search("a")
    ia.search("b")
    assert clock.slept == [1.0]


def test_backoff_on_429_then_success() -> None:
    clock = Clock()
    sess = FakeSession([FakeResp(429, headers={"Retry-After": "3"}), FakeResp(503), FakeResp(data=DOCS)])
    ia = IASearch(sess, min_interval_s=0, clock=clock.now, sleep=clock.sleep)
    assert len(ia.search("koran")) == 1
    assert clock.slept == [3.0, 2.0]  # Retry-After, then doubled backoff


def test_backoff_gives_up() -> None:
    clock = Clock()
    sess = FakeSession([FakeResp(503)] * 3)
    ia = IASearch(sess, min_interval_s=0, max_retries=2, clock=clock.now, sleep=clock.sleep)
    with pytest.raises(RuntimeError):
        ia.search("koran")


def test_import_rejects_oversize_file() -> None:
    meta = {"metadata": {"title": "Big", "possible-copyright-status": "NOT_IN_COPYRIGHT"},
            "files": [{"name": "big.pdf", "size": str(80 * 1024 * 1024)}]}
    sess = FakeSession([FakeResp(data=meta)])
    with pytest.raises(ValueError, match="limit"):
        ia_import.import_item("bigitem", lambda _i: True, session=sess, max_mb=50)
    assert len(sess.calls) == 1  # no download attempted


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("LIVE_IA") != "1", reason="set LIVE_IA=1 for the live smoke test")
def test_live_search_smoke() -> None:
    hits = IASearch().search("rodwell koran", "en", rows=5)
    assert hits and all(h.identifier for h in hits)


def test_filters_build_full_query() -> None:
    from scripture_rag.ia_search import Filters

    f = Filters(language="es", tradition="judaism", creator="Rodkinson", collection="americana",
                year_from=1800, year_to=1925, public_domain=True, sort="date_asc")
    q = build_query("talmud", filters=f)
    for part in ("subject:(Talmud", 'creator:"Rodkinson"', 'collection:"americana"', "date:[1800-01-01 TO 1925-12-31]",
                 "possible-copyright-status:NOT_IN_COPYRIGHT", "language:(spa"):
        assert part in q
    sess = FakeSession([FakeResp(data=DOCS)])
    IASearch(sess, min_interval_s=0).search("talmud", filters=f)
    assert sess.calls[0][1]["sort[]"] == "date asc"
    assert build_query("", filters=Filters(tradition="islam")).startswith("mediatype:texts")


def test_discovery_cluster_related_tags() -> None:
    import numpy as np

    from scripture_rag.discover import book_tags, cluster_hits, library_centroid, rank_related, related_query
    from scripture_rag.embed import HashingEmbedder
    from scripture_rag.ia_search import SearchHit
    from scripture_rag.topics import TopicResult

    def hit(i: str, t: str, s: str) -> SearchHit:
        return SearchHit(i, t, "", "", "", "", 0.0, "", s)

    hits = [hit("a", "Babylonian Talmud tract Sabbath", "Talmud"), hit("b", "Talmud Mishnah Sabbath laws", "Talmud"),
            hit("c", "Mishnah tract Sabbath Talmud", "Talmud"), hit("d", "Koran sura translation", "Koran"),
            hit("e", "Koran commentary sura notes", "Koran"), hit("f", "Koran sura Mecca Medina", "Koran")]
    emb = HashingEmbedder()
    groups = cluster_hits(hits, emb, k=2)
    assert sum(len(g.hits) for g in groups) == 6 and all(g.label for g in groups)
    ids = [{h.identifier for h in g.hits} for g in groups]
    assert {"a", "b", "c"} in ids and {"d", "e", "f"} in ids
    lib = ["the talmud tract sabbath mishnah"] * 3
    cen = library_centroid(emb.encode(lib))
    ranked = rank_related(hits, emb, cen, exclude={"a"})
    assert ranked[0][0].identifier in {"b", "c"} and all(h.identifier != "a" for h, _ in ranked)
    assert "talmud" in related_query(lib)
    res = TopicResult(np.array([0, 0, 1]), {0: ["sabbath", "law"], 1: ["sura", "mecca"]})
    assert book_tags(["t", "t", "k"], res) == {"t": ["sabbath / law"], "k": ["sura / mecca"]}
    assert cluster_hits([], emb) == []
