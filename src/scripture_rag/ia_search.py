"""Internet Archive search with polite rate limiting, backoff and a TTL cache (FR-18).

Endpoints (both send ``Access-Control-Allow-Origin: *``):

* Advanced Search: ``https://archive.org/advancedsearch.php`` (paged, ``rows`` and ``page``).
* Metadata: ``https://archive.org/metadata/{identifier}`` (files, sizes, license fields).
* Download: ``https://archive.org/download/{identifier}/{file}``.

Complexity: one search is one HTTP request plus O(r) parsing for r rows; the
cache lookup is O(1). Ingest of the chosen item is O(bytes) download plus the
usual O(p + n) parse and chunk.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

import requests

SEARCH_URL: Final[str] = "https://archive.org/advancedsearch.php"
DETAILS_URL: Final[str] = "https://archive.org/details/{id}"
FIELDS: Final[tuple[str, ...]] = ("identifier", "title", "creator", "date", "language",
                                 "possible-copyright-status", "licenseurl", "item_size",
                                 "description", "subject", "collection")
LANGUAGES: Final[dict[str, str]] = {"en": "(eng OR English)", "pt": "(por OR Portuguese)",
                                    "es": "(spa OR Spanish)"}
TRADITIONS: Final[dict[str, str]] = {
    "christianity": "(Bible OR Christianity OR Gospels OR Theology)",
    "judaism": "(Talmud OR Mishnah OR Judaism OR Torah OR Midrash)",
    "islam": "(Koran OR Quran OR Islam OR Hadith OR Tafsir)",
    "buddhism": "(Buddhism OR Dhammapada OR Sutra OR Pali)",
    "hinduism": "(Hinduism OR Vedas OR Upanishads OR Bhagavadgita)",
    "taoism": "(Taoism OR Tao OR Laozi OR Zhuangzi)",
}
SORTS: Final[dict[str, str]] = {"relevance": "", "date_asc": "date asc", "date_desc": "date desc",
                                "downloads": "downloads desc", "title": "titleSorter asc"}
_RETRY_STATUS: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class SearchHit:
    """One search result.

    Parameters
    ----------
    identifier : str
        IA identifier.
    title : str
        Title.
    creator : str
        Author or creator.
    date : str
        Publication date as listed.
    copyright_status : str
        ``possible-copyright-status`` field, empty when missing.
    license_url : str
        ``licenseurl`` field, empty when missing.
    size_mb : float
        Item size in MB (all files), 0 when unknown.
    description : str
        Description text (trimmed).
    subject : str
        Subjects joined with ``; ``.
    collection : str
        Collections joined with ``; ``.
    """

    identifier: str
    title: str
    creator: str
    date: str
    copyright_status: str
    license_url: str
    size_mb: float
    description: str = ""
    subject: str = ""
    collection: str = ""

    @property
    def preview_url(self) -> str:
        """Item page on archive.org."""
        return DETAILS_URL.format(id=self.identifier)


def _text(v: Any) -> str:
    """Flatten list fields that IA returns for multi-valued metadata."""
    if isinstance(v, list):
        return "; ".join(str(x) for x in v)
    return "" if v is None else str(v)


def _clean(text: str) -> str:
    """Drop Lucene syntax characters from user input."""
    return "".join(ch for ch in text if ch not in '"()[]{}:\\^~*?!').strip()


@dataclass(frozen=True)
class Filters:
    """Search filters (all optional).

    Parameters
    ----------
    language : str or None
        ``en``, ``pt`` or ``es``.
    tradition : str or None
        Key of ``TRADITIONS`` (adds subject terms).
    subject, creator, collection : str
        Field filters.
    year_from, year_to : int or None
        Inclusive publication year range.
    pdf_only : bool
        Only items with a PDF.
    public_domain : bool
        Only ``possible-copyright-status:NOT_IN_COPYRIGHT``.
    sort : str
        Key of ``SORTS``.
    """

    language: str | None = None
    tradition: str | None = None
    subject: str = ""
    creator: str = ""
    collection: str = ""
    year_from: int | None = None
    year_to: int | None = None
    pdf_only: bool = True
    public_domain: bool = False
    sort: str = "relevance"


def build_query(query: str, language: str | None = None, pdf_only: bool = True,
                filters: Filters | None = None) -> str:
    """Compose a Lucene query restricted to texts.

    Parameters
    ----------
    query : str
        User words; syntax characters are removed to keep the query valid.
    language : str or None
        Shortcut for ``filters.language``.
    pdf_only : bool
        Shortcut for ``filters.pdf_only``.
    filters : Filters or None
        Full filter set (overrides the shortcuts).

    Returns
    -------
    str
        Query string.
    """
    f = filters or Filters(language=language, pdf_only=pdf_only)
    words = _clean(query)
    if not words and not (f.tradition or f.subject or f.creator or f.collection):
        raise ValueError("Empty search query")
    parts = [f"({words})"] if words else []
    parts.append("mediatype:texts")
    if f.pdf_only:
        parts.append("format:PDF")
    if f.language:
        parts.append(f"language:{LANGUAGES[f.language]}")
    if f.tradition:
        parts.append(f"subject:{TRADITIONS[f.tradition]}")
    for field_name, value in (("subject", f.subject), ("creator", f.creator), ("collection", f.collection)):
        if _clean(value):
            parts.append(f'{field_name}:"{_clean(value)}"')
    if f.year_from or f.year_to:
        lo = f"{f.year_from:04d}-01-01" if f.year_from else "*"
        hi = f"{f.year_to:04d}-12-31" if f.year_to else "*"
        parts.append(f"date:[{lo} TO {hi}]")
    if f.public_domain:
        parts.append("possible-copyright-status:NOT_IN_COPYRIGHT")
    return " AND ".join(parts)


class IASearch:
    """Rate limited, cached client for Advanced Search.

    Parameters
    ----------
    session : requests.Session or None
        HTTP session (injectable for tests).
    min_interval_s : float
        Minimum gap between requests.
    max_retries : int
        Retries on 429 and 5xx with exponential backoff (honors Retry-After).
    timeout_s : float
        Per request timeout.
    cache_size : int
        LRU entries.
    ttl_s : float
        Cache lifetime.
    clock, sleep : callables
        Injectable for tests.
    """

    def __init__(self, session: requests.Session | None = None, min_interval_s: float = 1.0,
                 max_retries: int = 3, timeout_s: float = 20.0, cache_size: int = 128, ttl_s: float = 3600.0,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> None:
        """Create the client."""
        self.http = session or requests.Session()
        self.http.headers.setdefault("User-Agent", "scripture-rag (+https://github.com/IgorCSIS/scripture-rag)")
        self.min_interval_s, self.max_retries, self.timeout_s = min_interval_s, max_retries, timeout_s
        self.cache: OrderedDict[str, tuple[float, list[SearchHit]]] = OrderedDict()
        self.cache_size, self.ttl_s = cache_size, ttl_s
        self.clock, self.sleep = clock, sleep
        self._last = -1e9

    def __str__(self) -> str:
        """Return a short summary."""
        return f"IASearch(cached={len(self.cache)})"

    def _get(self, url: str, params: dict[str, Any]) -> requests.Response:
        """GET with spacing between calls and backoff on throttling."""
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval_s - (self.clock() - self._last)
            if wait > 0:
                self.sleep(wait)
            self._last = self.clock()
            resp = self.http.get(url, params=params, timeout=self.timeout_s)
            if resp.status_code not in _RETRY_STATUS or attempt == self.max_retries:
                resp.raise_for_status()
                return resp
            retry_after = resp.headers.get("Retry-After", "")
            self.sleep(float(retry_after) if retry_after.isdigit() else delay)
            delay *= 2
        raise RuntimeError("unreachable")

    def search(self, query: str, language: str | None = None, rows: int = 20, page: int = 1,
               pdf_only: bool = True, filters: Filters | None = None) -> list[SearchHit]:
        """Search texts on the Internet Archive.

        Parameters
        ----------
        query : str
            Words to search.
        language : str or None
            Language filter.
        rows : int
            Results per page (capped at 50).
        page : int
            1-based page.
        pdf_only : bool
            Only items that have a PDF.
        filters : Filters or None
            Full filter set, including sort.

        Returns
        -------
        list[SearchHit]
            Results in relevance order.
        """
        q = build_query(query, language, pdf_only, filters)
        sort = SORTS[(filters or Filters()).sort]
        rows = max(1, min(rows, 50))
        key = f"{q}|{sort}|{rows}|{page}"
        hit = self.cache.get(key)
        if hit and self.clock() - hit[0] < self.ttl_s:
            self.cache.move_to_end(key)
            return hit[1]
        params: dict[str, Any] = {"q": q, "fl[]": list(FIELDS), "rows": rows, "page": page, "output": "json"}
        if sort:
            params["sort[]"] = sort
        docs = self._get(SEARCH_URL, params).json().get("response", {}).get("docs", [])
        out = [SearchHit(_text(d.get("identifier")), _text(d.get("title")), _text(d.get("creator")),
                         _text(d.get("date"))[:10], _text(d.get("possible-copyright-status")),
                         _text(d.get("licenseurl")), round(float(d.get("item_size") or 0) / 1e6, 1),
                         _text(d.get("description"))[:500], _text(d.get("subject")), _text(d.get("collection")))
               for d in docs if d.get("identifier")]
        self.cache[key] = (self.clock(), out)
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return out
