"""Pluggable search backends.

All backends emit a unified list of SearchResult so the judge sees an identical format
(blind-test fairness). To add a competitor, implement a SearchBackend subclass and
register it in BACKENDS.

API keys are injected via environment variables: OCTEN_API_KEY / EXA_API_KEY / BRAVE_API_KEY /
TAVILY_API_KEY / PERPLEXITY_API_KEY / PARALLEL_API_KEY

Speed-tier note: the low-latency variants (tavily-ultrafast, exa-instant, parallel-turbo) are
registered as distinct names so a run comparing them against octen is explicit in run_meta. They
subclass the standard backend and only pin the tier, so the ONLY variable vs. the base backend is
the latency mode.
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import requests


def _load_dotenv() -> None:
    """Load .env from the project root (without overriding existing env vars). No third-party deps."""
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


_load_dotenv()


@dataclass
class SearchResult:
    rank: int
    title: str
    url: str
    snippet: str
    published_date: str | None = None  # ISO8601, used when judging freshness
    raw: dict = field(default_factory=dict, repr=False)  # raw API payload, used during attribution


@dataclass
class SearchResponse:
    backend: str
    query: str
    results: list[SearchResult]
    latency_ms: float                         # client-measured round-trip (network/TLS + server)
    error: str | None = None
    reported_latency_ms: float | None = None  # server-side time the backend reports, if any —
                                              # the fair engine-speed measure (octen meta.latency,
                                              # exa searchTime, tavily response_time). None = not reported.

    def to_judge_text(self, k: int = 10) -> str:
        """Render as the result-list text for the judge prompt."""
        lines = []
        for r in self.results[:k]:
            date = f" | published: {r.published_date}" if r.published_date else ""
            lines.append(f"[{r.rank}] {r.title}\n    URL: {r.url}{date}\n    Snippet: {r.snippet}")
        return "\n\n".join(lines) if lines else "(no results)"


def response_from_record(rec: dict, backend: str | None = None) -> SearchResponse:
    """Rebuild a SearchResponse from a stored responses.jsonl record, so cached run artifacts
    can be re-judged offline (calibration experiments) without new backend calls."""
    results = [SearchResult(rank=r["rank"], title=r["title"], url=r["url"], snippet=r["snippet"],
                            published_date=r.get("published_date"))
               for r in rec.get("results", [])]
    return SearchResponse(backend=backend or rec["backend"], query=rec.get("query", ""),
                          results=results, latency_ms=rec.get("latency_ms", 0.0),
                          error=rec.get("error"), reported_latency_ms=rec.get("reported_latency_ms"))


class SearchBackend(ABC):
    name: str = "base"

    @abstractmethod
    def _search(self, query: str, k: int) -> list[SearchResult]: ...

    def search(self, query: str, k: int = 10, retries: int = 2) -> SearchResponse:
        last_err = None
        for attempt in range(retries + 1):
            t0 = time.perf_counter()
            try:
                self._reported_latency_ms = None  # _search may set this from the backend payload
                results = self._search(query, k)
                latency = (time.perf_counter() - t0) * 1000
                return SearchResponse(self.name, query, results, latency,
                                      reported_latency_ms=self._reported_latency_ms)
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
                resp = getattr(e, "response", None)
                status = getattr(resp, "status_code", None)
                if status is not None and status != 429 and 400 <= status < 500:
                    break  # auth/validation errors are not transient — fail fast, don't burn retries
                if attempt < retries:  # no pointless sleep after the final attempt
                    ra = resp.headers.get("retry-after", "") if resp is not None else ""
                    delay = float(ra) if str(ra).replace(".", "", 1).isdigit() else 1.5 * (attempt + 1)
                    time.sleep(min(delay, 30))
        return SearchResponse(self.name, query, [], 0.0, error=last_err)


class OctenBackend(SearchBackend):
    """The system under test. Calibrated against the real API (2026-07): POST /search, response wrapped in a data layer.

    Result fields: title / url / highlight / full_content / authors /
                   time_published / time_last_crawled / favicon
    time_last_crawled is kept in raw — directly usable when attributing INDEX_STALE.
    """

    name = "octen"
    topic: str | None = None  # e.g. "news" — routes to the engine's topical index

    def __init__(self):
        self.api_key = os.environ["OCTEN_API_KEY"]
        self.endpoint = os.environ.get("OCTEN_ENDPOINT", "https://api.octen.ai/search")

    def _search(self, query: str, k: int) -> list[SearchResult]:
        body = {"query": query, "count": k}
        # single-scaffold parity knob: request full-length 2048-token highlight + full_content
        # (the default bare request gets shorter API-default highlights). Unset (default) keeps
        # the historical request shape (latency numbers depend on it).
        mt = os.environ.get("OCTEN_CONTENT_MAX_TOKENS")
        if mt:
            body["highlight"] = {"enable": True, "max_tokens": int(mt)}
            body["full_content"] = {"enable": True, "max_tokens": int(mt)}
        if self.topic:
            body["topic"] = self.topic
        resp = requests.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=body,
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        items = payload.get("data", {}).get("results", [])
        rl = (payload.get("meta") or {}).get("latency")  # server-side processing time (ms)
        self._reported_latency_ms = float(rl) if isinstance(rl, (int, float)) else None
        return [
            SearchResult(
                rank=i + 1,
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=(item.get("highlight") or item.get("full_content") or "")[:500],
                published_date=item.get("time_published") or None,
                raw=item,
            )
            for i, item in enumerate(items)
        ]


class ExaBackend(SearchBackend):
    """Verified against the real API (2026-07). Note: the occasional result genuinely carries
    title="" in the raw response (e.g. PDF documents with no HTML title); url/text are still
    populated, so this is accepted as a real trait of the source.
    """

    name = "exa"
    search_type: str | None = None  # None → API default ('auto'); subclasses pin a latency tier

    def __init__(self):
        self.api_key = os.environ["EXA_API_KEY"]

    def _contents(self) -> dict:
        # Full-text extraction, capped. The instant tier overrides this to request lighter,
        # query-relevant highlights instead (see ExaInstantBackend).
        return {"text": {"maxCharacters": 500}}

    def _search(self, query: str, k: int) -> list[SearchResult]:
        body = {"query": query, "numResults": k, "contents": self._contents()}
        if self.search_type:
            body["type"] = self.search_type
        resp = requests.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": self.api_key},
            json=body,
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        st = payload.get("searchTime")  # Exa reports server-side search time in ms
        self._reported_latency_ms = float(st) if isinstance(st, (int, float)) else None
        return [
            SearchResult(
                rank=i + 1,
                title=item.get("title") or "",
                url=item.get("url", ""),
                # highlights (extractive snippets) when present, else full text — one mapping
                # serves both the standard 'text' tier and the instant 'highlights' tier.
                snippet=(" … ".join(item.get("highlights") or []) or item.get("text") or "")[:500],
                published_date=item.get("publishedDate"),
                raw=item,
            )
            for i, item in enumerate(payload.get("results", []))
        ]


class ExaInstantBackend(ExaBackend):
    """Exa's 'instant' tier (type='instant'): the lowest-latency Exa mode, marketed sub-180ms.
    We request highlights rather than full-page text — extractive, query-relevant snippets are the
    content mode appropriate to a latency tier (full-text extraction would inflate the round-trip
    and misrepresent what the 'instant' product delivers). Snippet parity with the other backends
    is preserved via the shared highlights→text mapping in ExaBackend."""

    name = "exa-instant"
    search_type = "instant"

    def _contents(self) -> dict:
        # highlights supports numSentences (not maxCharacters — that's a `text` param); the
        # snippet is length-capped client-side by the [:500] join in ExaBackend._search.
        return {"highlights": {"numSentences": 3}}


class BraveBackend(SearchBackend):
    name = "brave"

    def __init__(self):
        self.api_key = os.environ["BRAVE_API_KEY"]

    def _search(self, query: str, k: int) -> list[SearchResult]:
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": self.api_key},
            params={"q": query, "count": k},
            timeout=30,
        )
        resp.raise_for_status()
        items = resp.json().get("web", {}).get("results", [])
        return [
            SearchResult(
                rank=i + 1,
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
                published_date=item.get("page_age"),
                raw=item,
            )
            for i, item in enumerate(items)
        ]


class TavilyBackend(SearchBackend):
    """Verified against the real API (2026-07): max_results is honored, but basic-depth search
    often returns fewer hits than requested (e.g. 7 of 10). We over-request and truncate to k —
    Tavily bills per request, not per result, so this is free.

    This source has no date field (general topic returns only content/raw_content/score/title/url;
    dates exist only under topic=news).
    """

    name = "tavily"
    search_depth: str | None = None  # None → API default ('basic'); subclasses pin a latency tier

    def __init__(self):
        self.api_key = os.environ["TAVILY_API_KEY"]

    def _search(self, query: str, k: int) -> list[SearchResult]:
        body = {"api_key": self.api_key, "query": query,
                "max_results": min(max(k * 2, k + 5), 20)}
        if self.search_depth:
            body["search_depth"] = self.search_depth
        resp = requests.post(
            "https://api.tavily.com/search",
            json=body,
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        rt = payload.get("response_time")  # Tavily reports server-side time in SECONDS → ms
        self._reported_latency_ms = float(rt) * 1000 if isinstance(rt, (int, float)) else None
        return [
            SearchResult(
                rank=i + 1,
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=(item.get("content") or "")[:500],
                published_date=item.get("published_date"),
                raw=item,
            )
            for i, item in enumerate(payload.get("results", [])[:k])
        ]


class TavilyUltraFastBackend(TavilyBackend):
    """Tavily's 'ultra-fast' search depth: the lowest-latency tier, marketed sub-second. Like the
    default basic depth it returns one NLP summary per URL and (on the general topic) no
    published_date. Costs 1 credit; safe_search/chunks_per_source are unsupported at this depth.
    Verified against docs 2026-07."""

    name = "tavily-ultrafast"
    search_depth = "ultra-fast"


class PerplexityBackend(SearchBackend):
    """Perplexity Search API (the native search endpoint, not the sonar chat-completions Q&A).

    Verified against the real API (2026-07): field mapping is correct, but a few results genuinely
    come back with an empty snippet (raw item has snippet="" and no alternative content field).
    This is a real quality trait of the source and is left as is — the judge scores it under
    snippet_quality.
    """

    name = "perplexity"

    def __init__(self):
        self.api_key = os.environ["PERPLEXITY_API_KEY"]

    def _search(self, query: str, k: int) -> list[SearchResult]:
        resp = requests.post(
            "https://api.perplexity.ai/search",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"query": query, "max_results": k},
            timeout=30,
        )
        resp.raise_for_status()
        return [
            SearchResult(
                rank=i + 1,
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=(item.get("snippet") or "")[:500],
                published_date=item.get("date") or item.get("last_updated"),
                raw=item,
            )
            for i, item in enumerate(resp.json().get("results", []))
        ]


class ParallelBackend(SearchBackend):
    """Parallel Web Systems Search API (POST /v1/search, x-api-key auth). Results carry an array of
    markdown `excerpts` rather than a single snippet; we join them and cap at 500 chars for parity
    with the other backends. `mode` defaults to the API's 'advanced'; subclasses pin a faster tier.

    The eval query is passed verbatim as the single search_query AND as the natural-language
    objective. Parallel's docs recommend 3-6-word keyword queries, so long NL questions sit outside
    its ideal input shape — this is the honest 'every backend gets the same query' mapping, flagged
    as a methodology caveat rather than silently reshaped. Verified against docs 2026-07.
    """

    name = "parallel"
    mode: str | None = None  # None → API default ('advanced'); ParallelTurboBackend pins 'turbo'

    def __init__(self):
        self.api_key = os.environ["PARALLEL_API_KEY"]
        self.endpoint = os.environ.get("PARALLEL_ENDPOINT", "https://api.parallel.ai/v1/search")

    def _search(self, query: str, k: int) -> list[SearchResult]:
        body = {
            "objective": query,
            "search_queries": [query],
            "advanced_settings": {"max_results": k,
                                  "excerpt_settings": {"max_chars_per_result": int(
                                      os.environ.get("PARALLEL_EXCERPT_MAX_CHARS", "500"))}},
        }
        if self.mode:
            body["mode"] = self.mode
        resp = requests.post(
            self.endpoint,
            headers={"x-api-key": self.api_key},
            json=body,
            timeout=30,
        )
        resp.raise_for_status()
        items = resp.json().get("results", [])
        return [
            SearchResult(
                rank=i + 1,
                title=item.get("title") or "",
                url=item.get("url", ""),
                snippet=(" … ".join(item.get("excerpts") or []))[:500],
                published_date=item.get("publish_date") or None,
                raw=item,
            )
            for i, item in enumerate(items[:k])
        ]


class ParallelTurboBackend(ParallelBackend):
    """Parallel's 'turbo' search mode: the low-latency tier marketed for voice/chat agents."""

    name = "parallel-turbo"
    mode = "turbo"


class OctenNewsBackend(OctenBackend):
    """octen routed to its news topical index (topic=news). A separate registered name keeps
    run_meta explicit — never mix topic and default runs under one label."""
    name = "octen-news"
    topic = "news"


BACKENDS: dict[str, type[SearchBackend]] = {
    "octen": OctenBackend,
    "octen-news": OctenNewsBackend,
    "exa": ExaBackend,
    "exa-instant": ExaInstantBackend,
    "brave": BraveBackend,
    "tavily": TavilyBackend,
    "tavily-ultrafast": TavilyUltraFastBackend,
    "perplexity": PerplexityBackend,
    "parallel": ParallelBackend,
    "parallel-turbo": ParallelTurboBackend,
}


def get_backend(name: str) -> SearchBackend:
    return BACKENDS[name]()
