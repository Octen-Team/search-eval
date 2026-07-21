"""Shared SERP access layer (serpapi.com, engine=google / bing).

Used by both gold_resolver and rubric_gen. Google/Bing act as out-of-band oracles in this
eval system: they are not on the evaluated-competitor list and are only used for gold
candidate discovery and rubric fact grounding.
Prefers the SERPAPI_API_KEY env var; when absent, falls back to: google → Firecrawl search
(FIRECRAWL_API_KEY, Google results), bing → the DuckDuckGo HTML endpoint (Bing-backed index,
no key needed; scraping bing.com directly hits challenge pages — verified 2026-07).
"""
from __future__ import annotations

import html as _html
import os
import re
import time
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlparse

import requests

from .backends import _load_dotenv

_load_dotenv()

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


@dataclass
class SerpItem:
    engine: str
    rank: int
    title: str
    url: str
    snippet: str


def serp_fetch(query: str, engine: str = "google", k: int = 8) -> list[SerpItem]:
    if not os.environ.get("SERPAPI_API_KEY"):
        return _fallback_fetch(query, engine, k)
    resp = None
    for attempt in range(3):  # light backoff for 429/5xx under concurrent rubric generation
        resp = requests.get(
            "https://serpapi.com/search.json",
            params={"engine": engine, "q": query, "num": k, "api_key": os.environ["SERPAPI_API_KEY"]},
            timeout=30,
        )
        if resp.status_code in (429, 500, 502, 503) and attempt < 2:
            time.sleep(3 * (attempt + 1))
            continue
        break
    resp.raise_for_status()
    organic = resp.json().get("organic_results", [])[:k]
    return [
        SerpItem(engine=engine, rank=i + 1, title=r.get("title", ""),
                 url=r.get("link", ""), snippet=r.get("snippet", ""))
        for i, r in enumerate(organic) if r.get("link")
    ]


def _fallback_fetch(query: str, engine: str, k: int) -> list[SerpItem]:
    """Degraded channel when no SERPAPI key: google→Firecrawl, bing→DDG html (Bing-backed index)."""
    if engine == "google":
        key = os.environ.get("FIRECRAWL_API_KEY")
        if not key:
            raise RuntimeError("google oracle unavailable: neither SERPAPI_API_KEY nor "
                               "FIRECRAWL_API_KEY is set")
        resp = requests.post(
            "https://api.firecrawl.dev/v1/search",
            headers={"Authorization": f"Bearer {key}"},
            json={"query": query, "limit": k},
            timeout=30,
        )
        resp.raise_for_status()
        return [
            SerpItem(engine="google", rank=i + 1, title=it.get("title") or "",
                     url=it.get("url") or "", snippet=(it.get("description") or "")[:300])
            for i, it in enumerate((resp.json().get("data") or [])[:k]) if it.get("url")
        ]
    resp = requests.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query},
        headers={"User-Agent": _UA},
        timeout=30,
    )
    resp.raise_for_status()
    text = resp.text
    # positional pairing: each result's snippet is searched only in the segment between this
    # link and the next — index-paired findall used to shift snippets onto wrong URLs whenever
    # one result had no snippet (observed corrupting grounding, 2026-07)
    link_iter = list(re.finditer(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', text))
    out = []
    for i, m in enumerate(link_iter[:k]):
        href, title = m.group(1), m.group(2)
        seg_end = link_iter[i + 1].start() if i + 1 < len(link_iter) else len(text)
        seg = text[m.end():seg_end]
        sm = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', seg, flags=re.S)
        snippet = re.sub(r"<[^>]+>", "", sm.group(1)) if sm else ""
        if "duckduckgo.com/l/" in href:
            href = unquote(parse_qs(urlparse(href).query).get("uddg", [href])[0])
        out.append(SerpItem(engine="bing", rank=i + 1,
                            title=_html.unescape(re.sub(r"<[^>]+>", "", title)).strip(),
                            url=href, snippet=_html.unescape(snippet).strip()[:300]))
    return out


def serp_both(query: str, k: int = 6) -> list[SerpItem]:
    """Google + Bing merged; one engine failing degrades LOUDLY to the other, both failing raises."""
    items, errors = [], []
    for eng in ("google", "bing"):
        try:
            items.extend(serp_fetch(query, eng, k))
        except Exception as e:  # noqa: BLE001
            errors.append(f"{eng}: {e}")
            print(f"WARN serp: {eng} oracle failed ({str(e)[:80]}) — grounding degraded to a single engine",
                  flush=True)
    if not items:
        raise RuntimeError("; ".join(errors))
    return items


def render_grounding(items: list[SerpItem], max_items: int = 10) -> str:
    """Render as the grounding-material text for the rubric generator. Deduped by URL across engines."""
    seen, lines = set(), []
    for it in sorted(items, key=lambda x: (x.rank, x.engine)):
        key = it.url.rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- [{it.engine}#{it.rank}] {it.title}\n  {it.url}\n  {it.snippet}")
        if len(lines) >= max_items:
            break
    return "\n".join(lines) if lines else "(no results)"
