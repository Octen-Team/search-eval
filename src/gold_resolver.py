"""Anchor gold-URL resolution: SERP candidates → rule filter → LLM adjudication → HTTP liveness.

Design principles (read before touching):
- Google SERP is only a candidate generator, never the final arbiter: top-1 can be an ad,
  an aggregator, or SEO-boosted content.
- The LLM may only pick from the candidate list — it must not produce URLs of its own
  (eliminates hallucination).
- The gold's provenance is written into meta.gold_source, keeping it traceable and auditable.
- difficulty=tail queries stay needs_review=true even after resolution; human spot-checks backstop.
- Google is not on the evaluated-competitor list; if SERP-wrapper competitors join later,
  watch for source bias when assessing the anchor set.

Environment variables:
  SERPAPI_API_KEY   serpapi.com key (engine=google / bing)
  LLM adjudication goes through src/llm.py (shared provider switch with the judge)

Candidate discovery: Google + Bing merged (via src/serp.py); both engines' organic results
enter the candidate pool, then rule filter + LLM adjudication + HTTP liveness handle the rest.

Usage:
  python -m src.gold_resolver --queries data/synth_xxx.jsonl [--out ...]
Only processes entries with intent=navigational and no verified gold yet; rewrites in place.
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests

from .backends import _load_dotenv
from .common import load_jsonl, today_utc, write_jsonl_atomic
from .llm import call_llm_json
from .serp import serp_both

_load_dotenv()

# Aggregator / second-hand content domain blacklist (a navigational gold must not land on these)
AGGREGATOR_BLACKLIST = {
    "medium.com", "zhihu.com", "zhuanlan.zhihu.com", "csdn.net", "blog.csdn.net",
    "juejin.cn", "cnblogs.com", "jianshu.com", "reddit.com", "quora.com",
    "stackoverflow.com", "segmentfault.com", "51cto.com", "baidu.com",
    "wikipedia.org",  # Wikipedia is a fine source but not an "official target page"
}


def _domain(url: str) -> str:
    return urlparse(url).netloc.removeprefix("www.").lower()


def rule_filter(urls: list[str]) -> list[str]:
    seen, out = set(), []
    for u in urls:
        d = _domain(u)
        if d in AGGREGATOR_BLACKLIST or any(d.endswith("." + b) for b in AGGREGATOR_BLACKLIST):
            continue
        key = u.rstrip("/")
        if key not in seen:
            seen.add(key)
            out.append(u)
    return out


def llm_adjudicate(query: str, candidates: list[str]) -> list[str]:
    """The LLM picks only official/original target pages from the candidates; returns 0-3. Empty list = no qualified candidate."""
    numbered = "\n".join(f"{i+1}. {u}" for i, u in enumerate(candidates))
    system = (
        "You are a search evaluation expert. Given a navigational search query and a list of "
        "candidate URLs, pick the URLs that truly are the query's official/original target pages "
        "(official site, official docs, original paper page, official repository). "
        "You may only choose from the candidate list — never output a URL outside it. "
        "Second-hand retellings, mirror sites, and aggregator pages are never picked. "
        "If no candidate qualifies, return an empty array — better none than wrong. "
        'Output JSON only: {"gold": ["url1", ...]} (0-3 items, ordered by confidence)'
    )
    user = f"Query: {query}\n\nCandidates:\n{numbered}"
    # via src/llm.py provider switch (shared with judge/rubric_gen)
    picked = call_llm_json(system, user, model=os.environ.get("RUBRIC_MODEL", "claude-sonnet-4-6"),
                           max_tokens=512).get("gold", [])
    cand_set = {u.rstrip("/") for u in candidates}
    return [u for u in picked if u.rstrip("/") in cand_set][:3]  # hard constraint: candidates only


_LIVENESS_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


def http_alive(url: str) -> bool:
    """Liveness check. Government/institutional sites often bot-block with 403 (observed 2026-07:
    pubmed/FDA/NMPA/HKEX all 403 on HEAD) — a 403 means protected, not dead, so it counts as alive;
    genuinely removed pages return 404/410."""
    try:
        r = requests.head(url, timeout=10, allow_redirects=True,
                          headers={"User-Agent": _LIVENESS_UA})
        if r.status_code in (403, 405, 429):  # HEAD unsupported or bot-challenged → retry with GET
            r = requests.get(url, timeout=10, stream=True,
                             headers={"User-Agent": _LIVENESS_UA})
        return r.status_code < 400 or r.status_code in (403, 429)
    except Exception:  # noqa: BLE001
        return False


def resolve_one(q: dict) -> dict:
    """Returns the updated query record. Candidates = synthesis-stage candidates + Google/Bing organic merged."""
    serp_items = serp_both(q["query"], k=6)
    synth_cands = q.get("meta", {}).get("candidate_gold_urls", [])
    candidates = rule_filter(synth_cands + [it.url for it in serp_items])

    if not candidates:
        q.setdefault("meta", {})["anchor_candidate"] = True
        q["meta"]["gold_resolve_note"] = "No qualified SERP candidates (all rule-filtered or no results); pending human review"
        return q

    picked = llm_adjudicate(q["query"], candidates)
    alive = [u for u in picked if http_alive(u)]

    if not alive:
        q.setdefault("meta", {})["anchor_candidate"] = True
        q["meta"]["gold_resolve_note"] = f"LLM adjudicated no qualified gold or all failed liveness ({len(candidates)} candidates)"
        return q

    # MERGE into existing gold: a wholesale replacement used to destroy pre-existing
    # gold.answer (QA anchors) and reset curated url_match modes
    gold = q.setdefault("gold", {})
    gold["gold_urls"] = alive
    gold.setdefault("url_match", "prefix")
    meta = q.setdefault("meta", {})
    meta["gold_source"] = "google+bing_serp+llm_adjudicate+http"
    meta["gold_resolved_at"] = today_utc()
    meta["needs_review"] = q.get("difficulty") == "tail"
    meta.pop("anchor_candidate", None)
    return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="query JSONL, rewritten in place")
    ap.add_argument("--out", default=None, help="output path; defaults to overwriting the input")
    ap.add_argument("--force", action="store_true", help="re-resolve entries that already have gold")
    ap.add_argument("--concurrency", type=int, default=4, help="parallel resolutions (serp.py owns the backoff)")
    args = ap.parse_args()

    path = Path(args.queries)
    queries = load_jsonl(path, tolerate_torn_tail=False)

    targets = [
        q for q in queries
        if q.get("intent") == "navigational" and (args.force or not q.get("gold", {}).get("gold_urls"))
    ]
    print(f"Navigational anchors to resolve: {len(targets)} / {len(queries)} total")
    print(f"Estimated consumption: {len(targets) * 2} SERP calls (google+bing) + ≤{len(targets)} LLM calls")

    n_ok, n_pending, n_done = 0, 0, 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(resolve_one, q): q for q in targets}
        for fut in as_completed(futures):
            q = futures[fut]
            n_done += 1
            try:
                fut.result()  # resolve_one mutates q in place
                if q.get("gold", {}).get("gold_urls"):
                    n_ok += 1
                    flag = " [REVIEW]" if q["meta"].get("needs_review") else ""
                    print(f"  [{n_done}/{len(targets)}] {q['qid']}{flag}: {q['gold']['gold_urls'][0]}", flush=True)
                else:
                    n_pending += 1
                    print(f"  [{n_done}/{len(targets)}] {q['qid']} PENDING: {q['meta']['gold_resolve_note']}", flush=True)
            except Exception as e:  # noqa: BLE001
                n_pending += 1
                print(f"  [{n_done}/{len(targets)}] {q['qid']} FAILED: {e}", flush=True)

    out = Path(args.out) if args.out else path
    write_jsonl_atomic(out, queries)
    print(f"\nResolved {n_ok} / pending human {n_pending} → {out}")
    n_review = sum(1 for q in queries if q.get("meta", {}).get("needs_review"))
    print(f"{n_review} of them flagged needs_review (tail queries); remove the flag after human spot-checks")


if __name__ == "__main__":
    main()
