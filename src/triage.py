"""Semi-automated failure attribution (triage).

Input:  losses.jsonl + responses.jsonl produced by report.py
Output: triage.jsonl — one record per failure case, with an automatically determined mode,
        or a ticket pending human work.

Automation levels:
  L0 (service failure)   fully automated: read error / n_results straight from responses
  L1 (index coverage)    automated via --probe: search our own backend with a URL-targeted
                         query and look for the URL in top-k. NOTE: octen does NOT honor the
                         site: operator (verified 2026-07), so the probe uses host + path
                         tokens instead. A probed miss cannot separate "not indexed" from
                         "catastrophically unranked" → mode INDEX_MISS_PROBED, confidence=low.
  L2-L4                  create tickets for human / engine-internal follow-up.

Usage:
  python -m src.triage --run results/run_20260707 [--probe]
"""
from __future__ import annotations

import argparse
import json
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

from .backends import _load_dotenv
from .common import load_jsonl, require_pairwise_meta, write_jsonl_atomic

_load_dotenv()  # load .env so the --probe backend picks up its API key


_PROBE_STOP = {"index", "html", "htm", "en", "www", "php", "aspx"}


def probe_terms(url: str) -> str:
    """Build a query aimed at retrieving one specific URL: host (no www) + up to 4 distinctive path tokens.
    (site: operator would be the right tool, but octen ignores it — verified 2026-07.)"""
    p = urlparse(url)
    host = p.netloc.removeprefix("www.")
    tokens = [t for t in re.split(r"[/\-_.]+", p.path)
              if t and not t.isdigit() and t.lower() not in _PROBE_STOP][:4]
    return " ".join([host] + tokens)


def probe_in_index(url: str, backend) -> bool | None:
    """Degraded L1 check: can our backend retrieve this URL for a query aimed straight at it?
    True = the exact URL is retrievable. False = likely index/recall miss (low confidence).
    None = probe itself failed. Child pages do NOT count — retrieving /docs/v2 says nothing
    about /docs itself (used to be a false 'indexed')."""
    resp = backend.search(probe_terms(url), k=10)
    if resp.error:
        return None
    target = url.rstrip("/")
    return any(r.url.rstrip("/") == target for r in resp.results)


def candidate_gold_urls(loss: dict, responses_by_key: dict) -> list[str]:
    """Extract "suspected correct URLs" from the winner's (competitor's) top-3 results, as L1 index-coverage check targets.

    Note: this is a heuristic — what a competitor ranks high is not necessarily gold, but it carries
    enough signal for coarse diagnoses like INDEX_MISS_PROBED. Precise gold depends on human labeling
    or the anchor set.
    """
    winner = loss["winner"]
    rec = responses_by_key.get((loss["qid"], winner))
    if not rec:
        return []
    return [r["url"] for r in rec.get("results", [])[:3]]


def triage_one(loss: dict, responses_by_key: dict, ours: str, probe_backend=None) -> dict:
    qid = loss["qid"]
    competitor = loss.get("system_y")  # kept on every record so reports can pair evidence correctly

    # data consistency first: a MISSING response record is not "the backend returned nothing"
    if (qid, ours) not in responses_by_key:
        return {"qid": qid, "competitor": competitor, "mode": "DATA_MISSING", "auto": False,
                "note": "no response record for ours — losses.jsonl and responses.jsonl are out of sync "
                        "(stale losses from an earlier report, or a pruned resume?)"}
    ours_rec = responses_by_key[(qid, ours)]

    # L0: service layer
    if ours_rec.get("error"):
        return {"qid": qid, "competitor": competitor, "mode": "SERVICE_ERROR",
                "detail": ours_rec["error"], "auto": True}
    if ours_rec.get("n_results", 0) == 0:
        return {"qid": qid, "competitor": competitor, "mode": "SERVICE_EMPTY",
                "detail": "no results returned", "auto": True}

    # L1: index coverage (reverse-check the competitor's top-3) via --probe against our own backend
    suspects = candidate_gold_urls(loss, responses_by_key)
    missing, unknown, present = [], [], []
    for u in suspects:
        in_idx = probe_in_index(u, probe_backend) if probe_backend is not None else None
        if in_idx is False:
            missing.append(u)
        elif in_idx is None:
            unknown.append(u)
        else:
            present.append(u)

    # ANY confirmed-missing suspected gold is an index-coverage signal — the most upstream
    # cause wins even when other suspects are present (partial misses used to be silently
    # misfiled into the ranking bucket)
    if missing:
        note = (f"{len(missing)}/{len(suspects)} suspected gold URLs not retrievable"
                + (f"; {len(present)} present" if present else "")
                + (f"; {len(unknown)} undeterminable" if unknown else "")
                + "; probe-based, cannot separate index-miss from catastrophic ranking miss")
        return {"qid": qid, "competitor": competitor, "mode": "INDEX_MISS_PROBED",
                "detail": {"missing_urls": missing, "present_urls": present, "unknown_urls": unknown},
                "auto": True, "confidence": "low", "note": note}
    if unknown:
        reason = "probe errored" if probe_backend is not None else "--probe not enabled"
        return {"qid": qid, "competitor": competitor, "mode": "PENDING_INDEX_CHECK",
                "detail": {"urls": unknown}, "auto": False,
                "note": f"index membership undeterminable for {len(unknown)} URLs ({reason})"}

    # L2-L4: needs engine-internal signals; create a ticket
    dim_gap = {d: round(loss["dim_scores_x"][d] - loss["dim_scores_y"][d], 2)
               for d in loss.get("dim_scores_x", {})}
    worst = min(dim_gap, key=dim_gap.get) if dim_gap else None
    return {
        "qid": qid,
        "competitor": competitor,
        "mode": "PENDING_L2_L4",
        "auto": False,
        "detail": {
            "evidence": loss.get("evidence", ""),
            "dim_gap": dim_gap,
            "suspect_gold_urls": suspects,
        },
        "note": "Suspected gold appears to be in the index but we failed to rank it. Needs engine-internal inspection: is the doc in the recall candidates, and how is it ranked?"
        + (" (snippet_quality shows the largest gap → check L4 extraction first)" if worst == "snippet_quality" else ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--probe", action="store_true",
                    help="L1 index/recall check: probe our own backend with URL-targeted "
                         "queries (1 search per suspect URL; results are confidence=low)")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="parallel probe episodes (probe mode does independent backend searches per case)")
    args = ap.parse_args()
    run = Path(args.run)

    meta = require_pairwise_meta(run)
    ours = meta["ours"]
    losses = load_jsonl(run / "losses.jsonl")
    responses = load_jsonl(run / "responses.jsonl")
    responses_by_key = {(r["qid"], r["backend"]): r for r in responses}

    probe_backend = None
    if args.probe:
        from .backends import get_backend
        probe_backend = get_backend(ours)

    out = run / "triage.jsonl"
    # resume: probe mode issues live backend searches per case — completed cases must survive
    # an interruption (append+flush per record, keyed by (qid, competitor), last-wins)
    prior = {(r["qid"], r.get("competitor")): r for r in load_jsonl(out)}
    todo = [l for l in losses if (l["qid"], l.get("system_y")) not in prior]
    if prior:
        print(f"Resume: {len(prior)} cases already triaged, {len(todo)} to run", flush=True)

    records = dict(prior)
    write_lock = threading.Lock()
    with out.open("a", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=max(1, args.concurrency if args.probe else 1)) as pool:
        futures = {pool.submit(triage_one, l, responses_by_key, ours, probe_backend): l for l in todo}
        for fut in as_completed(futures):
            rec = fut.result()
            with write_lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                records[(rec["qid"], rec.get("competitor"))] = rec
    # compact atomically (dedupe resumed duplicates, stable order)
    records = list(records.values())
    write_jsonl_atomic(out, records)

    dist = Counter(r["mode"] for r in records)
    print(f"\nFailure mode distribution (n={len(records)}) — iteration priorities live here:")
    for mode, n in dist.most_common():
        print(f"  {mode:24s} {n:4d}  ({n/len(records):.0%})")
    print(f"\nDetails → {out}")
    pending = sum(1 for r in records if not r.get("auto"))
    if pending:
        print(f"{pending} still need a human. "
              f"Run with --probe to auto-classify L1 index/recall misses.")


if __name__ == "__main__":
    main()
