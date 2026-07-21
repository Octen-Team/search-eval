"""Main evaluation pipeline (SERP-level).

Usage:
  python -m src.run_eval --queries data/main_queries.jsonl data/realtime_YYYYMMDD.jsonl \\
      --ours octen --competitors exa brave --k 10 --concurrency 4 \\
      --out results/run_$(date +%Y%m%d)

Artifacts (one directory per run, all JSONL for easy diffing and versioning):
  responses.jsonl   one record per (qid, backend): raw results + latency
  anchors.jsonl     anchor-query hit status (hit@1 / hit@k / rank) — skipped on fetch errors
  pairwise.jsonl    verdicts of ours vs each competitor
  run_meta.json     run parameters, timestamp, query-set hash (written at startup AND completion,
                    so report/gate tools work on interrupted runs)

Concurrency & resume:
- The unit of parallelism AND of resume is one query episode (fetch all backends serially
  within the episode — preserving the same-time-window invariant for realtime queries —
  then judge). Episodes run in parallel across queries.
- Records are appended and flushed per episode. On restart, only fully-complete qids are kept
  (responses for every backend, none errored, every expected pairwise verdict present);
  everything else — judge errors, fetch errors, partial episodes — is pruned and re-run.
  The pruning happens unconditionally, so stale records can never survive as duplicates.
- A torn final JSONL line (crash mid-write) is tolerated on load.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .backends import get_backend, SearchResponse
from .common import (load_jsonl, load_queries, stale_realtime, today_utc,
                     url_match, write_jsonl_atomic)
from .judge import judge_pair
from .rubric_gen import load_rubrics, qsha

FILES = ("responses.jsonl", "anchors.jsonl", "pairwise.jsonl")


def score_anchor(q: dict, resp: SearchResponse, k: int) -> dict | None:
    gold = q.get("gold", {})
    urls, mode = gold.get("gold_urls"), gold.get("url_match", "prefix")
    if not urls:
        return None
    best_rank = None
    for r in resp.results[:k]:
        if any(url_match(r.url, g, mode) for g in urls):
            best_rank = r.rank
            break
    return {
        "qid": q["qid"],
        "backend": resp.backend,
        "hit_at_1": best_rank == 1,
        "hit_at_k": best_rank is not None,
        "rank": best_rank,
        "k": k,
    }


def expected_pairs(resp_recs: dict[str, dict], ours: str, competitors: list[str]) -> set[str]:
    """Competitors for which a pairwise verdict is expected given fetch outcomes."""
    ours_rec = resp_recs.get(ours)
    if not ours_rec or ours_rec.get("error"):
        return set()
    return {c for c in competitors if resp_recs.get(c) and not resp_recs[c].get("error")}


def load_resume(out: Path, all_backends: list[str], ours: str, competitors: list[str],
                skip_judge: bool) -> tuple[set[str], dict[str, list[dict]], int]:
    """Returns (complete qids, kept records per file, total raw records loaded).

    A qid is complete only when every backend has a NON-ERRORED response and (unless
    skip_judge) every expected pairwise verdict exists. Fetch errors therefore re-run on
    resume instead of silently shrinking n. Records of incomplete qids are dropped.
    """
    raw: dict[str, list[dict]] = {name: load_jsonl(out / name) for name in FILES}

    resp_by_qid: dict[str, dict[str, dict]] = {}
    for r in raw["responses.jsonl"]:
        resp_by_qid.setdefault(r["qid"], {})[r["backend"]] = r
    judged_by_qid: dict[str, set[str]] = {}
    for p in raw["pairwise.jsonl"]:
        if "winner" in p:  # error records don't count as done → retried
            judged_by_qid.setdefault(p["qid"], set()).add(p["system_y"])

    complete = set()
    for qid, recs in resp_by_qid.items():
        if not set(all_backends) <= set(recs):
            continue
        if any(recs[b].get("error") for b in all_backends):
            continue  # a fetch error is transient until proven otherwise — re-run the episode
        if not skip_judge and not expected_pairs(recs, ours, competitors) <= judged_by_qid.get(qid, set()):
            continue
        complete.add(qid)

    kept = {name: [r for r in raw[name] if r.get("qid") in complete] for name in FILES}
    total_raw = sum(len(v) for v in raw.values())
    return complete, kept, total_raw


def run_episode(q: dict, backends: dict, ours: str, competitors: list[str], k: int,
                skip_judge: bool, rubrics: dict) -> dict[str, list[str]]:
    """One full query episode: fetch every backend (serially — same time window), score anchors, judge."""
    lines: dict[str, list[str]] = {name: [] for name in FILES}
    responses: dict[str, SearchResponse] = {}
    for name, be in backends.items():
        resp = be.search(q["query"], k=k)
        responses[name] = resp
        rec = {
            "qid": q["qid"], "backend": name, "latency_ms": round(resp.latency_ms, 1),
            "reported_latency_ms": resp.reported_latency_ms,  # server-side (octen/exa/tavily); None otherwise
            "error": resp.error, "n_results": len(resp.results),
            "results": [asdict(r) | {"raw": None} for r in resp.results[:k]],
        }
        lines["responses.jsonl"].append(json.dumps(rec, ensure_ascii=False))
        if not resp.error:  # an API failure is not a ranking miss — never score it as one
            anchor = score_anchor(q, resp, k)
            if anchor:
                lines["anchors.jsonl"].append(json.dumps(anchor, ensure_ascii=False))

    if not skip_judge and not responses[ours].error:
        for comp in competitors:
            if responses[comp].error:
                continue
            try:
                rub = rubrics.get(q["qid"])
                if rub and rub.get("query_sha") != qsha(q["query"]):
                    print(f"  WARN {q['qid']}: query text changed but the rubric was not regenerated; "
                          f"falling back to the generic rubric for this query", flush=True)
                    rub = None
                verdict = judge_pair(q["qid"], q, responses[ours], responses[comp], k=k, rubric_rec=rub)
                lines["pairwise.jsonl"].append(json.dumps(asdict(verdict), ensure_ascii=False))
            except Exception as e:  # noqa: BLE001
                lines["pairwise.jsonl"].append(json.dumps(
                    {"qid": q["qid"], "system_x": ours, "system_y": comp, "error": str(e)},
                    ensure_ascii=False))
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, nargs="+",
                    help="one or more query JSONLs (e.g. the main set + a same-day realtime batch)")
    ap.add_argument("--ours", default="octen")
    ap.add_argument("--competitors", nargs="+", default=["exa", "brave"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=4, help="parallel query episodes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-judge", action="store_true", help="fetch results and anchors only, skip the judge (saves tokens while debugging)")
    ap.add_argument("--rubrics", default="data/rubrics.jsonl", help="per-query rubric file; falls back to the generic rubric if missing")
    args = ap.parse_args()

    rubrics = load_rubrics(args.rubrics)
    if rubrics:
        print(f"Loaded {len(rubrics)} per-query rubrics ({args.rubrics})", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    queries = load_queries(args.queries)
    stale = stale_realtime(queries, today_utc())
    if stale:
        print(f"WARN: {len(stale)} realtime queries were synthesized on an earlier (UTC) day and are "
              f"likely stale ({stale[:5]}...). Regenerate with: python -m src.realtime_synth", flush=True)
    all_backends = [args.ours] + args.competitors
    backends = {name: get_backend(name) for name in all_backends}

    complete, kept, total_raw = load_resume(out, all_backends, args.ours, args.competitors, args.skip_judge)
    todo = [q for q in queries if q["qid"] not in complete]
    if total_raw:
        # prune UNCONDITIONALLY: incomplete-qid records must never survive to become duplicates
        for name in FILES:
            write_jsonl_atomic(out / name, kept[name])
        print(f"Resume: {len(complete)} qids already complete, {len(todo)} to run "
              f"(pruned {total_raw - sum(len(v) for v in kept.values())} stale records)", flush=True)

    h = hashlib.sha256()
    for f in args.queries:
        h.update(Path(f).read_bytes())
    qhash = h.hexdigest()[:12]

    def write_meta(status: str) -> None:
        (out / "run_meta.json").write_text(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "queries_file": args.queries, "queries_sha": qhash, "n_queries": len(queries),
            "ours": args.ours, "competitors": args.competitors, "k": args.k,
            "concurrency": args.concurrency, "status": status,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    write_meta("running")  # downstream tools must work on interrupted runs too

    handles = {name: (out / name).open("a", encoding="utf-8") for name in FILES}
    write_lock = threading.Lock()
    n_done = 0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(run_episode, q, backends, args.ours, args.competitors,
                               args.k, args.skip_judge, rubrics): q for q in todo}
        for fut in as_completed(futures):
            q = futures[fut]
            try:
                lines = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  EPISODE FAILED {q['qid']}: {e} (will re-run on resume)", flush=True)
                continue
            with write_lock:
                for name, ls in lines.items():
                    for l in ls:
                        handles[name].write(l + "\n")
                    handles[name].flush()
                n_done += 1
                print(f"[{n_done}/{len(todo)}] {q['qid']}: {q['query'][:60]}", flush=True)
    for hdl in handles.values():
        hdl.close()

    write_meta("complete")
    print(f"Done in {time.perf_counter() - t0:.0f}s → {out}", flush=True)


if __name__ == "__main__":
    main()
