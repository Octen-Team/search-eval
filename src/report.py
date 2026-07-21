"""Slice reports and run-to-run diff.

Usage:
  python -m src.report --run results/run_20260707 --queries data/seed_queries.jsonl
  python -m src.report --run results/run_20260714 --baseline results/run_20260707 --queries data/seed_queries.jsonl

Output:
  1. Overall win-rate table (ours vs each competitor: win/tie/loss, plus a second look excluding low-confidence)
  2. Slice win rates (by intent/difficulty/freshness/vertical/language/form) — the weakness map
  3. Average per-dimension score gaps (which dimension is dragging us down)
  4. Anchor set hit@1 / hit@k
  5. With a baseline: per-query win/loss flip list (regression = won last time, lost this time)
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

from .common import load_jsonl, require_pairwise_meta, write_jsonl_atomic

SLICE_KEYS = ["intent", "difficulty", "freshness", "vertical", "language", "form"]
DIMENSIONS = ["relevance", "authority", "freshness", "diversity", "snippet_quality"]

# Health red line: when a slice's position_conflict share exceeds this, the judge is close to
# coin-flipping on that slice; its win-rate numbers are auto-flagged LOW-TRUST and must not
# drive iteration decisions (lesson from the 2026-07 synth trial's 35% conflict rate)
CONFLICT_TRUST_THRESHOLD = 0.25


def conflict_rate(recs: list[dict]) -> float:
    return sum(1 for r in recs if r.get("position_conflict")) / max(len(recs), 1)


def slice_cell(val: str, recs: list[dict], ours: str) -> str:
    """Win-rate cell for one slice bucket; appends a low-trust marker when the conflict rate exceeds the threshold."""
    w = sum(1 for r in recs if r["winner"] == ours)
    l = sum(1 for r in recs if r["winner"] not in (ours, "tie"))
    cell = f"{val}={w / max(w + l, 1):.0%}(n={len(recs)})"
    rate = conflict_rate(recs)
    if rate > CONFLICT_TRUST_THRESHOLD:
        cell += f" [LOW-TRUST:conflict {rate:.0%}]"
    return cell


def wtl(records: list[dict], ours: str) -> tuple[int, int, int]:
    """Win/tie/loss counts — the single home for this convention (report, gen_report,
    collect_history all consume it, so a definition change can't drift between them)."""
    w = sum(1 for r in records if r.get("winner") == ours)
    t = sum(1 for r in records if r.get("winner") == "tie")
    return w, t, len(records) - w - t


def anchor_hit_rates(recs: list[dict]) -> tuple[float, float, int]:
    """(hit@1 rate, hit@k rate, n) over anchor records."""
    n = len(recs)
    if n == 0:
        return 0.0, 0.0, 0
    return (sum(1 for a in recs if a["hit_at_1"]) / n,
            sum(1 for a in recs if a["hit_at_k"]) / n, n)


def percentile(sorted_vals: list, q: float):
    """Nearest-rank percentile: the ceil(q*n)-th order statistic. The previous
    int(n*q)-index form was off by one and reported the MAX as P95 at n<=20."""
    idx = max(0, math.ceil(q * len(sorted_vals)) - 1)
    return sorted_vals[idx]


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion — sane at the small n this eval runs at."""
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def wr(records: list[dict], ours: str) -> str:
    w, t, l = wtl(records, ours)
    n = len(records) or 1
    decisive = w + l
    lo, hi = wilson_ci(w, decisive)
    small = "  [n too small for conclusions]" if decisive < 20 else ""
    return (f"W {w} / T {t} / L {l}  (win-rate excl. tie: {w / max(decisive, 1):.0%}, "
            f"95% CI {lo:.0%}-{hi:.0%}, n={n}){small}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--queries", required=True, nargs="+")
    ap.add_argument("--baseline", default=None)
    args = ap.parse_args()

    run = Path(args.run)
    meta = require_pairwise_meta(run)
    ours = meta["ours"]
    qmeta = {q["qid"]: q for f in args.queries for q in load_jsonl(Path(f))}
    pairs = [p for p in load_jsonl(run / "pairwise.jsonl") if "winner" in p]
    anchors = load_jsonl(run / "anchors.jsonl")
    responses = load_jsonl(run / "responses.jsonl")

    print(f"\n{'='*70}\nRUN: {run}  |  ours={ours}  |  {meta['timestamp']}\n{'='*70}")

    # 1. overall win rate
    by_comp = defaultdict(list)
    for p in pairs:
        by_comp[p["system_y"]].append(p)
    print("\n## Overall win rate (ours vs competitors)")
    for comp, recs in by_comp.items():
        hi = [r for r in recs if r.get("confidence") != "low"]
        print(f"  vs {comp:8s}: {wr(recs, ours)}")
        print(f"     (high/med confidence only): {wr(hi, ours)}")
        conflicts = sum(1 for r in recs if r.get("position_conflict"))
        rate = conflict_rate(recs)
        flag = f"  ⚠ LOW-TRUST: win rates for this comparison are unreliable (conflict > {CONFLICT_TRUST_THRESHOLD:.0%}; fix the rubric before trusting conclusions)" \
            if rate > CONFLICT_TRUST_THRESHOLD else ""
        print(f"     position conflicts: {conflicts}/{len(recs)}{flag}")

    # 2. slice win rates — the weakness map
    print("\n## Slice win rates (look here for weaknesses; win-rate excl. tie)")
    for key in SLICE_KEYS:
        buckets = defaultdict(list)
        for p in pairs:
            q = qmeta.get(p["qid"], {})
            buckets[q.get(key, "?")].append(p)
        cells = [slice_cell(val, recs, ours) for val, recs in sorted(buckets.items())]
        print(f"  {key:11s}: " + "  ".join(cells))

    # 3. dimension gaps
    print("\n## Average dimension score gap (ours - competitor; negative = that dimension is a weakness)")
    for comp, recs in by_comp.items():
        diffs = {d: 0.0 for d in DIMENSIONS}
        for r in recs:
            for d in DIMENSIONS:
                diffs[d] += r["dim_scores_x"][d] - r["dim_scores_y"][d]
        n = len(recs) or 1
        line = "  ".join(f"{d}:{diffs[d] / n:+.2f}" for d in DIMENSIONS)
        print(f"  vs {comp:8s}: {line}")

    # 4. anchor set
    if anchors:
        print("\n## Anchor set (objective queries)")
        by_be = defaultdict(list)
        for a in anchors:
            by_be[a["backend"]].append(a)
        for be, recs in sorted(by_be.items()):
            h1, hk, n = anchor_hit_rates(recs)
            print(f"  {be:8s}: hit@1={h1:.0%}  hit@k={hk:.0%}  (n={n})")

    # 5. latency — API-returned SERVER latency only (octen meta.latency / exa searchTime /
    #    tavily response_time). Backends that report no server time (parallel/brave/perplexity)
    #    are left blank; we deliberately do NOT substitute the e2e round-trip.
    if responses:
        print("\n## Latency P50/P90 (ms) — API-returned server latency (blank if the backend reports none)")
        srv = defaultdict(list)
        backends_seen = set()
        for r in responses:
            if r.get("error"):
                continue
            backends_seen.add(r["backend"])
            if r.get("reported_latency_ms") is not None:
                srv[r["backend"]].append(r["reported_latency_ms"])
        for be in sorted(backends_seen):
            s = sorted(srv.get(be, []))
            if s:
                print(f"  {be:16s}: latency P50={percentile(s, 0.5):.0f} P90={percentile(s, 0.9):.0f}  (n={len(s)})")
            else:
                print(f"  {be:16s}: latency —  (API returns no server-side time)")

    # 6. baseline diff
    if args.baseline:
        base_pairs = {(p["qid"], p["system_y"]): p for p in load_jsonl(Path(args.baseline) / "pairwise.jsonl") if "winner" in p}
        print(f"\n## Win/loss flips vs baseline ({args.baseline})")
        regressions, improvements = [], []
        for p in pairs:
            b = base_pairs.get((p["qid"], p["system_y"]))
            if not b:
                continue
            was_win, now_win = b["winner"] == ours, p["winner"] == ours
            if was_win and not now_win:
                regressions.append(p)
            elif not was_win and now_win:
                improvements.append(p)
        print(f"  regressions {len(regressions)} | improvements {len(improvements)}")
        for p in regressions[:20]:
            q = qmeta.get(p["qid"], {})
            print(f"  [REGRESS] {p['qid']} vs {p['system_y']}: {q.get('query','')[:50]} | {p.get('evidence','')[:80]}")

    # 7. failure case list → for triage
    losses = [p for p in pairs if p["winner"] not in (ours, "tie")]
    loss_file = run / "losses.jsonl"
    write_jsonl_atomic(loss_file, losses)
    print(f"\n{len(losses)} failure cases → {loss_file} (feed to src/triage.py next)")


if __name__ == "__main__":
    main()
