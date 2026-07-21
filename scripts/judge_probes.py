"""Known-answer degradation probes for the pairwise judge.

Human agreement measures "does the judge look like an annotator"; this measures the harder,
fully automatic property: "does the judge detect a KNOWN quality difference". Each probe takes
a real cached result list from a run and pits it against a deliberately degraded copy of
itself. The correct verdict is known by construction (the original must not lose), so no human
labels — and no assumptions about human carefulness — are involved.

Probes:
  drop_top1         remove the #1 result           → original should win (rank quality)
  drop_top3         remove the top 3 results       → original should win, strictly
  reverse_order     reverse the top-k ordering     → original should win (ranking sensitivity)
  truncate_snippets cut every snippet to 60 chars  → original should win (snippet quality)

A "hard fail" is the degraded side WINNING — that is a judge defect regardless of philosophy.
Ties on mild probes (drop_top1 / reverse / truncate) are "soft misses": reduced sensitivity,
not necessarily an error. Sensitivity = original-win rate per probe.

Concurrent + resumable: verdicts append to <run>/judge_probes.jsonl per record; interrupting
and re-running skips completed (qid, base_backend, probe) keys.

Usage:
  python -m scripts.judge_probes --run results/run_20260708_v4 \\
      --queries data/main_queries.jsonl data/realtime_20260708.jsonl \\
      --n 30 --concurrency 8
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.backends import SearchResponse, response_from_record
from src.common import load_jsonl
from src.judge import judge_pair
from src.rubric_gen import load_rubrics

PROBES = ["drop_top1", "drop_top3", "reverse_order", "truncate_snippets"]
STRICT_PROBES = {"drop_top3"}  # tie is a hard fail here — the gap is too large to miss


def degrade(resp: SearchResponse, probe: str, k: int) -> SearchResponse:
    """Return a degraded deep copy; ranks are renumbered so the judge sees a well-formed list."""
    d = copy.deepcopy(resp)
    d.backend = "degraded"
    if probe == "drop_top1":
        d.results = d.results[1:]
    elif probe == "drop_top3":
        d.results = d.results[3:]
    elif probe == "reverse_order":
        d.results = list(reversed(d.results[:k])) + d.results[k:]
    elif probe == "truncate_snippets":
        for r in d.results:
            if len(r.snippet) > 60:
                r.snippet = r.snippet[:60] + "…"
    else:
        raise ValueError(f"unknown probe: {probe}")
    for i, r in enumerate(d.results, start=1):
        r.rank = i
    return d


def outcome_of(winner: str, probe: str) -> str:
    """pass = original won; soft_miss = tie on a mild probe; hard_fail = degraded won (or tie on strict)."""
    if winner == "original":
        return "pass"
    if winner == "tie" and probe not in STRICT_PROBES:
        return "soft_miss"
    return "hard_fail"


def pick_bases(responses: dict, qmeta: dict, n: int, seed: int) -> list[tuple[str, str]]:
    """Choose (qid, backend) bases: round-robin verticals for coverage, cycle backends for
    diversity, require a full-enough list (≥8 results, no error) so degradation is meaningful."""
    by_vertical: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for (qid, backend), rec in sorted(responses.items()):
        if qid in qmeta and not rec.get("error") and len(rec.get("results", [])) >= 8:
            by_vertical[qmeta[qid].get("vertical", "general")].append((qid, backend))
    rng = random.Random(seed)
    for lst in by_vertical.values():
        rng.shuffle(lst)
    picked, used_qids = [], set()
    verticals = sorted(by_vertical)
    while len(picked) < n and any(by_vertical[v] for v in verticals):
        for v in verticals:
            while by_vertical[v]:
                cand = by_vertical[v].pop()
                if cand[0] not in used_qids:          # one base per qid
                    picked.append(cand)
                    used_qids.add(cand[0])
                    break
            if len(picked) >= n:
                break
    return picked[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--queries", required=True, nargs="+")
    ap.add_argument("--rubrics", default="data/rubrics.jsonl")
    ap.add_argument("--n", type=int, default=30, help="number of base result lists")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run = Path(args.run)
    responses = {(r["qid"], r["backend"]): r for r in load_jsonl(run / "responses.jsonl")}
    qmeta = {q["qid"]: q for f in args.queries for q in load_jsonl(f)}
    rubrics = load_rubrics(args.rubrics)

    bases = pick_bases(responses, qmeta, args.n, args.seed)
    tasks = [(qid, backend, probe) for qid, backend in bases for probe in PROBES]

    out_path = run / "judge_probes.jsonl"
    done = {(r["qid"], r["base_backend"], r["probe"]) for r in load_jsonl(out_path)}
    todo = [t for t in tasks if t not in done]
    print(f"{len(bases)} bases × {len(PROBES)} probes = {len(tasks)} pairs "
          f"({len(done)} done, {len(todo)} to judge ≈ {len(todo) * 2} LLM calls)")

    lock = threading.Lock()

    def one(task: tuple[str, str, str]) -> dict:
        qid, backend, probe = task
        original = response_from_record(responses[(qid, backend)], backend="original")
        degraded = degrade(original, probe, args.k)
        v = judge_pair(qid, qmeta[qid], original, degraded, k=args.k, rubric_rec=rubrics.get(qid))
        return {"qid": qid, "base_backend": backend, "probe": probe,
                "winner": v.winner, "outcome": outcome_of(v.winner, probe),
                "confidence": v.confidence, "position_conflict": v.position_conflict,
                "weighted_original": v.weighted_x, "weighted_degraded": v.weighted_y,
                "evidence": v.evidence}

    with out_path.open("a", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(one, t): t for t in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001 — keep the batch going; the key stays incomplete for resume
                print(f"  ! {futures[fut]}: {e}")
                continue
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} judged")

    records = load_jsonl(out_path)
    seen: dict[tuple, dict] = {(r["qid"], r["base_backend"], r["probe"]): r for r in records}
    print(f"\n=== Degradation probe summary ({len(seen)} pairs) ===")
    print(f"{'probe':<20} {'pass':>5} {'soft_miss':>10} {'hard_fail':>10} {'sensitivity':>12}")
    total_hard = 0
    for probe in PROBES:
        c = Counter(r["outcome"] for r in seen.values() if r["probe"] == probe)
        n = sum(c.values())
        total_hard += c["hard_fail"]
        print(f"{probe:<20} {c['pass']:>5} {c['soft_miss']:>10} {c['hard_fail']:>10} "
              f"{c['pass'] / max(n, 1):>11.0%}")
    print(f"\nhard fails total: {total_hard} — read each case: either a judge defect, or the "
          f"degradation accidentally IMPROVED a badly-ranked list (drop/reverse probes assume "
          f"the original top results were good; truncate_snippets is the only assumption-free probe).")
    for r in seen.values():
        if r["outcome"] == "hard_fail":
            gap = abs(r["weighted_original"] - r["weighted_degraded"])
            tag = "near-tie" if gap <= 0.25 else "decisive"
            print(f"  HARD FAIL [{tag}, gap={gap:.2f}] {r['qid']} {r['probe']} "
                  f"winner={r['winner']}: {r['evidence'][:160]}")


if __name__ == "__main__":
    main()
