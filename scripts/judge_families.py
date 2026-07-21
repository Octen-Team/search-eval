"""Cross-family judge agreement.

Re-judges a sample of pairs from a run with judge models from OTHER model families (default:
openai/gpt-5.2 + google/gemini-2.5-pro), using the run's cached responses — no backend calls.
The run's own verdicts (claude family) are the baseline. If three independent families agree,
the verdicts are properties of the evidence, not idiosyncrasies of one judge model.

Sampling reuses scripts.calibration.sample_pairs with the SAME default seed, so with matching
--n this covers the same pairs humans blind-label — disagreement cases can be cross-examined
against both the other families and the human labels.

Reported: pairwise agreement matrix (exact 3-class match), hard-flip rate (both decisive,
opposite winners — the damning kind), and how often the baseline deviates from the
cross-family majority.

Concurrent + resumable: verdicts append to <run>/judge_families.jsonl; re-running skips
completed (qid, system_y, judge_model) keys.

Usage:
  python -m scripts.judge_families --run results/run_20260708_v4 \\
      --queries data/main_queries.jsonl data/realtime_20260708.jsonl \\
      --n 40 --concurrency 8
"""
from __future__ import annotations

import argparse
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

from scripts.calibration import sample_pairs
from src.backends import response_from_record
from src.common import load_jsonl, require_pairwise_meta
from src.judge import judge_pair
from src.rubric_gen import load_rubrics

DEFAULT_FAMILIES = ["openai/gpt-5.2", "google/gemini-2.5-pro"]


def agreement_stats(verdicts: dict[str, dict[tuple, str]]) -> dict:
    """verdicts: judge_name -> {(qid, system_y): winner}. Pairwise over shared keys."""
    out = {}
    for a, b in combinations(sorted(verdicts), 2):
        shared = sorted(set(verdicts[a]) & set(verdicts[b]))
        if not shared:
            continue
        exact = sum(1 for k in shared if verdicts[a][k] == verdicts[b][k])
        flips = sum(1 for k in shared
                    if "tie" not in (verdicts[a][k], verdicts[b][k]) and verdicts[a][k] != verdicts[b][k])
        out[(a, b)] = {"n": len(shared), "exact": exact, "exact_rate": exact / len(shared),
                       "hard_flips": flips, "hard_flip_rate": flips / len(shared)}
    return out


def majority_deviation(baseline: str, verdicts: dict[str, dict[tuple, str]]) -> dict:
    """How often the baseline judge stands against a unanimous verdict of all other families."""
    others = [j for j in verdicts if j != baseline]
    keys = set(verdicts[baseline])
    for j in others:
        keys &= set(verdicts[j])
    unanimous = {k for k in keys if len({verdicts[j][k] for j in others}) == 1}
    outvoted = [k for k in unanimous if verdicts[baseline][k] != verdicts[others[0]][k]]
    return {"n": len(keys), "others_unanimous": len(unanimous), "baseline_outvoted": len(outvoted),
            "outvoted_keys": sorted(outvoted)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--queries", required=True, nargs="+")
    ap.add_argument("--rubrics", default="data/rubrics.jsonl")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42, help="keep = calibration export for the same pairs")
    args = ap.parse_args()

    run = Path(args.run)
    meta = require_pairwise_meta(run)
    ours = meta["ours"]
    pairs = [p for p in load_jsonl(run / "pairwise.jsonl") if "winner" in p]
    responses = {(r["qid"], r["backend"]): r for r in load_jsonl(run / "responses.jsonl")}
    qmeta = {q["qid"]: q for f in args.queries for q in load_jsonl(f)}
    rubrics = load_rubrics(args.rubrics)

    usable = [p for p in pairs
              if (p["qid"], p["system_x"]) in responses and (p["qid"], p["system_y"]) in responses
              and p["qid"] in qmeta]
    picked = sample_pairs(usable, args.n, ours, seed=args.seed)
    tasks = [(p, fam) for p in picked for fam in args.families]

    out_path = run / "judge_families.jsonl"
    done = {(r["qid"], r["system_y"], r["judge_model"]) for r in load_jsonl(out_path)}
    todo = [(p, fam) for p, fam in tasks if (p["qid"], p["system_y"], fam) not in done]
    print(f"{len(picked)} pairs × {len(args.families)} families = {len(tasks)} verdicts "
          f"({len(done)} done, {len(todo)} to judge ≈ {len(todo) * 2} LLM calls)")

    lock = threading.Lock()

    def one(task: tuple[dict, str]) -> dict:
        p, fam = task
        rx = response_from_record(responses[(p["qid"], p["system_x"])])
        ry = response_from_record(responses[(p["qid"], p["system_y"])])
        v = judge_pair(p["qid"], qmeta[p["qid"]], rx, ry, k=args.k,
                       rubric_rec=rubrics.get(p["qid"]), judge_model=fam)
        return {"qid": p["qid"], "system_x": p["system_x"], "system_y": p["system_y"],
                "judge_model": fam, "winner": v.winner, "confidence": v.confidence,
                "position_conflict": v.position_conflict,
                "weighted_x": v.weighted_x, "weighted_y": v.weighted_y}

    with out_path.open("a", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(one, t): t for t in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001 — keep the batch going; the key stays incomplete for resume
                p, fam = futures[fut]
                print(f"  ! ({p['qid']}, {fam}): {e}")
                continue
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} judged")

    # baseline = the run's own verdicts, restricted to the sampled pairs
    baseline_name = f"baseline({meta.get('judge_model', 'claude')})"
    verdicts: dict[str, dict[tuple, str]] = defaultdict(dict)
    picked_keys = {(p["qid"], p["system_y"]) for p in picked}
    for p in picked:
        verdicts[baseline_name][(p["qid"], p["system_y"])] = p["winner"]
    for r in load_jsonl(out_path):
        key = (r["qid"], r["system_y"])
        if key in picked_keys and r["judge_model"] in args.families:
            verdicts[r["judge_model"]][key] = r["winner"]

    print(f"\n=== Cross-family agreement ({len(picked)} pairs) ===")
    for (a, b), s in agreement_stats(verdicts).items():
        print(f"{a} ↔ {b}: exact {s['exact']}/{s['n']} = {s['exact_rate']:.0%}, "
              f"hard flips {s['hard_flips']} ({s['hard_flip_rate']:.0%})")
    dev = majority_deviation(baseline_name, verdicts)
    print(f"\nother families unanimous on {dev['others_unanimous']}/{dev['n']} pairs; "
          f"baseline outvoted on {dev['baseline_outvoted']}")
    for k in dev["outvoted_keys"]:
        fams = {j: verdicts[j].get(k) for j in verdicts}
        print(f"  OUTVOTED {k}: {fams}")


if __name__ == "__main__":
    main()
