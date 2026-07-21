"""Collect per-run summaries into data/run_history.jsonl and print trends.

The run directories themselves are gitignored artifacts; this file is the compact,
local time series used for trend tracking across runs.

Usage:
  python -m scripts.collect_history --run results/run_20260707        # append one run's summary
  python -m scripts.collect_history --trend                           # print the time series
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.common import load_jsonl, require_pairwise_meta
from src.report import anchor_hit_rates, conflict_rate, percentile, wtl

HISTORY = Path("data/run_history.jsonl")


def summarize_run(run: Path) -> dict:
    meta = require_pairwise_meta(run)  # clear error for agent-eval dirs / interrupted runs
    ours = meta["ours"]
    pairs = [p for p in load_jsonl(run / "pairwise.jsonl") if "winner" in p]
    anchors = load_jsonl(run / "anchors.jsonl")
    responses = load_jsonl(run / "responses.jsonl")

    by_comp = defaultdict(list)
    for p in pairs:
        by_comp[p["system_y"]].append(p)
    pairwise = {}
    for comp, recs in by_comp.items():
        w, t, l = wtl(recs, ours)  # shared counting — the git-versioned trend must not drift from report.py
        pairwise[comp] = {"w": w, "t": t, "l": l,
                          "conflict_rate": round(conflict_rate(recs), 3)}

    anchor_rates = {}
    by_be = defaultdict(list)
    for a in anchors:
        by_be[a["backend"]].append(a)
    for be, recs in by_be.items():
        h1, hk, n = anchor_hit_rates(recs)
        anchor_rates[be] = {"hit_at_1": round(h1, 3), "hit_at_k": round(hk, 3), "n": n}

    lat = {}
    by_be_lat = defaultdict(list)
    for r in responses:
        if not r.get("error"):
            by_be_lat[r["backend"]].append(r["latency_ms"])
    for be, ls in by_be_lat.items():
        ls.sort()
        lat[be] = round(percentile(ls, 0.5), 0)

    return {"run": str(run), "timestamp": meta["timestamp"], "queries_sha": meta.get("queries_sha"),
            "n_queries": meta.get("n_queries"), "ours": ours,
            "pairwise": pairwise, "anchors": anchor_rates, "latency_p50_ms": lat}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--trend", action="store_true")
    args = ap.parse_args()

    history = load_jsonl(HISTORY)

    if args.run:
        summary = summarize_run(Path(args.run))
        # compare resolved paths: 'results/run_X' and '/abs/.../results/run_X' are the same run
        run_resolved = Path(summary["run"]).resolve()
        if any(Path(h["run"]).resolve() == run_resolved for h in history):
            print(f"{summary['run']} already in history — skipped (delete the line to re-collect)")
        else:
            HISTORY.parent.mkdir(parents=True, exist_ok=True)
            with HISTORY.open("a", encoding="utf-8") as f:
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")
            history.append(summary)
            print(f"collected → {HISTORY} ({len(history)} runs)")

    if args.trend or not args.run:
        if not history:
            print("history is empty — collect runs first")
            return
        history.sort(key=lambda h: h["timestamp"])
        print(f"\n{'timestamp':20s} {'run':28s} {'n':>4s}  win-rate excl. tie (conflict) per competitor | ours hit@k")
        for h in history:
            cells = []
            for comp, s in sorted(h["pairwise"].items()):
                decisive = s["w"] + s["l"]
                wr = s["w"] / max(decisive, 1)
                cells.append(f"vs {comp}: {wr:.0%} ({s['conflict_rate']:.0%})")
            hk = h["anchors"].get(h["ours"], {}).get("hit_at_k")
            hk_s = f"{hk:.0%}" if hk is not None else "-"
            print(f"{h['timestamp'][:19]:20s} {Path(h['run']).name:28s} {h['n_queries'] or 0:>4d}  "
                  + "  ".join(cells) + f" | {hk_s}")


if __name__ == "__main__":
    main()
