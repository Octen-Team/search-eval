"""Anchor regression gate — the CI hook for shipping index/ranking changes.

Compares the anchor set's hit rates between a candidate run and a baseline run for the
system under test. hit@k must not drop by more than --tolerance (absolute); hit@1 is
reported but does not gate by default (enable with --gate-hit1).

Exit codes: 0 = pass, 1 = regression (block the release), 2 = missing/invalid input.

Usage:
  python -m scripts.anchor_gate --run results/run_NEW --baseline results/run_OLD [--backend octen]
  python -m scripts.anchor_gate ... --tolerance 0.02 --gate-hit1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.report import anchor_hit_rates


def load_anchors(run: Path, backend: str) -> dict[str, dict]:
    p = run / "anchors.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["backend"] == backend:
                out[r["qid"]] = r
    return out


def rates(recs: dict[str, dict]) -> tuple[float, float, int]:
    return anchor_hit_rates(list(recs.values()))


def gate(run_recs: dict[str, dict], base_recs: dict[str, dict],
         tolerance: float, gate_hit1: bool) -> tuple[bool, list[str]]:
    """Returns (passed, report lines). Compared on the intersection of qids so set growth doesn't skew."""
    common = sorted(set(run_recs) & set(base_recs))
    lines = []
    if not common:
        return False, ["no common anchor qids between run and baseline"]
    run_c = {q: run_recs[q] for q in common}
    base_c = {q: base_recs[q] for q in common}
    h1_new, hk_new, n = rates(run_c)
    h1_old, hk_old, _ = rates(base_c)
    lines.append(f"anchors compared: {n} (run has {len(run_recs)}, baseline has {len(base_recs)})")
    lines.append(f"hit@k: {hk_old:.1%} → {hk_new:.1%} ({hk_new - hk_old:+.1%})")
    lines.append(f"hit@1: {h1_old:.1%} → {h1_new:.1%} ({h1_new - h1_old:+.1%})")
    flips = [q for q in common if base_c[q]["hit_at_k"] and not run_c[q]["hit_at_k"]]
    if flips:
        lines.append(f"hit@k regressions ({len(flips)}): " + ", ".join(flips[:20]))
    # epsilon: hit rates are ratios of small integer counts — 0.72-0.70 > 0.02 in floats,
    # which used to block a release whose drop EQUALED the configured tolerance
    eps = 1e-9
    passed = (hk_old - hk_new) <= tolerance + eps
    if gate_hit1:
        passed = passed and (h1_old - h1_new) <= tolerance + eps
    return passed, lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--backend", default=None, help="defaults to run_meta.json 'ours'")
    ap.add_argument("--tolerance", type=float, default=0.0, help="allowed absolute drop in hit rate")
    ap.add_argument("--gate-hit1", action="store_true", help="also gate on hit@1 (default: report only)")
    args = ap.parse_args()

    run, base = Path(args.run), Path(args.baseline)
    backend = args.backend
    if not backend:
        meta_p = run / "run_meta.json"
        if not meta_p.exists():
            print("run_meta.json missing and no --backend given", file=sys.stderr)
            sys.exit(2)
        backend = json.loads(meta_p.read_text(encoding="utf-8"))["ours"]

    run_recs, base_recs = load_anchors(run, backend), load_anchors(base, backend)
    if not run_recs or not base_recs:
        print(f"missing anchors.jsonl records for backend={backend} "
              f"(run: {len(run_recs)}, baseline: {len(base_recs)})", file=sys.stderr)
        sys.exit(2)

    passed, lines = gate(run_recs, base_recs, args.tolerance, args.gate_hit1)
    print(f"ANCHOR GATE — backend={backend}, tolerance={args.tolerance:.1%}")
    for l in lines:
        print("  " + l)
    print("RESULT:", "PASS" if passed else "FAIL — anchor regression, block the release")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
