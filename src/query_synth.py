"""Topic → directly evaluable query set (two-stage synthesis).

Stage 1: topic → facets (sub-facets + entity lists)
Stage 2: generate small batches per facet × form × difficulty quota cell → diversity is
         guaranteed by structure
Post-processing: dedup (within set + against the existing main set) → schema enum validation
→ source=synthetic label → distribution report

Anchors: candidate_gold_urls produced by the LLM during synthesis only go into meta and
     **never become gold directly**; resolution is unified through src/gold_resolver.py
     (SERP candidates + LLM adjudication + HTTP liveness).

Hard rule: a synthesized set must get per-query rubrics (rubric_gen --only-missing) before
     any judge run. A single-topic synthetic set is highly homogeneous; under the generic
     rubric the judge's discriminative power collapses (position_conflict 35% in the 2026-07
     trial — close to coin-flipping), making win-rate conclusions untrustworthy.

Usage:
  python -m src.query_synth --topic "cross-border e-commerce logistics" --n 40 \\
      --lang "zh-CN=0.7,en=0.3" --out data/synth_ecomm_logistics.jsonl \\
      --dedup-against data/seed_queries.jsonl
  python -m src.gold_resolver --queries data/synth_ecomm_logistics.jsonl   # then resolve anchors
"""
from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import re
import time
from pathlib import Path

from .backends import _load_dotenv
from .common import today_utc, write_jsonl_atomic
from .llm import call_llm_json

_load_dotenv()

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "query_synth.md"
FORMS = ["keyword", "natural_question", "agent_generated"]
DIFFS = ["head", "torso", "tail"]
# Default quota weights: agent-customer oriented → slightly more agent_generated; tail is the
# difficulty tier with the highest eval value
FORM_W = {"keyword": 0.3, "natural_question": 0.3, "agent_generated": 0.4}
DIFF_W = {"head": 0.2, "torso": 0.4, "tail": 0.4}
VALID = {
    "intent": {"navigational", "informational", "transactional", "multi_hop_research"},
    "difficulty": set(DIFFS),
    "freshness": {"evergreen", "recent", "realtime"},
    "form": set(FORMS),
}


def _sections() -> dict[str, str]:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    out = {}
    for key in ["STAGE 1 SYSTEM PROMPT", "STAGE 1 USER PROMPT", "STAGE 2 SYSTEM PROMPT", "STAGE 2 USER PROMPT"]:
        seg = text.split(f"## {key}")[1]
        seg = re.split(r"\n## |\n---", seg)[0]
        first_line = seg.split("\n", 1)[0]
        # header may carry an annotation in ASCII or fullwidth parens; either way drop that line
        out[key] = seg.split("\n", 1)[1].strip() if ("(" in first_line or "（" in first_line) else seg.strip()
    return out


def _call_llm(system: str, user: str, max_tokens: int = 2048) -> dict:
    # via src/llm.py provider switch (shared with judge/rubric_gen/gold_resolver);
    # retry once on truncation/network errors
    last_err = None
    for _ in range(2):
        try:
            return call_llm_json(system, user, model=os.environ.get("RUBRIC_MODEL", "claude-sonnet-4-6"),
                                 max_tokens=max_tokens)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise last_err


def _norm(q: str) -> str:
    return re.sub(r"[\s\W]+", "", q.lower())


def _existing_block(existing: list[str]) -> str:
    """Render the 'already generated for this facet' list injected into Stage 2 (hard topic-overlap constraint)."""
    return "\n".join(f"- {q}" for q in existing) if existing else "(none)"


def is_dup(query: str, pool: list[str], threshold: float = 0.85) -> bool:
    n = _norm(query)
    for p in pool:
        if n == p:
            return True
        # cheap gates before the O(len²) ratio: length prefilter + quick_ratio upper bound
        if abs(len(p) - len(n)) / max(len(p), len(n), 1) > 1 - threshold:
            continue
        sm = difflib.SequenceMatcher(None, n, p)
        if sm.quick_ratio() >= threshold and sm.ratio() >= threshold:
            return True
    return False


def build_quota(n: int, n_facets: int, form_w: dict[str, float] | None = None) -> list[tuple[str, str, int]]:
    """(form, difficulty, TOTAL count) cells that sum to exactly n, via largest-remainder
    apportionment. n_facets is accepted for API stability but no longer affects the split.

    History: two earlier versions had starvation bugs — a 0.34 threshold wiped whole
    forms/difficulties at small n, and the backstop cells added to fix that were themselves
    starved because generation truncated at n in weight order. Exact apportionment removes the
    truncation entirely; every form and difficulty tier with weight > 0 is guaranteed ≥ 1.
    """
    fw = form_w or FORM_W
    cells = [(f, d, n * fw[f] * DIFF_W[d]) for f in FORMS for d in DIFFS if fw[f] > 0]
    counts = {(f, d): int(c) for f, d, c in cells}
    remaining = n - sum(counts.values())
    for f, d, c in sorted(cells, key=lambda x: x[2] - int(x[2]), reverse=True):
        if remaining <= 0:
            break
        counts[(f, d)] += 1
        remaining -= 1

    def tier_total(pred) -> int:
        return sum(v for k, v in counts.items() if pred(k))

    def donate_to(key) -> None:
        donor = max(counts, key=lambda k: counts[k] if k != key else -1)
        counts[donor] -= 1
        counts[key] += 1

    for f in FORMS:
        if fw.get(f, 0) > 0 and tier_total(lambda k: k[0] == f) == 0:
            donate_to((f, max(DIFFS, key=lambda d: DIFF_W[d])))
    for d in DIFFS:
        if tier_total(lambda k: k[1] == d) == 0:
            best_f = max((f for f in FORMS if fw.get(f, 0) > 0), key=lambda f: fw[f])
            donate_to((best_f, d))

    quota = [(f, d, c) for (f, d), c in counts.items() if c > 0]
    quota.sort(key=lambda x: fw[x[0]] * DIFF_W[x[1]], reverse=True)
    return quota


def parse_form_weights(spec: str) -> dict[str, float]:
    """Parse 'keyword=0.2,natural_question=0.2,agent_generated=0.6' into a weights dict."""
    fw = dict(FORM_W)
    for part in spec.split(","):
        k, _, v = part.partition("=")
        k = k.strip()
        if k not in FORM_W:
            raise ValueError(f"unknown form: {k}")
        fw[k] = float(v)
    return fw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--topic-note", default="", help="extra context, e.g. target users, scope limits")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--lang", default="zh-CN=0.5,en=0.5")
    ap.add_argument("--n-facets", type=int, default=6)
    ap.add_argument("--vertical", default="general", help="vertical label for every query of this topic")
    ap.add_argument("--form-weights", default=None,
                    help='override form quota weights, e.g. "keyword=0.2,natural_question=0.2,agent_generated=0.6"')
    ap.add_argument("--out", required=True)
    ap.add_argument("--dedup-against", nargs="*", default=[], help="existing query-set JSONLs to dedup the output against")
    args = ap.parse_args()
    form_w = parse_form_weights(args.form_weights) if args.form_weights else None

    sec = _sections()
    today = today_utc()

    # ---- Stage 1: facets
    s1_sys = sec["STAGE 1 SYSTEM PROMPT"].replace("{current_date}", today).replace("{n_facets}", str(args.n_facets))
    s1_user = sec["STAGE 1 USER PROMPT"].replace("{topic}", args.topic).replace("{topic_note}", args.topic_note or "none")
    facets = _call_llm(s1_sys, s1_user)["facets"]
    print(f"Stage 1: {len(facets)} facets")
    for f in facets:
        print(f"  - {f['name']} ({f['typical_freshness']}): {', '.join(f['entities'][:4])}")

    # ---- dedup pool
    pool: list[str] = []
    for dfile in args.dedup_against:
        for line in Path(dfile).read_text(encoding="utf-8").splitlines():
            if line.strip():
                pool.append(_norm(json.loads(line)["query"]))

    # ---- Stage 2: generate per quota cell
    quota = build_quota(args.n, len(facets), form_w)
    print(f"Stage 2: {len(quota)} cells summing to {sum(c for _, _, c in quota)} queries "
          f"across {len(facets)} facets", flush=True)

    slug = re.sub(r"[^a-z0-9]+", "-", args.topic.lower()).strip("-")[:20] or "topic"
    s2_sys = sec["STAGE 2 SYSTEM PROMPT"].replace("{current_date}", today)
    results, n_dup, n_invalid, seq = [], 0, 0, 1

    # Each cell carries its TOTAL count; facets are consumed round-robin inside the cell so
    # every facet contributes (later facets — tools/products, the main navigational source —
    # used to starve when generation truncated at n in cell order)
    facet_seen: dict[str, list[str]] = {}  # facet → queries generated so far, injected into later cells to prevent topic clustering
    for form, diff, cell_total in quota:
        cell_left = cell_total
        per_call = max(1, math.ceil(cell_total / len(facets)))
        attempts = 0
        fi = 0
        while cell_left > 0 and attempts < len(facets) * 2 and len(results) < args.n:
            facet = facets[fi % len(facets)]
            fi += 1
            attempts += 1
            existing = facet_seen.setdefault(facet["name"], [])
            user = (
                sec["STAGE 2 USER PROMPT"]
                .replace("{topic}", args.topic)
                .replace("{facet_name}", facet["name"])
                .replace("{facet_description}", facet["description"])
                .replace("{entities}", ", ".join(facet["entities"]))
                .replace("{n}", str(min(per_call, cell_left)))
                .replace("{form}", form)
                .replace("{difficulty}", diff)
                .replace("{lang_spec}", args.lang)
                .replace("{facet_existing}", _existing_block(existing))
            )
            try:
                batch = _call_llm(s2_sys, user)["queries"]
            except Exception as e:  # noqa: BLE001
                print(f"  cell ({facet['name']}/{form}/{diff}) failed: {e}")
                continue
            for item in batch:
                if cell_left <= 0 or len(results) >= args.n:
                    break
                # schema enum validation: form/difficulty come from the quota cell (the model occasionally improvises)
                item["form"], item["difficulty"] = form, diff
                if any(item.get(k) not in VALID[k] for k in ("intent", "freshness")):
                    n_invalid += 1
                    continue
                if is_dup(item["query"], pool):
                    n_dup += 1
                    continue
                pool.append(_norm(item["query"]))
                rec = {
                    "qid": f"q-synth-{slug}-{seq:04d}",
                    "query": item["query"],
                    "intent": item["intent"],
                    "difficulty": diff,
                    "freshness": item["freshness"],
                    "vertical": args.vertical,  # topic-determined, passed via --vertical
                    "language": item.get("language", "zh-CN"),
                    "form": form,
                    "source": "synthetic",
                    "meta": {
                        "topic": args.topic,
                        "facet": facet["name"],
                        "synth_at": today,
                        **({"candidate_gold_urls": item["candidate_gold_urls"]}
                           if item.get("candidate_gold_urls") else {}),
                    },
                }
                results.append(rec)
                existing.append(item["query"])
                seq += 1
                cell_left -= 1
            time.sleep(0.3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(out, results)

    # ---- distribution report
    print(f"\nProduced {len(results)} (dropped {n_dup} dups, {n_invalid} validation failures) → {out}")
    for key in ["form", "difficulty", "freshness", "intent", "language"]:
        from collections import Counter
        dist = Counter(r[key] for r in results)
        print(f"  {key:10s}: " + "  ".join(f"{k}={v}" for k, v in dist.most_common()))
    n_cand = sum(1 for r in results if r["meta"].get("candidate_gold_urls"))
    print(f"  {n_cand} entries carry anchor candidates")
    print("\nNext steps (order is mandatory, none skippable):")
    print(f"  1. python -m src.gold_resolver --queries {out}")
    print(f"  2. python -m src.rubric_gen --queries {out} --only-missing")
    print("     ^ hard rule: a synth set must get per-query rubrics before any judge run —")
    print("       under the generic rubric a homogeneous topic measured 35% position_conflict")
    print("       (2026-07); win-rate conclusions are untrustworthy")
    print(f"  3. python -m src.run_eval --queries {out} ...")


if __name__ == "__main__":
    main()
