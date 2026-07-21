"""SimpleQA / FreshQA → data/benchmark_queries.jsonl adaptation script (Task 4.3, reproducible).

Data sources (download to /tmp, or point --simpleqa/--freshqa at other paths):
  SimpleQA: https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv
  FreshQA (2026-04-21): https://docs.google.com/spreadsheets/d/1_8mi-yuK30mvoDJu1KQXD6ODem7MKMcIgVAwDSzJkjM/export?format=csv

Selection rules:
- SimpleQA: first 6 per topic (question ≤200 chars with an answer), 60 total. Evergreen facts,
  gold.answer directly usable as anchors; metadata.urls go into meta.reference_urls
  (evidence pages, not navigational gold).
- FreshQA: split=TEST, false_premise=FALSE, has answer_0, question ≤200 chars.
  12 fast-changing / 12 slow-changing / 6 never-changing, 30 total;
  fast-changing answers go stale over time, so no gold.answer — the original answer is kept in
  meta.reference_answer with meta.freshqa_answer_asof; slow/never get gold.answer
  (slow additionally marked meta.gold_volatility).
- Everything is meta.auto_labeled=true; labels are rule-mapped (not LLM); the user reviews
  before merging into the main set.

Usage: python -m scripts.adapt_benchmarks [--simpleqa PATH] [--freshqa PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path

FRESHQA_VERSION = "2026-04-21"

SIMPLEQA_VERTICAL = {
    "Science and technology": "academic",
    "History": "academic",
    "Geography": "general",
    "Other": "general",
    "Politics": "news",
    "Sports": "entertainment",
    "Art": "entertainment",
    "Music": "entertainment",
    "TV shows": "entertainment",
    "Video games": "entertainment",
}

NEWS_KW = ("president", "prime minister", "election", "senator", "governor", "congress")
ENT_KW = ("team", "cup", "league", "champion", "olympic", "nba", "nfl", "tournament", "grand slam")
FIN_KW = ("stock", "ceo", "company", "billion", "market cap", "acquisition")


def freshqa_vertical(question: str) -> str:
    ql = question.lower()
    if any(k in ql for k in NEWS_KW):
        return "news"
    if any(k in ql for k in ENT_KW):
        return "entertainment"
    if any(k in ql for k in FIN_KW):
        return "finance"
    return "general"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simpleqa", default="/tmp/simpleqa.csv")
    ap.add_argument("--freshqa", default="/tmp/freshqa.csv")
    ap.add_argument("--out", default="data/benchmark_queries.jsonl")
    ap.add_argument("--per-topic", type=int, default=6)
    args = ap.parse_args()

    records: list[dict] = []
    seqs: dict[str, int] = defaultdict(lambda: 2000)  # benchmark set starts at q-{vertical}-2001, clear of the seed set

    def add(query: str, *, intent: str, difficulty: str, freshness: str, vertical: str,
            gold: dict | None, meta: dict) -> None:
        seqs[vertical] += 1
        rec = {
            "qid": f"q-{vertical}-{seqs[vertical]:04d}",
            "query": query,
            "intent": intent,
            "difficulty": difficulty,
            "freshness": freshness,
            "vertical": vertical,
            "language": "en",
            "form": "natural_question",
            "source": "public_benchmark",
        }
        if gold:
            rec["gold"] = gold
        rec["meta"] = {"auto_labeled": True, **meta}
        records.append(rec)

    # --- SimpleQA ---
    with open(args.simpleqa, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        md = ast.literal_eval(r["metadata"])
        q, a = r["problem"].strip(), r["answer"].strip()
        if q and a and len(q) <= 200 and len(by_topic[md["topic"]]) < args.per_topic:
            by_topic[md["topic"]].append({"q": q, "a": a, "urls": md.get("urls", [])[:3], "topic": md["topic"]})
    n_sqa = 0
    for topic in sorted(by_topic):
        for item in by_topic[topic]:
            add(item["q"],
                intent="informational", difficulty="tail", freshness="evergreen",
                vertical=SIMPLEQA_VERTICAL.get(topic, "general"),
                gold={"answer": item["a"]},
                meta={"benchmark": "SimpleQA", "benchmark_topic": topic,
                      "reference_urls": item["urls"]})
            n_sqa += 1

    # --- FreshQA ---
    with open(args.freshqa, encoding="utf-8") as f:
        lines = list(csv.reader(f))
    hdr = lines[2]
    frows = [dict(zip(hdr, r)) for r in lines[3:] if len(r) >= len(hdr) and r[0].strip()]
    eligible = [r for r in frows
                if r["split"] == "TEST" and r["false_premise"] == "FALSE"
                and r["answer_0"].strip() and len(r["question"].strip()) <= 200]
    quota = {"fast-changing": 12, "slow-changing": 12, "never-changing": 6}
    # multi-hop coverage: include up to half multi-hop per fact_type first, fill with one-hop
    eligible.sort(key=lambda r: 0 if r["num_hops"] == "multi-hop" else 1)
    taken: dict[str, int] = defaultdict(int)
    taken_multi: dict[str, int] = defaultdict(int)
    n_fq = 0
    for r in eligible:
        ft = r["fact_type"]
        if taken[ft] >= quota.get(ft, 0):
            continue
        if r["num_hops"] == "multi-hop" and taken_multi[ft] >= quota.get(ft, 0) // 2:
            continue
        if r["num_hops"] == "multi-hop":
            taken_multi[ft] += 1
        taken[ft] += 1
        q = r["question"].strip()
        multi = r["num_hops"] == "multi-hop"
        freshness = "evergreen" if ft == "never-changing" else "recent"
        gold, meta_extra = None, {}
        if ft == "fast-changing":
            # the answer goes stale over time: no anchor use, original answer kept for reference only
            meta_extra = {"reference_answer": r["answer_0"].strip(),
                          "freshqa_answer_asof": FRESHQA_VERSION}
        else:
            gold = {"answer": r["answer_0"].strip()}
            if ft == "slow-changing":
                meta_extra = {"gold_volatility": "slow", "freshqa_answer_asof": FRESHQA_VERSION}
        add(q,
            intent="multi_hop_research" if multi else "informational",
            difficulty="torso", freshness=freshness,
            vertical=freshqa_vertical(q), gold=gold,
            meta={"benchmark": f"FreshQA-{FRESHQA_VERSION}", "fact_type": ft,
                  "num_hops": r["num_hops"], **meta_extra})
        n_fq += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    print(f"SimpleQA {n_sqa} + FreshQA {n_fq} = {len(records)} → {out}")

    from src.query_intake import quota_report
    print("\n" + quota_report(records))
    anchors = sum(1 for r in records if r.get("gold", {}).get("answer"))
    no_gold = sum(1 for r in records if not r.get("gold"))
    print(f"\nWith gold.answer (usable as QA anchors): {anchors}; without gold (fast-changing, answer kept in meta.reference_answer): {no_gold}")
    print("Note: everything is meta.auto_labeled=true; the user reviews before merging into the main query set.")


if __name__ == "__main__":
    main()
