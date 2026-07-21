"""Convert open-source QA benchmarks (SimpleQA, FreshQA) into our query-set JSONL schema.

Produces three independently selectable eval sets (pick one via --queries at run time):
  data/simpleqa_verified.jsonl   Google DeepMind verified subset (1,001)
  data/simpleqa_full.jsonl       OpenAI original SimpleQA (4,326)
  data/freshqa.jsonl             FreshQA (~658; freshness-sensitive, multi-answer)

Every record carries meta.benchmark ("SimpleQA"/"FreshQA") so agent_eval routes it to the
official grader (src/grading.py). Self-built sets have no meta.benchmark and keep gold/rubric grading.

Source CSVs are read from data/datasets/{simpleqa,freshqa}/ (checked into the repo). Refresh them
from the official sources if needed:
  SimpleQA original : https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv
  SimpleQA verified : https://huggingface.co/datasets/google/simpleqa-verified
  FreshQA           : https://github.com/freshllms/freshqa (Google Sheets export)

Usage: python -m scripts.load_benchmarks [--which simpleqa_verified,simpleqa_full,freshqa]
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path

from scripts.adapt_benchmarks import SIMPLEQA_VERTICAL  # single source of truth for topic→vertical

DATA = Path(__file__).parent.parent / "data"
DS = DATA / "datasets"

# FreshQA fact_type → our freshness taxonomy
FACT_FRESHNESS = {"never-changing": "evergreen", "slow-changing": "recent",
                  "fast-changing": "realtime", "real-time": "realtime"}


def _sqa_record(qid, problem, answer, topic, answer_type, urls, variant, extra=None):
    return {
        "qid": qid, "query": problem, "intent": "informational", "difficulty": "tail",
        "freshness": "evergreen", "vertical": SIMPLEQA_VERTICAL.get(topic, "general"),
        "language": "en", "form": "natural_question", "source": "public_benchmark",
        "gold": {"answer": answer},
        "meta": {"benchmark": "SimpleQA", "variant": variant, "benchmark_topic": topic,
                 "answer_type": answer_type, "reference_urls": urls, **(extra or {})},
    }


def load_simpleqa_verified() -> list[dict]:
    out = []
    with open(DS / "simpleqa/simpleqa_verified.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q, a = (row.get("problem") or "").strip(), (row.get("answer") or "").strip()
            if not q or not a:
                continue
            urls = [u for u in (row.get("urls") or "").replace("\n", ",").split(",") if u.strip()]
            out.append(_sqa_record(
                f"simpleqa-v-{row.get('original_index') or len(out)}", q, a,
                (row.get("topic") or "").strip(), (row.get("answer_type") or "").strip(),
                urls, "verified",
                {"multi_step": row.get("multi_step"), "requires_reasoning": row.get("requires_reasoning")}))
    return out


def load_simpleqa_full() -> list[dict]:
    out = []
    with open(DS / "simpleqa/simpleqa_original.csv", newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            q, a = (row.get("problem") or "").strip(), (row.get("answer") or "").strip()
            if not q or not a:
                continue
            meta = {}
            try:
                meta = ast.literal_eval(row.get("metadata") or "{}")
            except (ValueError, SyntaxError):
                pass
            out.append(_sqa_record(
                f"simpleqa-{i}", q, a, meta.get("topic", ""), meta.get("answer_type", ""),
                list(meta.get("urls", [])), "full"))
    return out


def _freshqa_rows():
    """FreshQA official CSV: 2-line warning header, then the real header."""
    with open(DS / "freshqa/freshqa.csv", newline="", encoding="utf-8") as f:
        lines = f.readlines()
    # locate the real header row (starts with 'id,split,question')
    start = next((i for i, l in enumerate(lines) if l.startswith("id,split,question")), None)
    if start is None:
        sys.exit(f"FreshQA header row (starting 'id,split,question') not found in {DS/'freshqa/freshqa.csv'} "
                 "— the CSV format may have changed; re-download from the official source.")
    return list(csv.DictReader(lines[start:]))


def load_freshqa() -> list[dict]:
    out = []
    for row in _freshqa_rows():
        q = (row.get("question") or "").strip()
        # csv.DictReader fills missing trailing columns with None (restval), so guard with `or ""`
        # before .strip() — a CSV export that omits empty trailing cells would otherwise crash.
        answers = [(row.get(f"answer_{i}") or "").strip() for i in range(10)
                   if (row.get(f"answer_{i}") or "").strip()]
        if not q or not answers:
            continue
        fact_type = (row.get("fact_type") or "").strip()
        false_premise = (row.get("false_premise") or "").strip().upper() == "TRUE"
        num_hops = (row.get("num_hops") or "").strip()
        rid = (row.get("id") or str(len(out))).strip()
        out.append({
            "qid": f"freshqa-{rid}", "query": q, "intent": "informational",
            "difficulty": "tail" if num_hops == "multi-hop" else "torso",
            "freshness": FACT_FRESHNESS.get(fact_type, "recent"),
            "vertical": "news" if fact_type in ("fast-changing", "real-time") else "general",
            "language": "en", "form": "natural_question", "source": "public_benchmark",
            # gold.answer = primary answer (keeps it "gradeable"); FreshEval uses the joined list.
            "gold": {"answer": answers[0], "answers": answers},
            # note: the answer list lives in gold.answers (the grader's source of truth); not
            # duplicated here. fact_type/num_hops/false_premise/effective_year feed FreshQA breakdowns.
            "meta": {"benchmark": "FreshQA", "fact_type": fact_type, "num_hops": num_hops,
                     "false_premise": false_premise,
                     "effective_year": (row.get("effective_year") or "").strip(),
                     "split": (row.get("split") or "").strip().upper(),
                     "next_review": (row.get("next_review") or "").strip()},
        })
    return out


LOADERS = {
    "simpleqa_verified": (load_simpleqa_verified, "simpleqa_verified.jsonl"),
    "simpleqa_full": (load_simpleqa_full, "simpleqa_full.jsonl"),
    "freshqa": (load_freshqa, "freshqa.jsonl"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="simpleqa_verified,simpleqa_full,freshqa",
                    help="comma-separated subset of: " + ", ".join(LOADERS))
    args = ap.parse_args()
    for name in [w.strip() for w in args.which.split(",") if w.strip()]:
        if name not in LOADERS:
            sys.exit(f"unknown benchmark '{name}'; choose from {list(LOADERS)}")
        loader, fname = LOADERS[name]
        recs = loader()
        out = DATA / fname
        out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")
        print(f"{name:18} → data/{fname}  ({len(recs)} queries)")


if __name__ == "__main__":
    main()
