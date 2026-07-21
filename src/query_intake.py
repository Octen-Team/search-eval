"""Raw query intake: dedup → LLM pre-labeling per schema → JSONL + quota distribution table.

Usage:
  python -m src.query_intake --input raw_queries.txt --out data/intake_batch1.jsonl \
      --dedup-against data/seed_queries.jsonl
  python -m src.query_intake --report data/intake_batch1.jsonl        # six-dimension quota distribution only
  python -m src.query_intake --input raw.csv --out ... --batch-size 10

Mechanics:
- Input is txt (one query per line) or csv (the "query" column if headered, else the first column).
- Dedup: normalized exact + near-dup (SequenceMatcher ratio ≥ 0.92), both against the
  --dedup-against file and within the batch; dropped entries are printed with the reason.
- LLM batch pre-labeling (one call per batch-size queries, via the src/llm.py provider switch);
  label enums stay consistent with schema/query_schema.json; results carry meta.auto_labeled=true
  for human spot-checking.
- qids are auto-numbered q-{vertical}-{seq:04d}, continuing from the max existing sequence in
  the dedup-against and output files.
- Consumes LLM tokens: prints the estimated call count up front and requires --yes to skip the
  interactive confirmation (guards against accidental burn).
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from .backends import _load_dotenv
from .common import load_jsonl, write_jsonl_atomic
from .llm import call_llm_json

_load_dotenv()

SLICE_KEYS = ["intent", "difficulty", "freshness", "vertical", "language", "form"]

ENUMS = {
    "intent": ["navigational", "informational", "transactional", "multi_hop_research"],
    "difficulty": ["head", "torso", "tail"],
    "freshness": ["evergreen", "recent", "realtime"],
    "vertical": ["tech_code", "academic", "news", "ecommerce", "local",
                 "medical_legal", "finance", "entertainment", "general"],
    "form": ["keyword", "natural_question", "agent_generated"],
}

LABEL_SYSTEM = """You are a search-eval data labeling expert. Label every given search query against this taxonomy:

- intent: navigational (find a specific site/page) | informational (find facts/knowledge) | transactional (find an executable action) | multi_hop_research (needs synthesis across sources)
- difficulty: head (popular high-frequency) | torso (mid-frequency) | tail (long-tail / rare entities / deep pages)
- freshness: evergreen (answer stays stable long-term) | recent (last few weeks-months) | realtime (minute/hour granularity)
- vertical: tech_code | academic | news | ecommerce | local | medical_legal | finance | entertainment | general
- language: BCP-47 (e.g. zh-CN / en / ja); cross-language (a Chinese query about English content) uses zh-CN>en
- form: keyword (keyword style) | natural_question (natural-language question) | agent_generated (operators / keyword stacking / machine-shaped)

Output JSON only: {"labels": [{"i": <index>, "intent": "...", "difficulty": "...", "freshness": "...",
"vertical": "...", "language": "...", "form": "..."}, ...]}, one entry per query, no omissions."""


def read_raw(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if not rows:
            return []
        header = [c.strip().lower() for c in rows[0]]
        if "query" in header:
            idx = header.index("query")
            return [r[idx].strip() for r in rows[1:] if len(r) > idx and r[idx].strip()]
        # no 'query' column: take the first column, but a headered CSV whose query column
        # just isn't named 'query' must not ingest its header cell as a query
        body = rows
        try:
            sample = "\n".join(",".join(r) for r in rows[:10])
            if csv.Sniffer().has_header(sample):
                body = rows[1:]
        except csv.Error:
            pass
        return [r[0].strip() for r in body if r and r[0].strip()]
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _norm(q: str) -> str:
    return " ".join(q.lower().split())


def dedup(raw: list[str], against: list[str], near_threshold: float = 0.92) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Returns (kept list, dropped [(query, reason, matched-against)]). Near-dup uses SequenceMatcher; O(n²), fine up to ~10k."""
    kept, dropped = [], []
    seen = {_norm(a) for a in against}
    kept_norm: list[str] = []
    against_norm = [_norm(a) for a in against]
    for q in raw:
        n = _norm(q)
        if n in seen:
            dropped.append((q, "exact", "existing/batch"))
            continue
        near = None
        for other in against_norm + kept_norm:
            if abs(len(other) - len(n)) / max(len(other), len(n), 1) > 0.2:
                continue
            sm = SequenceMatcher(None, n, other)
            if sm.quick_ratio() < near_threshold:  # cheap upper bound before the O(len²) ratio
                continue
            if sm.ratio() >= near_threshold:
                near = other
                break
        if near:
            dropped.append((q, "near-dup", near[:50]))
            continue
        seen.add(n)
        kept_norm.append(n)
        kept.append(q)
    return kept, dropped


def label_batch(queries: list[str]) -> list[dict]:
    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(queries))
    out = call_llm_json(LABEL_SYSTEM, f"Queries to label:\n{numbered}",
                        model=os.environ.get("RUBRIC_MODEL", "claude-sonnet-4-6"),
                        max_tokens=4096)
    labels = {rec["i"]: rec for rec in out.get("labels", [])}
    results = []
    for i, q in enumerate(queries):
        rec = labels.get(i + 1, {})
        lab = {}
        for key in ("intent", "difficulty", "freshness", "vertical", "form"):
            val = rec.get(key, "")
            lab[key] = val if val in ENUMS[key] else None
        lab["language"] = rec.get("language") or None
        results.append({"query": q, **lab})
    return results


def next_seq(existing: list[dict]) -> dict[str, int]:
    seqs: dict[str, int] = defaultdict(int)
    for q in existing:
        parts = q.get("qid", "").rsplit("-", 1)
        if len(parts) == 2 and parts[1].isdigit():
            prefix = parts[0].removeprefix("q-")
            seqs[prefix] = max(seqs[prefix], int(parts[1]))
    return seqs


def quota_report(records: list[dict]) -> str:
    lines = [f"Six-dimension quota distribution (n={len(records)})"]
    for key in SLICE_KEYS:
        c = Counter(r.get(key) or "?" for r in records)
        cells = "  ".join(f"{v}={n}({n/len(records):.0%})" for v, n in c.most_common())
        lines.append(f"  {key:11s}: {cells}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="raw query file (txt: one per line / csv: query column)")
    ap.add_argument("--out", help="output JSONL")
    ap.add_argument("--dedup-against", default=None, help="existing query-set JSONL for dedup and qid continuation")
    ap.add_argument("--batch-size", type=int, default=10, help="queries labeled per LLM call")
    ap.add_argument("--report", default=None, metavar="JSONL", help="only print the six-dimension quota table of the given JSONL")
    ap.add_argument("--yes", action="store_true", help="skip the LLM-consumption confirmation")
    args = ap.parse_args()

    if args.report:
        print(quota_report(load_jsonl(args.report)))
        return

    if not args.input or not args.out:
        ap.error("--input and --out are required (or use --report)")

    raw = read_raw(Path(args.input))
    existing = load_jsonl(args.dedup_against) if args.dedup_against else []
    # the output file also participates in dedup + qid continuation — two batches labeled on
    # different days against the same seed set must not mint colliding qids
    existing_out = load_jsonl(args.out)
    kept, dropped = dedup(raw, [q["query"] for q in existing + existing_out])
    print(f"Input {len(raw)} → {len(kept)} after dedup (exact {sum(1 for d in dropped if d[1]=='exact')}, near-dup {sum(1 for d in dropped if d[1]=='near-dup')})")
    for q, why, ref in dropped[:10]:
        print(f"  [drop:{why}] {q[:60]}  ~  {ref}")

    n_calls = (len(kept) + args.batch_size - 1) // args.batch_size
    print(f"Estimated LLM calls: {n_calls} (batch={args.batch_size})")
    if not args.yes:
        resp = input("Continue? [y/N] ").strip().lower()
        if resp != "y":
            print("Cancelled")
            return

    seqs = next_seq(existing + existing_out)
    out_records = list(existing_out)  # append semantics: relabeling into the same file continues it
    for i in range(0, len(kept), args.batch_size):
        batch = kept[i:i + args.batch_size]
        try:
            labeled = label_batch(batch)
        except Exception as e:  # noqa: BLE001
            print(f"  batch {i//args.batch_size + 1} FAILED: {e} (batch skipped; rerun later)")
            continue
        for rec in labeled:
            vert = rec["vertical"] or "general"
            seqs[vert] += 1
            missing = [k for k in ("intent", "difficulty", "freshness", "vertical", "language", "form") if not rec.get(k)]
            out_records.append({
                "qid": f"q-{vert}-{seqs[vert]:04d}",
                "query": rec["query"],
                "intent": rec["intent"] or "informational",
                "difficulty": rec["difficulty"] or "torso",
                "freshness": rec["freshness"] or "evergreen",
                "vertical": vert,
                "language": rec["language"] or "und",
                "form": rec["form"] or "keyword",
                "source": "traffic_log",
                "meta": {"auto_labeled": True,
                         **({"label_fallback_fields": missing} if missing else {})},
            })
        print(f"  batch {i//args.batch_size + 1}/{n_calls} done ({len(out_records)} records)")

    out = Path(args.out)
    write_jsonl_atomic(out, out_records)
    print(f"\n→ {out}")
    print(quota_report(out_records))
    print("\nNote: labels are LLM pre-labels (meta.auto_labeled=true); human spot-checks required before "
          "merging into the main set. source defaults to traffic_log — adjust if the origin differs.")


if __name__ == "__main__":
    main()
