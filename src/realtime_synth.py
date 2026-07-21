"""Run-time realtime eval-set generation: produce a FRESH batch of realtime queries anchored
to what is happening today, immediately before an eval run.

Why a dedicated generator: realtime queries (live scores, prices, breaking news, today's
schedules) go stale within hours — a realtime slice baked into the main set is dead on
arrival. This tool probes the SERP oracle for today's actual events and has the LLM turn them
into schema-conformant queries, stamped with meta.synth_at and an expiry hint.

Operating rules:
- The output file is dated (data/realtime_YYYYMMDD.jsonl) and is NOT merged into the main set.
- Generate → rubric_gen --only-missing → run the eval, all on the same day. run_eval warns
  when it sees realtime queries synthesized on an earlier day.
- run_eval fetches all backends serially within an episode, so same-time-window comparability
  is already guaranteed.

Usage (before each run):
  python -m src.realtime_synth --n 12 --dedup-against data/main_queries.jsonl
  python -m src.rubric_gen --queries data/realtime_$(date -u +%Y%m%d).jsonl
  python -m src.run_eval --queries data/main_queries.jsonl data/realtime_$(date -u +%Y%m%d).jsonl ...
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .backends import _load_dotenv
from .common import load_jsonl, today_utc, write_jsonl_atomic
from .llm import call_llm_json
from .query_synth import _norm, is_dup  # shared dedup — thresholds must not drift between synthesizers

_load_dotenv()

# (category, probe query for today's events, default vertical)
PROBES = [
    ("news", "breaking news headlines right now world", "news"),
    ("sports", "live sports scores today major leagues matches", "news"),
    ("finance", "stock market today biggest movers index", "finance"),
    ("weather", "major weather events warnings today", "general"),
    ("tech", "tech product launch release announcement today", "tech_code"),
]

# News-focused probe set (--category news): wider event surface so large batches (n≈100)
# don't collapse onto the same handful of stories.
NEWS_PROBES = [
    ("world", "breaking news world headlines right now", "news"),
    ("politics", "political news today government policy announcement", "news"),
    ("geopolitics", "international conflict diplomacy summit news today", "news"),
    ("business", "business news today merger earnings announcement", "finance"),
    ("markets", "stock market crypto today biggest moves", "finance"),
    ("tech", "technology news today AI product launch", "tech_code"),
    ("sports", "sports news today match results transfers injuries", "news"),
    ("entertainment", "entertainment celebrity news today box office", "entertainment"),
    ("science_health", "science health news today study outbreak discovery", "news"),
    ("disaster_weather", "natural disaster extreme weather warning news today", "news"),
    ("china", "China news today economy policy companies", "news"),
    ("legal", "court ruling lawsuit verdict regulation news today", "news"),
]

VALID = {
    "intent": {"navigational", "informational", "transactional", "multi_hop_research"},
    "difficulty": {"head", "torso", "tail"},
    "form": {"keyword", "natural_question", "agent_generated"},
    "vertical": {"tech_code", "academic", "news", "ecommerce", "local",
                 "medical_legal", "finance", "entertainment", "general"},
}

SYSTEM = """You are building the REALTIME slice of a web-search eval set. Given today's date and \
live SERP material describing what is happening right now, write {n} search queries that a real \
user or LLM agent would plausibly issue TODAY, and whose correct answer changes within minutes to \
hours (live scores, market prices, breaking developments, today's schedules/availability).

Rules:
- Anchor every query in a REAL entity or event visible in the grounding material — never invent events.
- The answer must still be in motion: a finished, settled fact is 'recent', not realtime — skip it.
- Mix forms: keyword / natural_question / agent_generated (operators, dense noun stacking). \
Mostly English; at most ~20% Chinese where natural.
- Spread across the grounding categories (news / sports / finance / weather / tech).
- Label each query with EXACTLY these enum values:
  intent: navigational (find a specific page) | informational (find facts) | transactional \
(executable action) | multi_hop_research (needs multiple sources)
  difficulty: head (popular high-frequency) | torso (mid-frequency) | tail (long-tail/rare)
  form: keyword | natural_question | agent_generated
  vertical: tech_code|academic|news|ecommerce|local|medical_legal|finance|entertainment|general
  language: BCP-47 (en / zh-CN / zh-CN>en)

Output JSON only:
{"queries": [{"query": "...", "intent": "...", "difficulty": "...", "form": "...", "vertical": "...", "language": "..."}]}"""


def gather_grounding(probes: list[tuple] = PROBES, k: int = 5) -> str:
    """Probe the SERP oracle for today's live events; single probes may fail without aborting."""
    from .serp import serp_fetch
    blocks = []
    for cat, probe, _ in probes:
        try:
            items = serp_fetch(probe, "google", k)
            lines = [f"### {cat}"] + [f"- {it.title} | {it.snippet[:150]}" for it in items[:k]]
            blocks.append("\n".join(lines))
        except Exception as e:  # noqa: BLE001
            blocks.append(f"### {cat}: (probe failed: {e})")
    return "\n\n".join(blocks)


def build_records(items: list[dict], today: str, pool: list[str],
                  batch_id: str | None = None, start_seq: int = 1) -> list[dict]:
    """Validate labels, dedup, and stamp records. pool = normalized existing queries (mutated).
    batch_id namespaces the qids: a same-day regeneration must NOT reuse yesterday-hour's qids,
    or resume/rubrics silently attach old verdicts to new query texts.
    start_seq continues qid numbering across chunked generations within one batch."""
    batch_id = batch_id or today.replace("-", "")
    records, seq = [], start_seq
    for it in items:
        if not it.get("query") or any(it.get(k) not in VALID[k] for k in ("intent", "difficulty", "form", "vertical")):
            continue
        if is_dup(it["query"], pool):
            continue
        pool.append(_norm(it["query"]))
        records.append({
            "qid": f"q-realtime-{batch_id}-{seq:03d}",
            "query": it["query"],
            "intent": it["intent"],
            "difficulty": it["difficulty"],
            "freshness": "realtime",
            "vertical": it["vertical"],
            "language": it.get("language", "en"),
            "form": it["form"],
            "source": "synthetic",
            "meta": {"synth_at": today, "expires": "24h", "realtime_batch": True,
                     "auto_labeled": True},
        })
        seq += 1
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default=None, help="defaults to data/realtime_YYYYMMDD.jsonl")
    ap.add_argument("--dedup-against", nargs="*", default=["data/main_queries.jsonl"])
    ap.add_argument("--category", choices=["mixed", "news"], default="mixed",
                    help="news = hot-news-only batch with the wider NEWS_PROBES event surface")
    ap.add_argument("--chunk-size", type=int, default=20,
                    help="queries per LLM call; large n is chunked and each chunk sees a sample "
                         "of what earlier chunks produced (anti-clustering)")
    args = ap.parse_args()

    today = today_utc()
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")  # unique per regeneration
    out = Path(args.out) if args.out else Path(f"data/realtime_{today.replace('-', '')}.jsonl")

    pool: list[str] = []
    for f in args.dedup_against:
        pool.extend(_norm(rec["query"]) for rec in load_jsonl(f))

    probes = NEWS_PROBES if args.category == "news" else PROBES
    print(f"Probing today's events ({len(probes)} SERP probes)...", flush=True)
    grounding = gather_grounding(probes)
    categories = " / ".join(c for c, _, _ in probes)
    system_base = SYSTEM.replace(
        "Spread across the grounding categories (news / sports / finance / weather / tech).",
        f"Spread across the grounding categories ({categories})."
        + (" EVERY query must be about a live NEWS development (a story still unfolding today);"
           " skip evergreen topics." if args.category == "news" else ""))

    records: list[dict] = []
    n_chunks = -(-args.n // args.chunk_size)
    max_attempts = n_chunks + 3  # dedup losses get top-up chunks, but never loop forever
    for ci in range(max_attempts):
        want = min(args.chunk_size, args.n - len(records))
        if want <= 0:
            break
        system = system_base.replace("{n}", str(want))
        user = f"Today's date: {today}\n\n### Live SERP material\n{grounding}"
        if records:
            recent = "\n".join(f"- {r['query']}" for r in records[-40:])
            user += ("\n\n### Already generated this batch (write about DIFFERENT stories/angles;"
                     f" near-duplicates are dropped)\n{recent}")
        out_json = call_llm_json(system, user, model="claude-sonnet-4-6", max_tokens=4096)
        got = build_records(out_json.get("queries", []), today, pool,
                            batch_id=batch_id, start_seq=len(records) + 1)
        records.extend(got)
        print(f"  chunk {ci + 1}/{max_attempts}: +{len(got)} (total {len(records)}/{args.n})", flush=True)

    write_jsonl_atomic(out, records)
    print(f"Produced {len(records)} realtime queries (requested {args.n}) → {out}", flush=True)
    for r in records:
        print(f"  [{r['form']:16s}|{r['vertical']:10s}|{r['language']:6s}] {r['query'][:90]}")
    print("\nNext steps — SAME DAY, order mandatory:")
    print(f"  1. python -m src.rubric_gen --queries {out}")
    print(f"  2. python -m src.run_eval --queries data/main_queries.jsonl {out} ...")
    print("This batch expires: run_eval warns on realtime queries whose synth_at is not today.")


if __name__ == "__main__":
    main()
