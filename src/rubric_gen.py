"""Automatic per-query rubric generation.

Usage:
  python -m src.rubric_gen --queries data/seed_queries.jsonl --out data/rubrics.jsonl
  python -m src.rubric_gen --queries data/seed_queries.jsonl --out data/rubrics.jsonl --force  # full regen

Mechanics:
- Each rubric record carries query_sha (hash of the query text) + rubric_version.
  Incremental is the DEFAULT: entries that already exist with a matching hash and current
  version are skipped, so crash-relaunches and set growth never re-burn tokens or reset
  review results. --force regenerates everything (deliberately).
- The generator never sees any evaluated system's results (anti-bias; rationale in
  prompts/rubric_generator.md). Google+Bing SERP results serve as out-of-band fact grounding.
- Entries whose intent_interpretation carries the [UNCERTAIN] marker, plus those with
  difficulty=tail or form=agent_generated, get needs_review=true → human review queue
  (view with --list-review).
- judge.py automatically loads the default path data/rubrics.jsonl (the --out default) and
  injects it into the judge prompt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

from .backends import _load_dotenv
from .common import load_jsonl, today_utc, write_jsonl_atomic
from .llm import call_llm_json

_load_dotenv()

RUBRIC_VERSION = "v4-grounded"  # v4: anti-fabrication hardening (topic-coverage-not-values, no false-premise intent, disqualifier restraint, terse freshness). v3 English prompts; v2 SERP grounding. Versions are not comparable.
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "rubric_generator.md"


def qsha(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]


@lru_cache(maxsize=1)
def _load_prompts() -> tuple[str, str]:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    system = text.split("## SYSTEM PROMPT (template)")[1].split("## USER PROMPT (template)")[0].strip()
    user = text.split("## USER PROMPT (template)")[1].strip().removesuffix("---").strip()
    return system, user


def _call_llm(system: str, user: str) -> dict:
    # retries/backoff/parse-recovery live in llm.call_llm_json
    # max_tokens=2048: grounded rubrics get truncated into invalid JSON at 1024 (observed 2026-07)
    return call_llm_json(system, user, model=os.environ.get("RUBRIC_MODEL", "claude-sonnet-4-6"),
                         max_tokens=2048)


def _validate_rubric(r: dict) -> None:
    """Shape-check the LLM's rubric BEFORE persisting — a malformed record used to crash
    render_for_judge in every later judge run (and --list-review, and the progress print)."""
    if not isinstance(r.get("intent_interpretation"), str) or not r["intent_interpretation"].strip():
        raise ValueError("rubric missing intent_interpretation")
    checklist = r.get("checklist")
    if not isinstance(checklist, list) or not checklist:
        raise ValueError("rubric missing checklist")
    for c in checklist:
        if not (isinstance(c, dict) and c.get("id") and c.get("desc")
                and isinstance(c.get("weight"), (int, float))):
            raise ValueError(f"malformed checklist item: {c!r}")
    r.setdefault("authority_expectation", "")
    r.setdefault("freshness_window", "any")
    if not isinstance(r.get("disqualifiers"), list):
        r["disqualifiers"] = []


def generate_one(q: dict, system_t: str, user_t: str, grounding: bool = True,
                 feedback: str = "") -> dict:
    meta_note = ""
    if q.get("meta", {}).get("note"):
        meta_note = f"Annotator note: {q['meta']['note']}"

    serp_text, grounded = "(grounding disabled; generating from model knowledge)", False
    if grounding:
        try:
            from .serp import serp_both, render_grounding
            serp_text = render_grounding(serp_both(q["query"], k=6))
            grounded = True
        except Exception as e:  # noqa: BLE001
            serp_text = f"(grounding fetch failed; generating from model knowledge: {e})"

    user = (
        user_t.replace("{query}", q["query"])
        .replace("{intent}", q.get("intent", "?"))
        .replace("{difficulty}", q.get("difficulty", "?"))
        .replace("{freshness}", q.get("freshness", "?"))
        .replace("{vertical}", q.get("vertical", "?"))
        .replace("{language}", q.get("language", "?"))
        .replace("{form}", q.get("form", "?"))
        .replace("{meta_note}", meta_note)
        .replace("{serp_grounding}", serp_text)
    )
    if feedback:
        user += (
            "\n\n### A previous version of this rubric failed review — fix these issues\n"
            f"{feedback}\n"
            "Regenerate the full rubric from scratch avoiding these failure modes. Do not simply "
            "delete flagged items — if the intent genuinely requires that ground, restate it as an "
            "objective, verifiable requirement without unsupported specifics."
        )
    rubric = _call_llm(system_t, user)
    _validate_rubric(rubric)
    uncertain = rubric.get("intent_interpretation", "").startswith("[UNCERTAIN]")
    needs_review = uncertain or q.get("difficulty") == "tail" or q.get("form") == "agent_generated"
    return {
        "qid": q["qid"],
        "query_sha": qsha(q["query"]),
        "rubric_version": RUBRIC_VERSION,
        "generated_at": today_utc(),
        "generated_by": os.environ.get("RUBRIC_MODEL", "claude-sonnet-4-6"),
        "grounded": grounded,
        "grounding_source": "google+bing_serp" if grounded else None,
        "needs_review": needs_review,
        "reviewed": False,
        "rubric": rubric,
    }


def load_rubrics(path: str | Path = "data/rubrics.jsonl") -> dict[str, dict]:
    """qid → rubric record (last-wins across appended duplicates). Called by judge.py."""
    return {rec["qid"]: rec for rec in load_jsonl(path)}


def render_for_judge(rec: dict) -> str:
    """Render as the text block injected into the judge prompt."""
    r = rec["rubric"]
    lines = [f"Intent interpretation: {r['intent_interpretation']}"]
    lines.append("Checklist (an ideal result set should cover these; weight 3=required / 1=bonus):")
    for c in r.get("checklist", []):
        lines.append(f"  [{c['id']}] (w{c['weight']}) {c['desc']}")
    lines.append(f"Authority expectation: {r.get('authority_expectation', '')}")
    lines.append(f"Freshness window: {r.get('freshness_window', 'any')}")
    if r.get("disqualifiers"):
        lines.append("Disqualifiers (heavily penalize hits): " + "; ".join(r["disqualifiers"]))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", default="data/rubrics.jsonl")
    ap.add_argument("--only-missing", action="store_true",
                    help="deprecated: incremental is now the default; use --force to regenerate everything")
    ap.add_argument("--force", action="store_true",
                    help="regenerate even current entries (overwrites review results — use deliberately)")
    ap.add_argument("--list-review", action="store_true", help="only list the rubrics pending human review")
    ap.add_argument("--no-grounding", action="store_true", help="skip SERP grounding (degraded mode when no SERPAPI key)")
    ap.add_argument("--concurrency", type=int, default=4, help="parallel rubric generations")
    ap.add_argument("--regen-failed", action="store_true",
                    help="regenerate only rubrics whose auto_review verdict is 'fail', feeding the panel's issues back into the prompt")
    args = ap.parse_args()

    out = Path(args.out)
    existing = load_rubrics(out)

    if args.list_review:
        pending = [r for r in existing.values() if r["needs_review"] and not r["reviewed"]]
        print(f"Rubrics pending review: {len(pending)}")
        for r in pending:
            print(f"  {r['qid']}: {r['rubric']['intent_interpretation'][:80]}")
        return

    queries = load_jsonl(args.queries, tolerate_torn_tail=False)
    system_t, user_t = _load_prompts()
    system_t = system_t.replace("{current_date}", today_utc())

    def needs_gen(q: dict) -> bool:
        # skip-if-current is the DEFAULT (crash-relaunch must not re-burn tokens or reset
        # review results); --force regenerates deliberately
        if args.force:
            return True
        old = existing.get(q["qid"])
        return not (old and old.get("query_sha") == qsha(q["query"])
                    and old.get("rubric_version") == RUBRIC_VERSION)

    records: dict[str, dict] = dict(existing)
    feedback_map: dict[str, str] = {}
    if args.regen_failed:
        for qid, r in existing.items():
            issues = r.get("auto_review", {}).get("issues", [])
            if r.get("auto_review", {}).get("verdict") == "fail":
                feedback_map[qid] = "\n".join(
                    f"- [{i['severity']}] {i['field']}: {i['note']}"
                    for i in issues if i["severity"] in ("major", "minor"))[:3000]
        todo = [q for q in queries if q["qid"] in feedback_map]
    else:
        todo = [q for q in queries if needs_gen(q)]
    print(f"{len(todo)} to generate, {len(queries) - len(todo)} skipped (already current); "
          f"concurrency={args.concurrency}", flush=True)

    # crash-safe persistence: append+flush per record (load_rubrics is last-wins per qid),
    # then compact at the end. Interrupting the run loses nothing; just relaunch.
    out.parent.mkdir(parents=True, exist_ok=True)
    n_gen, n_fail = 0, 0
    lock = threading.Lock()
    with out.open("a", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        # caveat: force-regenerating an unchanged query overwrites any human review result
        futures = {pool.submit(generate_one, q, system_t, user_t, not args.no_grounding,
                               feedback_map.get(q["qid"], "")): q
                   for q in todo}
        for fut in as_completed(futures):
            q = futures[fut]
            with lock:
                try:
                    rec = fut.result()
                except Exception as e:  # noqa: BLE001
                    n_fail += 1
                    print(f"  {q['qid']} FAILED: {e}", flush=True)
                    continue
                records[q["qid"]] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n_gen += 1
                flag = " [REVIEW]" if rec["needs_review"] else ""
                print(f"  [{n_gen}/{len(todo)}] {q['qid']}{flag}: "
                      f"{rec['rubric'].get('intent_interpretation', '')[:60]}", flush=True)

    # compact: dedupe by qid, sorted — atomic so a crash mid-compaction can't destroy the file
    write_jsonl_atomic(out, [records[qid] for qid in sorted(records)])
    n_review = sum(1 for r in records.values() if r["needs_review"] and not r["reviewed"])
    print(f"\nGenerated {n_gen} / failed {n_fail} → {out} ({len(records)} total)", flush=True)
    print(f"{n_review} pending human review (view with --list-review; set reviewed to true after reviewing)", flush=True)


if __name__ == "__main__":
    main()
