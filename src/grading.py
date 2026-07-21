"""Official grading for open-source QA benchmarks (SimpleQA, FreshQA).

Routed by a query's meta.benchmark (see agent_eval). Self-built query sets keep the existing
gold/rubric grading in agent_eval; only SimpleQA/FreshQA use the canonical protocols here:

- SimpleQA: openai/simple-evals A/B/C grader → correct / incorrect / not_attempted;
  official metrics correct_rate, correct_given_attempted, F1.
- FreshQA:  freshllms/freshqa FreshEval (strict/relaxed, 15 demos, multi-answer ' | ',
  false-premise) → binary correct / incorrect; accuracy + per-category breakdowns.

Prompts are ported verbatim in benchmark_prompts.py — do not paraphrase.
"""
from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone

from .benchmark_prompts import SIMPLEQA_JUDGE, build_freshqa_prompt
from .llm import call_llm_text

_GRADER_SYS = "You are an exact grader. Follow the instructions precisely and output only what is requested."


def _grader_model() -> str:
    return os.environ.get("GRADER_MODEL", "claude-sonnet-4-6")


# ─────────────────────────── SimpleQA ───────────────────────────

_SQA_MAP = {"A": "correct", "B": "incorrect", "C": "not_attempted"}


def _parse_simpleqa_grade(raw: str) -> str:
    """Parse the A/B/C output of the official SimpleQA grader (ported)."""
    text = (raw or "").strip()
    if text.upper() in _SQA_MAP:
        return _SQA_MAP[text.upper()]
    m = re.match(r"^([ABC])\b", text.upper())
    if m:
        return _SQA_MAP[m.group(1)]
    low = text.lower()
    if "not_attempted" in low or "not attempted" in low:
        return "not_attempted"
    if "incorrect" in low:
        return "incorrect"
    if "correct" in low:
        return "correct"
    return "not_attempted"  # official fallback


def grade_simpleqa(question: str, gold: str, answer: str) -> str:
    """Grade one answer against the SimpleQA gold target → correct/incorrect/not_attempted."""
    prompt = SIMPLEQA_JUDGE.format(question=question, ground_truth=gold,
                                   model_answer=answer or "(no answer)")
    # Headroom for reasoning judge models (they emit hidden thinking before the A/B/C letter);
    # the parser ignores everything except the grade, so extra tokens are harmless.
    raw = call_llm_text(_GRADER_SYS, prompt, _grader_model(), max_tokens=1024)
    return _parse_simpleqa_grade(raw)


def simpleqa_metrics(grades: list[str]) -> dict:
    """Official SimpleQA metrics (openai/simple-evals)."""
    c = Counter(grades)
    total = len(grades) or 1
    correct, incorrect, na = c.get("correct", 0), c.get("incorrect", 0), c.get("not_attempted", 0)
    correct_rate = correct / total
    attempted = correct + incorrect
    cga = correct / attempted if attempted else 0.0
    f1 = (2 * correct_rate * cga / (correct_rate + cga)) if (correct_rate + cga) else 0.0
    return {
        "total": len(grades), "correct": correct, "incorrect": incorrect, "not_attempted": na,
        "correct_rate": round(correct_rate, 4),
        "incorrect_rate": round(incorrect / total, 4),
        "not_attempted_rate": round(na / total, 4),
        "correct_given_attempted": round(cga, 4),
        "f1": round(f1, 4),
    }


# ─────────────────────────── FreshQA ───────────────────────────

def _parse_fresheval_grade(raw: str) -> str:
    """Parse FreshEval output ('evaluation: correct'|'incorrect') — binary (ported)."""
    text = (raw or "").strip().lower()
    for line in text.split("\n"):
        if "evaluation:" in line:
            if "incorrect" in line:
                return "incorrect"
            if "correct" in line:
                return "correct"
    if "the response is not credited" in text:
        return "incorrect"
    if "the response is credited" in text:
        return "correct"
    tail = text.split(".")[-1]
    if "incorrect" in tail:
        return "incorrect"
    if "correct" in tail:
        return "correct"
    return "incorrect"  # official fallback


def grade_freshqa(question: str, ground_truth: str, answer: str, strict: bool = True) -> str:
    """Grade one answer with FreshEval → correct/incorrect. ground_truth = ' | '-joined answers."""
    current_date = datetime.now(timezone.utc).strftime("%B %d, %Y")
    prompt = build_freshqa_prompt(question=question, ground_truth=ground_truth,
                                  model_answer=answer or "(no answer)",
                                  current_date=current_date, strict=strict)
    # FreshEval emits a free-text comment then "evaluation: correct|incorrect"; reasoning judges
    # add hidden thinking on top — give enough room so the verdict line is never truncated.
    raw = call_llm_text(_GRADER_SYS, prompt, _grader_model(), max_tokens=1024)
    return _parse_fresheval_grade(raw)


def _wilson_ci(k: int, n: int) -> str:
    """95% Wilson interval as a display string for the FreshQA tables. Reuses report.wilson_ci
    (single source of truth for the interval math; deferred import avoids load-order coupling)."""
    if not n:
        return "—"
    from .report import wilson_ci
    lo, hi = wilson_ci(k, n)
    return f"{k / n:.0%} [{lo:.0%}–{hi:.0%}]"


def freshqa_metrics(records: list[dict]) -> dict:
    """records: [{grade, meta}] where meta has fact_type/num_hops/false_premise/effective_year.
    Returns overall accuracy + CI + per-category breakdowns (aligned with the FreshQA tables)."""
    def acc(rs):
        n = len(rs); k = sum(1 for r in rs if r["grade"] == "correct")
        return {"n": n, "correct": k, "accuracy": round(k / n, 4) if n else 0.0,
                "accuracy_ci": _wilson_ci(k, n)}
    out = acc(records)
    breakdowns: dict[str, dict] = {}
    for field in ("fact_type", "num_hops", "false_premise", "effective_year"):
        groups: dict[str, list] = defaultdict(list)
        for r in records:
            groups[str(r.get("meta", {}).get(field, "?"))].append(r)
        breakdowns[field] = {k: acc(v) for k, v in sorted(groups.items())}
    out["breakdowns"] = breakdowns
    return out
