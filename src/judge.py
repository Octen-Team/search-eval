"""Pairwise LLM judge.

Core mechanics:
- Each pair is judged twice (A/B positions swapped); the verdict is trusted only when both rounds agree;
  contradictions are recorded as tie and flagged position_conflict.
- Per-dimension scores are aggregated into a weighted score using vertical weights, for slice analysis;
  overall uses the judge's holistic call.
- The judge model is configured via the JUDGE_MODEL env var, default claude-sonnet-4-6.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .common import today_utc
from .llm import call_llm_json

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "pairwise_judge.md"

# Per-vertical dimension weights: authority raised for medical_legal / finance
VERTICAL_WEIGHTS: dict[str, dict[str, float]] = {
    "default":       {"relevance": 0.40, "authority": 0.20, "freshness": 0.15, "diversity": 0.10, "snippet_quality": 0.15},
    "medical_legal": {"relevance": 0.30, "authority": 0.40, "freshness": 0.10, "diversity": 0.05, "snippet_quality": 0.15},
    "finance":       {"relevance": 0.30, "authority": 0.35, "freshness": 0.20, "diversity": 0.05, "snippet_quality": 0.10},
    "news":          {"relevance": 0.30, "authority": 0.20, "freshness": 0.35, "diversity": 0.05, "snippet_quality": 0.10},
}

DIMENSIONS = ["relevance", "authority", "freshness", "diversity", "snippet_quality"]


@lru_cache(maxsize=1)
def _load_prompts() -> tuple[str, str]:
    # cached: judge_pair used to re-read and re-split this static file on every verdict
    text = PROMPT_PATH.read_text(encoding="utf-8")
    system = text.split("## SYSTEM PROMPT (template)")[1].split("## USER PROMPT (template)")[0].strip()
    user = (text.split("## USER PROMPT (template)")[1].split("## Calibration process")[0]
            .strip().removesuffix("---").strip())  # strip('-') was a char-set footgun
    return system, user


def _round_ok(r: dict) -> bool:
    """Judge output shape check — validate BEFORE paying for the second round / crashing later."""
    if r.get("overall") not in ("A", "B", "tie"):
        return False
    scores = r.get("scores", {})
    return all(d in scores.get(side, {}) for side in ("A", "B") for d in DIMENSIONS)


def _call_judge(system: str, user: str, model: str | None = None) -> dict:
    # retries/backoff/parse-recovery live in llm.call_llm_json; here we only add a shape
    # re-ask (a malformed-but-valid-JSON verdict is invisible to the JSON layer).
    # JUDGE_TEMPERATURE (e.g. 0) reduces verdict flip-noise on near-tie pairs; leave unset for
    # the historical default. Changing it makes runs incomparable with earlier ones.
    temp_env = os.environ.get("JUDGE_TEMPERATURE", "")
    temperature = float(temp_env) if temp_env != "" else None
    last = None
    for _ in range(2):
        # max_tokens covers reasoning models (gemini/gpt burn "thinking" tokens from the same
        # budget; 1024 truncates their JSON mid-string) — billed by usage, harmless for claude
        r = call_llm_json(system, user, model=model or os.environ.get("JUDGE_MODEL", "claude-sonnet-4-6"),
                          max_tokens=8192, temperature=temperature)
        if _round_ok(r):
            return r
        last = r
    raise ValueError(f"judge output malformed after re-ask: overall={last.get('overall')!r}, "
                     f"score keys={ {s: sorted(last.get('scores', {}).get(s, {})) for s in ('A', 'B')} }")


@dataclass
class PairwiseVerdict:
    qid: str
    system_x: str          # real system name (not the A/B anonymization)
    system_y: str
    winner: str            # system_x name / system_y name / "tie"
    confidence: str
    position_conflict: bool
    dim_scores_x: dict
    dim_scores_y: dict
    weighted_x: float
    weighted_y: float
    evidence: str
    checklist_x: list = field(default_factory=list)  # checklist item ids covered by this system (intersection of both rounds — conservative)
    checklist_y: list = field(default_factory=list)
    rubric_version: str | None = None
    # raw slot verdicts per round ("A"/"B"/"tie") — lets reports measure the slot-A win rate,
    # the DIRECT position-bias statistic (should hover near 50% for an unbiased judge)
    round1_overall: str | None = None
    round2_overall: str | None = None


def judge_pair(qid: str, query_meta: dict, resp_x, resp_y, k: int = 10,
               rubric_rec: dict | None = None, judge_model: str | None = None) -> PairwiseVerdict:
    """resp_x / resp_y: backends.SearchResponse. Runs twice with positions swapped.

    rubric_rec: this qid's record as returned by rubric_gen.load_rubrics(); None falls back to the generic rubric.
    judge_model: per-call judge override (cross-family calibration); None uses the JUDGE_MODEL env var.
    """
    from .rubric_gen import render_for_judge  # deferred import to avoid a circular dependency

    system_t, user_t = _load_prompts()
    system_t = system_t.replace("{k}", str(k)).replace("{current_date}", today_utc())
    rubric_text = render_for_judge(rubric_rec) if rubric_rec else "(not provided; evaluate against the generic dimension criteria)"

    def one_round(a, b) -> dict:
        user = (
            user_t.replace("{query}", query_meta["query"])
            .replace("{intent}", query_meta.get("intent", "unknown"))
            .replace("{freshness}", query_meta.get("freshness", "unknown"))
            .replace("{vertical}", query_meta.get("vertical", "general"))
            .replace("{query_rubric}", rubric_text)
            .replace("{results_a}", a.to_judge_text(k))
            .replace("{results_b}", b.to_judge_text(k))
        )
        return _call_judge(system_t, user, model=judge_model)

    r1 = one_round(resp_x, resp_y)          # round1: A=x, B=y
    r2 = one_round(resp_y, resp_x)          # round2: A=y, B=x (positions swapped)

    # Map both rounds' A/B back to the real systems
    w1 = {"A": "x", "B": "y", "tie": "tie"}[r1["overall"]]
    w2 = {"A": "y", "B": "x", "tie": "tie"}[r2["overall"]]

    conflict = (w1 != w2) and "tie" not in (w1, w2)
    if conflict:
        winner, confidence = "tie", "low"
        # the verdict is a forced tie — storing one round's confident one-sided argument misleads
        # every downstream reader (losses, triage tickets, report case cards)
        evidence = f"[position-conflict] round1: {r1.get('evidence', '')} | round2: {r2.get('evidence', '')}"
    elif w1 == w2:
        winner, confidence = w1, r1.get("confidence", "medium")
        evidence = r1.get("evidence", "")
    else:  # one round tie, the other has a preference → adopt the preferring round but lower confidence
        winner, confidence = (w1 if w1 != "tie" else w2), "low"
        evidence = (r1 if w1 != "tie" else r2).get("evidence", "")  # evidence from the DECIDING round

    weights = VERTICAL_WEIGHTS.get(query_meta.get("vertical", ""), VERTICAL_WEIGHTS["default"])

    def avg_scores(key_r1: str, key_r2: str) -> dict:
        return {d: (r1["scores"][key_r1][d] + r2["scores"][key_r2][d]) / 2 for d in DIMENSIONS}

    sx, sy = avg_scores("A", "B"), avg_scores("B", "A")
    wx = sum(sx[d] * weights[d] for d in DIMENSIONS)
    wy = sum(sy[d] * weights[d] for d in DIMENSIONS)

    # checklist coverage: round1 has A=x/B=y, round2 has A=y/B=x; take the intersection of both rounds (conservative)
    def coverage(round1_key: str, round2_key: str) -> list:
        c1 = set(r1.get("checklist_coverage", {}).get(round1_key, []))
        c2 = set(r2.get("checklist_coverage", {}).get(round2_key, []))
        return sorted(c1 & c2) if (c1 or c2) else []

    return PairwiseVerdict(
        qid=qid,
        system_x=resp_x.backend,
        system_y=resp_y.backend,
        winner={"x": resp_x.backend, "y": resp_y.backend, "tie": "tie"}[winner],
        confidence=confidence,
        position_conflict=conflict,
        dim_scores_x=sx,
        dim_scores_y=sy,
        weighted_x=round(wx, 3),
        weighted_y=round(wy, 3),
        evidence=evidence,
        checklist_x=coverage("A", "B"),
        checklist_y=coverage("B", "A"),
        rubric_version=(rubric_rec or {}).get("rubric_version"),
        round1_overall=r1["overall"],
        round2_overall=r2["overall"],
    )
