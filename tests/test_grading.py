"""Unit tests for the open-benchmark grading (SimpleQA / FreshQA) — parsers + metrics.

Deterministic; no real API calls (the LLM-calling graders themselves are covered by the
smoke run in the docs). Run: python -m pytest tests/test_grading.py -q
"""
from __future__ import annotations

from src.benchmark_prompts import SIMPLEQA_JUDGE, build_freshqa_prompt
from src.grading import (_parse_fresheval_grade, _parse_simpleqa_grade,
                         freshqa_metrics, simpleqa_metrics)


def test_simpleqa_parse():
    assert _parse_simpleqa_grade("A") == "correct"
    assert _parse_simpleqa_grade("B") == "incorrect"
    assert _parse_simpleqa_grade("C") == "not_attempted"
    assert _parse_simpleqa_grade("A: CORRECT") == "correct"
    assert _parse_simpleqa_grade("Grade: INCORRECT") == "incorrect"   # label-text fallback
    assert _parse_simpleqa_grade("") == "not_attempted"          # official fallback
    assert _parse_simpleqa_grade("garbled") == "not_attempted"


def test_fresheval_parse():
    assert _parse_fresheval_grade("evaluation: correct") == "correct"
    assert _parse_fresheval_grade("...\nevaluation: incorrect") == "incorrect"
    assert _parse_fresheval_grade("the response is not credited") == "incorrect"
    assert _parse_fresheval_grade("the response is credited") == "correct"
    assert _parse_fresheval_grade("") == "incorrect"             # official binary fallback


def test_simpleqa_metrics():
    m = simpleqa_metrics(["correct", "correct", "incorrect", "not_attempted"])
    assert m["correct"] == 2 and m["incorrect"] == 1 and m["not_attempted"] == 1
    assert m["correct_rate"] == 0.5
    assert m["correct_given_attempted"] == round(2 / 3, 4)       # 2 correct of 3 attempted
    # F1 = harmonic mean of correct_rate(0.5) and CGA(0.667)
    assert 0.55 < m["f1"] < 0.58
    assert simpleqa_metrics([])["f1"] == 0.0                     # empty is safe


def test_freshqa_metrics_breakdowns():
    recs = [
        {"grade": "correct", "meta": {"fact_type": "fast-changing", "false_premise": False}},
        {"grade": "incorrect", "meta": {"fact_type": "fast-changing", "false_premise": False}},
        {"grade": "correct", "meta": {"fact_type": "slow-changing", "false_premise": True}},
    ]
    m = freshqa_metrics(recs)
    assert m["n"] == 3 and m["correct"] == 2
    assert m["accuracy"] == round(2 / 3, 4)
    ft = m["breakdowns"]["fact_type"]
    assert ft["fast-changing"]["accuracy"] == 0.5 and ft["fast-changing"]["n"] == 2
    assert ft["slow-changing"]["accuracy"] == 1.0
    assert m["breakdowns"]["false_premise"]["True"]["n"] == 1


def test_benchmark_report(tmp_path):
    import json
    from scripts.benchmark_report import build_report
    recs = [
        {"qid": "simpleqa-v-1", "backend": "octen", "grade_mode": "simpleqa", "grade": "correct",
         "n_searches": 1, "rewrote_query": True},
        {"qid": "simpleqa-v-2", "backend": "octen", "grade_mode": "simpleqa", "grade": "incorrect",
         "n_searches": 2, "rewrote_query": True},
        {"qid": "freshqa-1", "backend": "octen", "grade_mode": "freshqa", "grade": "correct",
         "n_searches": 1, "rewrote_query": False,
         "bench_meta": {"fact_type": "fast-changing", "false_premise": False}},
    ]
    (tmp_path / "agent_eval.jsonl").write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    md = build_report(tmp_path)
    assert "## SimpleQA" in md and "F1" in md
    assert "## FreshQA" in md and "fact_type" in md


def test_prompts_wellformed():
    # SimpleQA judge keeps the A/B/C protocol and the three format slots
    p = SIMPLEQA_JUDGE.format(question="q", ground_truth="g", model_answer="a")
    assert "A: CORRECT" in p and "q" in p and "g" in p and "a" in p
    # FreshEval strict vs relaxed differ, both carry the 15 demos + the new question
    strict = build_freshqa_prompt("q", "g1 | g2", "a", "July 14, 2026", strict=True)
    relaxed = build_freshqa_prompt("q", "g1 | g2", "a", "July 14, 2026", strict=False)
    assert strict.count("evaluation:") == 15 and strict != relaxed
    assert "g1 | g2" in strict
