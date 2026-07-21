"""Pipeline unit tests: judge parsing / report aggregation / triage classification / anchor scoring / LLM JSON robustness.

Everything mocked — no real API calls. Run: python -m pytest tests/ -q
"""
from __future__ import annotations

import json

import pytest

from src import judge as judge_mod
from src import llm as llm_mod
from src import triage as triage_mod
from src.backends import SearchResult, SearchResponse
from src.judge import judge_pair, VERTICAL_WEIGHTS, DIMENSIONS
from src.report import wr
from src.rubric_gen import render_for_judge
from src.run_eval import url_match, score_anchor
from src.triage import triage_one


def make_resp(backend: str, n: int = 3) -> SearchResponse:
    results = [
        SearchResult(rank=i + 1, title=f"{backend} title {i+1}",
                     url=f"https://example-{backend}.com/page{i+1}",
                     snippet=f"snippet {i+1}")
        for i in range(n)
    ]
    return SearchResponse(backend=backend, query="test query", results=results, latency_ms=100.0)


def scores(a: float, b: float) -> dict:
    return {"A": {d: a for d in DIMENSIONS}, "B": {d: b for d in DIMENSIONS}}


QMETA = {"query": "test query", "intent": "informational", "freshness": "evergreen", "vertical": "general"}


def patch_judge(monkeypatch, rounds: list[dict]):
    it = iter(rounds)
    monkeypatch.setattr(judge_mod, "_call_judge", lambda s, u, model=None: next(it))


class TestJudgePair:
    def test_consistent_winner(self, monkeypatch):
        # round1: A(=x) wins; round2 after position swap B(=x) wins → consistent, x wins
        r1 = {"overall": "A", "confidence": "high", "evidence": "e1", "scores": scores(4, 2),
              "checklist_coverage": {"A": ["c1", "c2"], "B": ["c1"]}}
        r2 = {"overall": "B", "confidence": "high", "evidence": "e2", "scores": scores(2, 4),
              "checklist_coverage": {"A": ["c1"], "B": ["c1", "c2"]}}
        patch_judge(monkeypatch, [r1, r2])
        v = judge_pair("q1", QMETA, make_resp("octen"), make_resp("exa"))
        assert v.winner == "octen"
        assert v.position_conflict is False
        assert v.confidence == "high"
        # x's dimension score = (r1.A + r2.B) / 2 = (4+4)/2 = 4
        assert v.dim_scores_x["relevance"] == 4
        assert v.dim_scores_y["relevance"] == 2
        # checklist intersection across rounds: x = r1.A ∩ r2.B = {c1,c2}; y = r1.B ∩ r2.A = {c1}
        assert v.checklist_x == ["c1", "c2"]
        assert v.checklist_y == ["c1"]

    def test_position_conflict_records_tie(self, monkeypatch):
        # both rounds call A the winner → contradictory after mapping back to real systems (x and y win once each) → tie + conflict
        r = {"overall": "A", "confidence": "high", "evidence": "", "scores": scores(3, 3)}
        patch_judge(monkeypatch, [dict(r), dict(r)])
        v = judge_pair("q1", QMETA, make_resp("octen"), make_resp("exa"))
        assert v.winner == "tie"
        assert v.position_conflict is True
        assert v.confidence == "low"

    def test_tie_plus_preference_lowers_confidence(self, monkeypatch):
        r1 = {"overall": "tie", "confidence": "high", "evidence": "r1 said tie", "scores": scores(3, 3)}
        r2 = {"overall": "B", "confidence": "high", "evidence": "r2 preferred x", "scores": scores(3, 3)}  # round2 B=x
        patch_judge(monkeypatch, [r1, r2])
        v = judge_pair("q1", QMETA, make_resp("octen"), make_resp("exa"))
        assert v.winner == "octen"
        assert v.confidence == "low"
        assert v.position_conflict is False
        assert v.evidence == "r2 preferred x"  # evidence from the DECIDING round, not blindly r1

    def test_conflict_evidence_shows_both_rounds(self, monkeypatch):
        r = {"overall": "A", "confidence": "high", "evidence": "one-sided claim", "scores": scores(3, 3)}
        patch_judge(monkeypatch, [dict(r), dict(r)])
        v = judge_pair("q1", QMETA, make_resp("octen"), make_resp("exa"))
        assert v.winner == "tie" and "[position-conflict]" in v.evidence

    def test_malformed_judge_output_raises_after_reask(self, monkeypatch):
        from src import judge as judge_mod2
        bad = {"overall": "A wins!", "scores": scores(3, 3)}
        monkeypatch.setattr(judge_mod2, "call_llm_json", lambda *a, **k: dict(bad))
        with pytest.raises(ValueError, match="judge output malformed"):
            judge_mod2._call_judge("s", "u")

    def test_weighted_scores_use_vertical_weights(self, monkeypatch):
        r1 = {"overall": "A", "confidence": "high", "evidence": "", "scores": scores(5, 1)}
        r2 = {"overall": "B", "confidence": "high", "evidence": "", "scores": scores(1, 5)}
        patch_judge(monkeypatch, [r1, r2])
        meta = dict(QMETA, vertical="medical_legal")
        v = judge_pair("q1", meta, make_resp("octen"), make_resp("exa"))
        w = VERTICAL_WEIGHTS["medical_legal"]
        assert v.weighted_x == round(sum(5 * w[d] for d in DIMENSIONS), 3)


class TestAnchor:
    def test_url_match_modes(self):
        assert url_match("https://github.com/a/b", "https://github.com/a/b/", "exact")
        assert url_match("https://github.com/a/b/tree/main", "https://github.com/a/b", "prefix")
        assert url_match("https://www.example.com/x", "https://example.com/y", "domain")
        assert not url_match("https://other.com/a/b", "https://github.com/a/b", "prefix")

    def test_url_match_prefix_is_path_boundary_aware(self):
        # sibling paths and lookalike hosts used to inflate anchor hits
        assert not url_match("https://github.com/org/repo-archive", "https://github.com/org/repo", "prefix")
        assert not url_match("https://www.who.int.evil.com/", "https://www.who.int/", "prefix")
        assert not url_match("https://www.rust-lang.org/en-USSR", "https://www.rust-lang.org/en-US", "prefix")
        assert url_match("https://github.com/org/repo?tab=readme", "https://github.com/org/repo", "prefix")
        assert url_match("https://numpy.org/doc/stable", "https://numpy.org/doc/", "prefix")

    def test_score_anchor_hit_rank(self):
        q = {"qid": "q1", "gold": {"gold_urls": ["https://example-octen.com/page2"], "url_match": "prefix"}}
        rec = score_anchor(q, make_resp("octen"), k=3)
        assert rec["hit_at_1"] is False and rec["hit_at_k"] is True and rec["rank"] == 2

    def test_score_anchor_no_gold_returns_none(self):
        assert score_anchor({"qid": "q1"}, make_resp("octen"), k=3) is None


class TestReport:
    def test_wr_counts_and_rate(self):
        recs = [{"winner": "octen"}, {"winner": "tie"}, {"winner": "exa"}, {"winner": "octen"}]
        s = wr(recs, "octen")
        assert "W 2 / T 1 / L 1" in s and "67%" in s

    def test_slice_cell_flags_high_conflict_as_low_trust(self):
        from src.report import slice_cell
        # 2 of 4 records conflict = 50% > 25% threshold → must carry the LOW-TRUST marker
        recs = [{"winner": "exa", "position_conflict": True},
                {"winner": "exa", "position_conflict": True},
                {"winner": "octen", "position_conflict": False},
                {"winner": "tie", "position_conflict": False}]
        cell = slice_cell("general", recs, "octen")
        assert "LOW-TRUST" in cell and "50%" in cell

    def test_slice_cell_clean_below_threshold(self):
        from src.report import slice_cell
        recs = [{"winner": "octen", "position_conflict": False} for _ in range(4)]
        assert "LOW-TRUST" not in slice_cell("general", recs, "octen")

    def test_percentile_nearest_rank(self):
        from src.report import percentile
        vals = list(range(1, 21))  # 1..20
        assert percentile(vals, 0.95) == 19  # int(n*0.95) indexing used to return 20 (the max)
        assert percentile(vals, 0.5) == 10
        assert percentile([7], 0.95) == 7


class TestWilsonCI:
    def test_bounds_and_monotonicity(self):
        from src.report import wilson_ci
        lo, hi = wilson_ci(8, 10)
        assert 0.0 <= lo < 0.8 < hi <= 1.0
        lo_big, hi_big = wilson_ci(80, 100)
        assert (hi_big - lo_big) < (hi - lo)  # more data → tighter interval
        assert wilson_ci(0, 0) == (0.0, 1.0)


class TestRunEvalResume:
    def test_expected_pairs_skips_errored_fetches(self):
        from src.run_eval import expected_pairs
        recs = {"octen": {"error": None}, "exa": {"error": None}, "brave": {"error": "boom"}}
        assert expected_pairs(recs, "octen", ["exa", "brave"]) == {"exa"}
        recs_ours_err = {"octen": {"error": "down"}, "exa": {"error": None}}
        assert expected_pairs(recs_ours_err, "octen", ["exa"]) == set()

    def test_load_resume_completeness(self, tmp_path):
        from src.run_eval import load_resume
        # q1 complete; q2 missing its verdict → re-run; q3 fetch-errored → re-run (not silently lost)
        resp = [{"qid": "q1", "backend": "octen", "error": None},
                {"qid": "q1", "backend": "exa", "error": None},
                {"qid": "q2", "backend": "octen", "error": None},
                {"qid": "q2", "backend": "exa", "error": None},
                {"qid": "q3", "backend": "octen", "error": "boom"},
                {"qid": "q3", "backend": "exa", "error": None}]
        pair = [{"qid": "q1", "system_y": "exa", "winner": "octen"},
                {"qid": "q2", "system_y": "exa", "error": "judge blew up"}]
        (tmp_path / "responses.jsonl").write_text("\n".join(json.dumps(r) for r in resp))
        (tmp_path / "pairwise.jsonl").write_text("\n".join(json.dumps(p) for p in pair))
        complete, kept, total_raw = load_resume(tmp_path, ["octen", "exa"], "octen", ["exa"], skip_judge=False)
        assert complete == {"q1"}
        assert total_raw == 8
        assert all(r["qid"] == "q1" for recs in kept.values() for r in recs)  # incomplete records dropped

    def test_load_resume_tolerates_torn_final_line(self, tmp_path):
        from src.run_eval import load_resume
        (tmp_path / "responses.jsonl").write_text(
            json.dumps({"qid": "q1", "backend": "octen", "error": None}) + "\n" + '{"qid": "q2", "backe')
        complete, kept, total_raw = load_resume(tmp_path, ["octen"], "octen", [], skip_judge=True)
        assert complete == {"q1"} and total_raw == 1  # torn line dropped, resume proceeds


class TestAnchorGate:
    def _recs(self, hits: dict[str, bool]) -> dict:
        return {q: {"qid": q, "backend": "octen", "hit_at_1": h, "hit_at_k": h, "rank": 1 if h else None}
                for q, h in hits.items()}

    def test_pass_when_no_drop(self):
        from scripts.anchor_gate import gate
        base = self._recs({"a": True, "b": False})
        run = self._recs({"a": True, "b": True})
        passed, _ = gate(run, base, tolerance=0.0, gate_hit1=False)
        assert passed

    def test_fail_on_regression(self):
        from scripts.anchor_gate import gate
        base = self._recs({"a": True, "b": True})
        run = self._recs({"a": True, "b": False})
        passed, lines = gate(run, base, tolerance=0.0, gate_hit1=False)
        assert not passed and any("b" in l for l in lines)

    def test_tolerance_allows_small_drop(self):
        from scripts.anchor_gate import gate
        base = self._recs({c: True for c in "abcdefghij"})
        run = self._recs({**{c: True for c in "abcdefghi"}, "j": False})
        passed, _ = gate(run, base, tolerance=0.10, gate_hit1=False)
        assert passed

    def test_float_boundary_drop_equal_to_tolerance_passes(self):
        # 36/50 → 35/50 with tolerance 0.02: 0.72-0.70 > 0.02 in floats used to FAIL the gate
        from scripts.anchor_gate import gate
        base = self._recs({f"q{i}": i < 36 for i in range(50)})
        run = self._recs({f"q{i}": i < 35 for i in range(50)})
        passed, _ = gate(run, base, tolerance=0.02, gate_hit1=False)
        assert passed


class TestReportEscaping:
    def test_md_escape_neutralizes_html(self):
        from scripts.gen_report import md_escape
        out = md_escape('rows >100M with Vec<T> & "quotes" | pipe')
        assert "<" not in out and ">" not in out and "&lt;" in out and "\\|" in out


class TestCommonIO:
    def test_write_jsonl_atomic_trailing_newline_and_replace(self, tmp_path):
        from src.common import write_jsonl_atomic, load_jsonl
        p = tmp_path / "x.jsonl"
        write_jsonl_atomic(p, [{"a": 1}, {"b": 2}])
        text = p.read_text()
        assert text.endswith("\n") and len(text.splitlines()) == 2  # cat-merge can't fuse records
        assert load_jsonl(p) == [{"a": 1}, {"b": 2}]
        assert not p.with_name(p.name + ".tmp").exists()

    def test_load_jsonl_torn_tail_tolerated_but_middle_corruption_raises(self, tmp_path):
        from src.common import load_jsonl
        p = tmp_path / "x.jsonl"
        p.write_text('{"a": 1}\n{"torn')
        assert load_jsonl(p) == [{"a": 1}]
        p.write_text('{"corrupt\n{"a": 1}\n')
        with pytest.raises(json.JSONDecodeError):
            load_jsonl(p)


class TestTriage:
    def _loss(self):
        return {"qid": "q1", "winner": "exa", "evidence": "e",
                "dim_scores_x": {d: 2.0 for d in DIMENSIONS},
                "dim_scores_y": {d: 4.0 for d in DIMENSIONS}}

    def _responses(self):
        return {
            ("q1", "octen"): {"error": None, "n_results": 3, "results": []},
            ("q1", "exa"): {"error": None, "n_results": 3,
                            "results": [{"url": f"https://gold.com/{i}"} for i in range(3)]},
        }

    def test_service_error(self):
        rbk = self._responses()
        rbk[("q1", "octen")] = {"error": "boom", "n_results": 0}
        assert triage_one(self._loss(), rbk, "octen")["mode"] == "SERVICE_ERROR"

    def test_service_empty(self):
        rbk = self._responses()
        rbk[("q1", "octen")] = {"error": None, "n_results": 0}
        assert triage_one(self._loss(), rbk, "octen")["mode"] == "SERVICE_EMPTY"

    def test_pending_index_check_without_probe(self):
        rec = triage_one(self._loss(), self._responses(), "octen")
        assert rec["mode"] == "PENDING_INDEX_CHECK"
        assert rec["auto"] is False

    def test_probe_confirms_all_missing(self, monkeypatch):
        monkeypatch.setattr(triage_mod, "probe_in_index", lambda url, be: False)
        rec = triage_one(self._loss(), self._responses(), "octen", probe_backend=object())
        assert rec["mode"] == "INDEX_MISS_PROBED"
        assert len(rec["detail"]["missing_urls"]) == 3

    def test_pending_l2_l4_when_indexed(self, monkeypatch):
        monkeypatch.setattr(triage_mod, "probe_in_index", lambda url, be: True)
        rec = triage_one(self._loss(), self._responses(), "octen", probe_backend=object())
        assert rec["mode"] == "PENDING_L2_L4"
        assert rec["detail"]["dim_gap"]["relevance"] == -2.0

    def test_probe_flags_low_confidence_miss(self, monkeypatch):
        monkeypatch.setattr(triage_mod, "probe_in_index", lambda url, be: False)
        rec = triage_one(self._loss(), self._responses(), "octen", probe_backend=object())
        assert rec["mode"] == "INDEX_MISS_PROBED"
        assert rec["confidence"] == "low" and rec["auto"] is True

    def test_probe_hit_escalates_to_l2_l4(self, monkeypatch):
        monkeypatch.setattr(triage_mod, "probe_in_index", lambda url, be: True)
        rec = triage_one(self._loss(), self._responses(), "octen", probe_backend=object())
        assert rec["mode"] == "PENDING_L2_L4"

    def test_probe_terms_extracts_host_and_path_tokens(self):
        from src.triage import probe_terms
        t = probe_terms("https://www.postgresql.org/docs/release/16.0/")
        assert t.startswith("postgresql.org") and "docs" in t and "release" in t
        assert probe_terms("https://www.reuters.com/") == "reuters.com"

    def test_partial_index_miss_not_misfiled_as_ranking(self, monkeypatch):
        # 2 confirmed-missing + 1 present used to fall through to PENDING_L2_L4
        seq = iter([False, True, False])
        monkeypatch.setattr(triage_mod, "probe_in_index", lambda url, be: next(seq))
        rec = triage_one(self._loss(), self._responses(), "octen", probe_backend=object())
        assert rec["mode"] == "INDEX_MISS_PROBED"
        assert len(rec["detail"]["missing_urls"]) == 2 and len(rec["detail"]["present_urls"]) == 1

    def test_missing_response_record_is_data_missing_not_service_empty(self):
        rec = triage_one(self._loss(), {}, "octen")
        assert rec["mode"] == "DATA_MISSING" and rec["auto"] is False

    def test_triage_records_carry_competitor(self, monkeypatch):
        monkeypatch.setattr(triage_mod, "probe_in_index", lambda url, be: True)
        loss = dict(self._loss(), system_y="exa")
        rec = triage_one(loss, self._responses(), "octen", probe_backend=object())
        assert rec["competitor"] == "exa"  # report case-cards pair triage evidence by competitor


class TestRubricRender:
    def test_render_for_judge_contains_checklist(self):
        rec = {"rubric": {"intent_interpretation": "find the official repository",
                          "checklist": [{"id": "c1", "desc": "includes the official repo", "weight": 3}],
                          "authority_expectation": "github.com",
                          "freshness_window": "any",
                          "disqualifiers": ["fork repositories"]}}
        text = render_for_judge(rec)
        assert "[c1]" in text and "w3" in text and "fork repositories" in text


class TestIntakeDedup:
    def test_exact_and_near_dup(self):
        from src.query_intake import dedup
        kept, dropped = dedup(
            ["litellm github repo", "LiteLLM  GitHub Repo", "litellm github repos", "totally different query"],
            against=[])
        assert kept == ["litellm github repo", "totally different query"]
        assert {d[1] for d in dropped} == {"exact", "near-dup"}

    def test_dedup_against_existing(self):
        from src.query_intake import dedup
        kept, dropped = dedup(["anthropic messages api streaming example python"],
                              against=["Anthropic Messages API streaming example Python"])
        assert kept == [] and dropped[0][1] == "exact"


class TestQuerySynth:
    def test_quota_covers_all_difficulty_tiers(self):
        from src.query_synth import build_quota, DIFFS
        quota = build_quota(20, 6)
        assert {d for _, d, _ in quota} == set(DIFFS)  # head must not be filtered out at small n

    def test_quota_agent_first(self):
        from src.query_synth import build_quota
        assert build_quota(20, 6)[0][0] == "agent_generated"  # highest-weight cells fill first so truncation can't eat their share

    def test_form_weights_override_changes_quota_order(self):
        from src.query_synth import build_quota, parse_form_weights
        fw = parse_form_weights("keyword=0.6,natural_question=0.2,agent_generated=0.2")
        assert build_quota(20, 6, fw)[0][0] == "keyword"
        with pytest.raises(ValueError):
            parse_form_weights("bogus_form=1.0")

    def test_quota_covers_all_forms_at_small_n(self):
        # regression: at n=15 with 0.2-weight forms, the threshold used to wipe out entire forms
        from src.query_synth import build_quota, parse_form_weights, FORMS
        fw = parse_form_weights("keyword=0.2,natural_question=0.2,agent_generated=0.6")
        assert {f for f, _, _ in build_quota(15, 6, fw)} == set(FORMS)

    def test_quota_sums_to_exactly_n(self):
        # the backstop-cell version guaranteed cells EXISTED but truncation starved them;
        # exact apportionment means the counts themselves sum to n — nothing can be starved
        from src.query_synth import build_quota, parse_form_weights, FORMS, DIFFS
        for n in (12, 15, 20, 40, 120):
            quota = build_quota(n, 6)
            assert sum(c for _, _, c in quota) == n
        fw = parse_form_weights("keyword=0.2,natural_question=0.2,agent_generated=0.6")
        quota = build_quota(15, 6, fw)
        assert sum(c for _, _, c in quota) == 15
        assert {f for f, _, _ in quota} == set(FORMS)
        assert {d for _, d, _ in quota} == set(DIFFS)

    def test_existing_block_render(self):
        from src.query_synth import _existing_block
        assert _existing_block([]) == "(none)"
        assert _existing_block(["q1", "q2"]) == "- q1\n- q2"

    def test_stage2_template_has_existing_placeholder(self):
        from src.query_synth import _sections
        assert "{facet_existing}" in _sections()["STAGE 2 USER PROMPT"]


class TestAgentEval:
    class FakeBackend:
        def __init__(self):
            self.calls = []

        def search(self, terms, k=8):
            self.calls.append(terms)
            return make_resp("fake")

    def test_search_then_answer(self, monkeypatch):
        from src import agent_eval
        actions = iter([{"action": "search", "query": "rewritten entity terms"},
                        {"action": "answer", "answer": "42"}])
        monkeypatch.setattr(agent_eval, "_call_agent", lambda s, u: next(actions))
        be = self.FakeBackend()
        ep = agent_eval.run_agent_one("what is the answer to everything?", be, k=3, max_searches=3, system="sys")
        assert ep["answer"] == "42" and ep["forced"] is False
        assert be.calls == ["rewritten entity terms"]
        assert agent_eval.did_rewrite("what is the answer to everything?", ep["searches"]) is True

    def test_budget_exhaustion_forces_answer(self, monkeypatch):
        from src import agent_eval
        # the agent keeps trying to search; after the budget it must be forced to conclude
        monkeypatch.setattr(agent_eval, "_call_agent",
                            lambda s, u: {"action": "search", "query": "again"})
        be = self.FakeBackend()
        ep = agent_eval.run_agent_one("q", be, k=3, max_searches=2, system="sys")
        assert len(ep["searches"]) == 2 and ep["forced"] is True and ep["answer"] == ""

    def test_no_rewrite_detection(self):
        from src.agent_eval import did_rewrite
        assert did_rewrite("Foo Bar", [{"terms": "  foo   bar "}]) is False

    def test_salvage_action_with_unescaped_quotes(self):
        from src.agent_eval import _salvage_action
        text = '{"action": "answer", "answer": "set "proxy_buffering off;" in the location block"}'
        act = _salvage_action(text)
        assert act["action"] == "answer" and 'proxy_buffering off;' in act["answer"]
        assert _salvage_action("no protocol here") is None
        s = _salvage_action('{"action": "search", "query": "nginx sse "buffering" config"}')
        assert s["action"] == "search" and "buffering" in s["query"]

    def test_salvage_cuts_trailing_sibling_fields(self):
        # greedy capture used to fold ', "confidence": "high"' into the answer payload
        from src.agent_eval import _salvage_action
        act = _salvage_action('{"action": "answer", "answer": "42", "confidence": "high"}')
        assert act["answer"] == "42"

    def test_backend_error_episode_becomes_error_record(self, monkeypatch):
        # an outage must be retried on resume, never frozen into accuracy aggregates as a grade
        from src import agent_eval
        monkeypatch.setattr(agent_eval, "run_agent_one",
                            lambda *a, **k: {"answer": "I could not find this information",
                                             "searches": [{"terms": "x", "n_results": 0,
                                                           "latency_ms": 0.0, "error": "429 too many requests"}],
                                             "forced": False})
        rec = agent_eval.run_episode({"qid": "q1", "query": "q", "gold": {"answer": "42"}},
                                     "exa", object(), 8, 3, "sys")
        assert "error" in rec and "grade" not in rec

    def test_grade_enum_validation(self, monkeypatch):
        from src import agent_eval
        monkeypatch.setattr(agent_eval, "_call_llm_retry", lambda s, u, e, m: {"grade": "correct"})
        assert agent_eval.grade_answer("q", "gold", "pred") == "CORRECT"
        monkeypatch.setattr(agent_eval, "_call_llm_retry", lambda s, u, e, m: {"grade": "weird"})
        assert agent_eval.grade_answer("q", "gold", "pred") == "INCORRECT"

    def test_resume_skips_graded_and_retries_errors(self, tmp_path):
        from src.agent_eval import load_done
        p = tmp_path / "agent_eval.jsonl"
        p.write_text(
            json.dumps({"qid": "q1", "backend": "octen", "grade": "CORRECT"}) + "\n"
            + json.dumps({"qid": "q1", "backend": "exa", "error": "boom"}) + "\n"
            + json.dumps({"qid": "q2", "backend": "octen", "grade_mode": "rubric", "rubric_score": 0.0}) + "\n")
        done = load_done(p)
        assert ("q1", "octen") in done      # graded → skipped on resume
        assert ("q1", "exa") not in done    # error → re-run on resume
        assert ("q2", "octen") in done      # rubric-mode graded (even score 0.0) → skipped

    def test_rubric_grading_score_math(self, monkeypatch):
        from src import agent_eval
        rub = {"rubric": {"intent_interpretation": "x",
                          "checklist": [{"id": "c1", "desc": "a", "weight": 3},
                                        {"id": "c2", "desc": "b", "weight": 2},
                                        {"id": "c3", "desc": "c", "weight": 1}],
                          "disqualifiers": ["trap one", "trap two"]}}
        monkeypatch.setattr(agent_eval, "_call_llm_retry",
                            lambda s, u, e, m: {"covered": ["c1", "c2", "bogus"],
                                                "disqualifiers_hit": [1, 9],
                                                "intent_addressed": True})
        g = agent_eval.grade_answer_rubric("q", rub, "answer")
        # coverage (3+2)/6, one valid trap hit halves it; bogus ids and out-of-range hits dropped
        assert g["checklist_covered"] == ["c1", "c2"]
        assert g["disqualifiers_hit"] == [1]
        assert g["rubric_score"] == round((5 / 6) * 0.5, 3)

    def test_rubric_grading_intent_gate_zeroes_score(self, monkeypatch):
        from src import agent_eval
        rub = {"rubric": {"intent_interpretation": "x",
                          "checklist": [{"id": "c1", "desc": "a", "weight": 3}], "disqualifiers": []}}
        monkeypatch.setattr(agent_eval, "_call_llm_retry",
                            lambda s, u, e, m: {"covered": ["c1"], "disqualifiers_hit": [],
                                                "intent_addressed": False})
        assert agent_eval.grade_answer_rubric("q", rub, "answer")["rubric_score"] == 0.0


class TestRealtimeSynth:
    def test_build_records_validates_dedups_and_stamps(self):
        from src.realtime_synth import build_records
        items = [
            {"query": "nasdaq movers right now", "intent": "informational", "difficulty": "head",
             "form": "keyword", "vertical": "finance", "language": "en"},
            {"query": "NASDAQ movers  right   now", "intent": "informational", "difficulty": "head",
             "form": "keyword", "vertical": "finance", "language": "en"},  # near-dup → dropped
            {"query": "bad labels", "intent": "nope", "difficulty": "head",
             "form": "keyword", "vertical": "finance"},                      # invalid enum → dropped
        ]
        recs = build_records(items, "2026-07-08", pool=[])
        assert len(recs) == 1
        r = recs[0]
        assert r["qid"] == "q-realtime-20260708-001"
        assert r["freshness"] == "realtime" and r["meta"]["synth_at"] == "2026-07-08"

    def test_stale_realtime_warning(self):
        from src.run_eval import stale_realtime
        qs = [{"qid": "a", "freshness": "realtime", "meta": {"synth_at": "2026-07-07"}},
              {"qid": "b", "freshness": "realtime", "meta": {"synth_at": "2026-07-08"}},
              {"qid": "c", "freshness": "evergreen", "meta": {}}]
        assert stale_realtime(qs, "2026-07-08") == ["a"]

    def test_load_queries_multi_file_and_dup_guard(self, tmp_path):
        from src.run_eval import load_queries
        f1, f2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
        f1.write_text(json.dumps({"qid": "q1", "query": "x"}))
        f2.write_text(json.dumps({"qid": "q2", "query": "y"}))
        assert len(load_queries([str(f1), str(f2)])) == 2
        f2.write_text(json.dumps({"qid": "q1", "query": "y"}))
        with pytest.raises(SystemExit):
            load_queries([str(f1), str(f2)])


class TestRubricReviewPanel:
    def test_aggregate_majority_pass(self):
        from src.rubric_review import aggregate
        assert aggregate({"a": "pass", "b": "pass", "c": "fail"}) == "pass"
        assert aggregate({"a": "pass", "b": "fail", "c": "fail"}) == "fail"

    def test_aggregate_with_errors_is_conservative(self):
        from src.rubric_review import aggregate
        # one reviewer erroring: 1 pass + 1 fail → no pass majority → fail (stays with humans)
        assert aggregate({"a": "pass", "b": "fail", "c": "error"}) == "fail"
        assert aggregate({"a": "pass", "b": "pass", "c": "error"}) == "pass"
        # fewer than 2 valid votes → error (re-reviewed on resume)
        assert aggregate({"a": "pass", "b": "error", "c": "error"}) == "error"

    def test_should_review_on_content_change(self):
        from src.rubric_review import should_review, rubric_sha
        rec = {"qid": "q1", "rubric": {"intent_interpretation": "v2"}}
        assert should_review(rec, None) is True
        assert should_review(rec, {"verdict": "error", "rubric_sha": rubric_sha(rec)}) is True
        assert should_review(rec, {"verdict": "fail", "rubric_sha": rubric_sha(rec)}) is False
        # regenerated rubric → sha differs → must re-review despite a logged verdict
        assert should_review(rec, {"verdict": "fail", "rubric_sha": "0" * 12}) is True

    def test_regen_feedback_injected_into_prompt(self, monkeypatch):
        from src import rubric_gen
        captured = {}
        def fake_llm(system, user):
            captured["user"] = user
            return {"intent_interpretation": "x",
                    "checklist": [{"id": "c1", "desc": "a", "weight": 3}],
                    "authority_expectation": "", "freshness_window": "any", "disqualifiers": []}
        monkeypatch.setattr(rubric_gen, "_call_llm", fake_llm)
        q = {"qid": "q1", "query": "test", "meta": {}}
        rubric_gen.generate_one(q, "sys {current_date}", "Query: {query}\n{meta_note}\n{serp_grounding}",
                                grounding=False, feedback="- [major] checklist.c1: fabricated entity")
        assert "failed review" in captured["user"] and "fabricated entity" in captured["user"]

    def test_rubric_shape_validation_rejects_malformed(self):
        from src.rubric_gen import _validate_rubric
        with pytest.raises(ValueError):
            _validate_rubric({"intent_interpretation": "x", "checklist": []})  # empty checklist
        with pytest.raises(ValueError):
            _validate_rubric({"checklist": [{"id": "c1", "desc": "a", "weight": 3}]})  # no intent
        ok = {"intent_interpretation": "x", "checklist": [{"id": "c1", "desc": "a", "weight": 3}]}
        _validate_rubric(ok)  # defaults filled in
        assert ok["freshness_window"] == "any" and ok["disqualifiers"] == []

    def test_review_one_votes_and_merges_issues(self, monkeypatch):
        from src import rubric_review
        outs = {"m1": {"verdict": "pass", "issues": []},
                "m2": {"verdict": "fail", "issues": [{"field": "checklist.c1", "severity": "major", "note": "filler"}]},
                "m3": {"verdict": "pass", "issues": [{"field": "weights", "severity": "minor", "note": "c3 weight"}]}}
        monkeypatch.setattr(rubric_review, "_call_reviewer", lambda m, u: outs[m])
        monkeypatch.setattr("src.serp.serp_both", lambda q, k=6: [])
        q = {"qid": "q1", "query": "test", "intent": "informational"}
        rec = {"qid": "q1", "rubric": {"intent_interpretation": "x", "checklist": []}}
        rev = rubric_review.review_one(q, rec, ["m1", "m2", "m3"])
        assert rev["verdict"] == "pass"
        assert len(rev["issues"]) == 2 and rev["votes"]["m2"] == "fail"


class TestCalibration:
    def _pairs(self):
        mk = lambda i, w: {"qid": f"q{i}", "system_x": "octen", "system_y": "exa", "winner": w}
        return ([mk(i, "octen") for i in range(10)]
                + [mk(i + 10, "exa") for i in range(30)]
                + [mk(i + 40, "tie") for i in range(10)])

    def test_sample_stratified_and_deterministic(self):
        from scripts.calibration import sample_pairs
        s1 = sample_pairs(self._pairs(), 20, "octen", seed=7)
        s2 = sample_pairs(self._pairs(), 20, "octen", seed=7)
        assert [p["qid"] for p in s1] == [p["qid"] for p in s2]  # reproducible
        assert len(s1) == 20
        outcomes = {p["winner"] for p in s1}
        assert "tie" in outcomes and "octen" in outcomes and "exa" in outcomes

    def test_human_majority(self):
        from scripts.calibration import human_majority
        assert human_majority(["octen", "octen"]) == "octen"
        assert human_majority(["octen", "exa"]) is None          # 2 annotators must agree
        assert human_majority(["octen", "octen", "tie"]) == "octen"
        assert human_majority(["octen", "exa", "tie"]) is None   # no strict majority

    def test_score_labels_math(self):
        from scripts.calibration import score_labels
        pairs = {("q1", "exa"): {"winner": "exa"}, ("q2", "exa"): {"winner": "tie"},
                 ("q3", "exa"): {"winner": "octen"}}
        alice = [{"qid": q, "system_y": "exa", "annotator": "alice", "choice": c}
                 for q, c in (("q1", "exa"), ("q2", "exa"), ("q3", "exa"))]
        bob = [{"qid": q, "system_y": "exa", "annotator": "bob", "choice": c}
               for q, c in (("q1", "exa"), ("q2", "tie"), ("q3", "exa"))]
        s = score_labels(pairs, [alice, bob])
        # q1: consensus exa, judge exa → agree; q2: humans disagree → excluded;
        # q3: consensus exa, judge octen → hard flip
        assert s["pairs_labeled_by_all"] == 3
        assert s["human_disagreement"] == 1
        assert s["consensus_pairs"] == 2
        assert s["judge_agreement"] == 1 and s["hard_flips"] == 1


class TestJudgeProbes:
    def _resp(self, n=10):
        from src.backends import SearchResponse, SearchResult
        return SearchResponse("octen", "q", [
            SearchResult(rank=i, title=f"t{i}", url=f"https://e.com/{i}", snippet="s" * 100)
            for i in range(1, n + 1)], 1.0)

    def test_degrade_variants(self):
        from scripts.judge_probes import degrade
        base = self._resp()
        d1 = degrade(base, "drop_top1", 10)
        assert len(d1.results) == 9 and d1.results[0].title == "t2" and d1.results[0].rank == 1
        d3 = degrade(base, "drop_top3", 10)
        assert len(d3.results) == 7 and d3.results[0].title == "t4"
        rev = degrade(base, "reverse_order", 10)
        assert rev.results[0].title == "t10" and rev.results[0].rank == 1
        tr = degrade(base, "truncate_snippets", 10)
        assert all(len(r.snippet) == 61 for r in tr.results)
        assert base.results[0].snippet == "s" * 100  # deep copy: base untouched
        assert base.results[0].rank == 1

    def test_outcome_rules(self):
        from scripts.judge_probes import outcome_of
        assert outcome_of("original", "drop_top1") == "pass"
        assert outcome_of("tie", "drop_top1") == "soft_miss"
        assert outcome_of("tie", "drop_top3") == "hard_fail"    # strict probe
        assert outcome_of("degraded", "reverse_order") == "hard_fail"

    def test_pick_bases_one_per_qid_and_full_lists_only(self):
        from scripts.judge_probes import pick_bases
        responses = {("q1", "octen"): {"results": [{}] * 10}, ("q1", "exa"): {"results": [{}] * 10},
                     ("q2", "octen"): {"results": [{}] * 3},   # too short → excluded
                     ("q3", "exa"): {"results": [{}] * 10, "error": "boom"}}  # errored → excluded
        qmeta = {q: {"vertical": "general"} for q in ("q1", "q2", "q3")}
        picked = pick_bases(responses, qmeta, 5, seed=1)
        assert len(picked) == 1 and picked[0][0] == "q1"


class TestBenchmarkLatency:
    def test_server_latency_cell(self):
        from scripts.benchmark_report import _server_latency_cell
        # backend reports server latency → P50/P95
        assert _server_latency_cell([{"searches": [{"reported_latency_ms": 120.0}]},
                                     {"searches": [{"reported_latency_ms": 140.0}]}]) == "120 / 140"
        # backend returns no server time (e.g. parallel) → blank
        assert _server_latency_cell([{"searches": [{"reported_latency_ms": None}]}]) == "—"
        # run predates latency capture (no search trail, e.g. imported run) → blank
        assert _server_latency_cell([{"n_searches": 1}]) == "—"


class TestJudgeFamilies:
    def test_agreement_and_flips(self):
        from scripts.judge_families import agreement_stats
        v = {"j1": {("q1", "e"): "octen", ("q2", "e"): "tie", ("q3", "e"): "exa"},
             "j2": {("q1", "e"): "octen", ("q2", "e"): "exa", ("q3", "e"): "octen"}}
        s = agreement_stats(v)[("j1", "j2")]
        assert s["n"] == 3 and s["exact"] == 1
        assert s["hard_flips"] == 1  # only q3 is decisive-vs-decisive opposite; q2 involves a tie

    def test_majority_deviation(self):
        from scripts.judge_families import majority_deviation
        v = {"base": {("q1", "e"): "exa", ("q2", "e"): "octen"},
             "j1": {("q1", "e"): "octen", ("q2", "e"): "octen"},
             "j2": {("q1", "e"): "octen", ("q2", "e"): "octen"}}
        d = majority_deviation("base", v)
        assert d["others_unanimous"] == 2 and d["baseline_outvoted"] == 1
        assert d["outvoted_keys"] == [("q1", "e")]


class TestLLMJsonRobustness:
    @pytest.fixture(autouse=True)
    def _no_retry_sleep(self, monkeypatch):
        monkeypatch.setattr(llm_mod, "_RETRY_SLEEP", 0)

    def test_plain_json(self, monkeypatch):
        monkeypatch.setattr(llm_mod, "call_llm_text", lambda *a, **k: '{"a": 1}')
        assert llm_mod.call_llm_json("s", "u", "m") == {"a": 1}

    def test_fenced_json(self, monkeypatch):
        monkeypatch.setattr(llm_mod, "call_llm_text", lambda *a, **k: '```json\n{"a": 1}\n```')
        assert llm_mod.call_llm_json("s", "u", "m") == {"a": 1}

    def test_prose_wrapped_json(self, monkeypatch):
        text = 'Looking at both result sets carefully:\n\n{"overall": "A", "n": 2}\n\nHope this helps!'
        monkeypatch.setattr(llm_mod, "call_llm_text", lambda *a, **k: text)
        assert llm_mod.call_llm_json("s", "u", "m")["overall"] == "A"

    def test_multiple_concatenated_objects_takes_first(self, monkeypatch):
        # models occasionally emit two actions in one turn; the first complete object wins
        text = '{"action": "search", "query": "a"}\n{"action": "answer", "answer": "b"}'
        monkeypatch.setattr(llm_mod, "call_llm_text", lambda *a, **k: text)
        assert llm_mod.call_llm_json("s", "u", "m") == {"action": "search", "query": "a"}

    def test_json_mode_sets_response_format(self):
        from src.llm import _openrouter_body
        body = _openrouter_body("claude-sonnet-4-6", "s", "u", 512, json_mode=True)
        assert body["response_format"] == {"type": "json_object"}
        assert "response_format" not in _openrouter_body("claude-sonnet-4-6", "s", "u", 512, json_mode=False)

    def test_temperature_passthrough(self):
        from src.llm import _openrouter_body
        assert _openrouter_body("claude-sonnet-4-6", "s", "u", 512, False, temperature=0)["temperature"] == 0
        assert "temperature" not in _openrouter_body("claude-sonnet-4-6", "s", "u", 512, False)

    def test_control_chars_in_strings_tolerated(self, monkeypatch):
        # raw newline inside a JSON string value — strict json.loads rejects, ours must not
        monkeypatch.setattr(llm_mod, "call_llm_text", lambda *a, **k: '{"a": "line1\nline2"}')
        assert llm_mod.call_llm_json("s", "u", "m") == {"a": "line1\nline2"}

    def test_garbage_raises(self, monkeypatch):
        monkeypatch.setattr(llm_mod, "call_llm_text", lambda *a, **k: "no json here")
        with pytest.raises(json.JSONDecodeError):
            llm_mod.call_llm_json("s", "u", "m")

    def test_inner_fences_preserved(self):
        # MULTILINE fence-stripping used to delete fences INSIDE string values
        text = '{"answer": "use\n```nginx\nproxy_buffering off;\n```\ndone"}'
        out = llm_mod.parse_llm_json(text)
        assert "```nginx" in out["answer"] and "```" in out["answer"].rsplit("done", 1)[0]

    def test_empty_completion_raises_not_returns(self, monkeypatch):
        monkeypatch.setattr(llm_mod, "_fetch_text", lambda *a, **k: "  ")
        with pytest.raises(RuntimeError, match="empty completion"):
            llm_mod.call_llm_text("s", "u", "m", attempts=2)

    def test_openrouter_id_strictness(self):
        from src.llm import _openrouter_id
        assert _openrouter_id("openai/gpt-5.2") == "openai/gpt-5.2"          # passthrough
        assert _openrouter_id("claude-sonnet-4-6") == "anthropic/claude-sonnet-4.6"
        assert _openrouter_id("claude-opus-4-8") == "anthropic/claude-opus-4.8"
        with pytest.raises(ValueError):
            _openrouter_id("claude-sonnet-4-5-20250929")  # snapshot names used to 404 mid-run
        with pytest.raises(ValueError):
            _openrouter_id("gpt-4o")  # non-Anthropic short name used to get 'anthropic/' prefixed
