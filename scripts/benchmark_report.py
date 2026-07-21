"""Markdown report for open-source benchmark runs (SimpleQA / FreshQA).

Reads an agent_eval run directory's agent_eval.jsonl, recomputes the OFFICIAL metrics
(src/grading), and emits a clean Markdown report — SimpleQA F1/CGA table, FreshQA accuracy + CI
+ per-category breakdowns, plus agent behavior (searches / rewrite). Self-built runs use
scripts/gen_report.py instead; this one is for meta.benchmark sets.

Usage: python -m scripts.benchmark_report --run results/<run> [--out <run>/BENCHMARK_REPORT.md]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.common import load_jsonl
from src.grading import freshqa_metrics, simpleqa_metrics


def _latest(recs: list[dict]) -> list[dict]:
    """Last record per (qid, backend) — a resumed run may carry an error then a graded retry."""
    d: dict[tuple, dict] = {}
    for r in recs:
        d[(r["qid"], r["backend"])] = r
    return list(d.values())


def _behavior_row(rs: list[dict]) -> str:
    ok = [r for r in rs if "error" not in r]
    if not ok:
        return "—"
    avg_s = sum(r.get("n_searches", 0) for r in ok) / len(ok)
    rw = sum(1 for r in ok if r.get("rewrote_query")) / len(ok)
    return f"{avg_s:.1f} / {rw:.0%}"


def build_report(run: Path) -> str:
    recs = _latest(load_jsonl(run / "agent_eval.jsonl"))
    by_be: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_be[r["backend"]].append(r)
    backends = sorted(by_be)

    sqa = [r for r in recs if r.get("grade_mode") == "simpleqa" and r.get("grade")]
    fresh = [r for r in recs if r.get("grade_mode") == "freshqa" and r.get("grade")]

    L: list[str] = [f"# 开源基准评测报告 — {run.name}", ""]
    L.append(f"- 运行目录:`{run}` ｜ backends:{', '.join(backends)}")
    errs = sum(1 for r in recs if "error" in r and not r.get("grade"))
    L.append(f"- 记录:{len(recs)} episode(错误 {errs};已按 (qid,backend) 去重取最新)")
    L.append("")

    # ── SimpleQA ──
    if sqa:
        variant = "verified" if any(r["qid"].startswith("simpleqa-v-") for r in sqa) else "full"
        n_q = len(set(r["qid"] for r in sqa))
        L.append(f"## SimpleQA（{variant}，{n_q} 题）")
        L.append("> 官方指标:correct-rate=正确率;CGA=作答中正确率;**F1=二者调和均值(主排名指标)**。")
        L.append("")
        L.append("| 排名 | backend | **F1** | 正确率 | CGA | 正确/错误/未答 | 搜索次数/改写率 |")
        L.append("|--:|---|--:|--:|--:|:--|:--|")
        rows = []
        for be in backends:
            g = [r["grade"] for r in by_be[be] if r.get("grade_mode") == "simpleqa" and r.get("grade")]
            if g:
                rows.append((be, simpleqa_metrics(g)))
        for i, (be, m) in enumerate(sorted(rows, key=lambda x: -x[1]["f1"]), 1):
            L.append(f"| {i} | {be} | **{m['f1']:.1%}** | {m['correct_rate']:.1%} | "
                     f"{m['correct_given_attempted']:.1%} | {m['correct']}/{m['incorrect']}/{m['not_attempted']} "
                     f"| {_behavior_row([r for r in by_be[be] if r.get('grade_mode')=='simpleqa'])} |")
        L.append("")

    # ── FreshQA ──
    if fresh:
        n_q = len(set(r["qid"] for r in fresh))
        L.append(f"## FreshQA（FreshEval，{n_q} 题）")
        L.append("> 二元 correct/incorrect;准确率带 95% Wilson 置信区间;按 fact_type / false_premise 分项。")
        L.append("")
        L.append("| 排名 | backend | **准确率 (95% CI)** | n | 搜索次数/改写率 |")
        L.append("|--:|---|--:|--:|:--|")
        fm = {be: freshqa_metrics([{"grade": r["grade"], "meta": r.get("bench_meta", {})}
                                   for r in by_be[be] if r.get("grade_mode") == "freshqa" and r.get("grade")])
              for be in backends if any(r.get("grade_mode") == "freshqa" for r in by_be[be])}
        for i, be in enumerate(sorted(fm, key=lambda b: -fm[b]["accuracy"]), 1):
            m = fm[be]
            L.append(f"| {i} | {be} | **{m['accuracy_ci']}** | {m['n']} "
                     f"| {_behavior_row([r for r in by_be[be] if r.get('grade_mode')=='freshqa'])} |")
        L.append("")
        # per-category breakdown (backends × fact_type)
        cats = sorted({k for m in fm.values() for k in m["breakdowns"]["fact_type"]})
        if cats:
            L.append("**按 fact_type 分项准确率:**")
            L.append("")
            L.append("| backend | " + " | ".join(cats) + " |")
            L.append("|---|" + "|".join("--:" for _ in cats) + "|")
            for be in sorted(fm, key=lambda b: -fm[b]["accuracy"]):
                ft = fm[be]["breakdowns"]["fact_type"]
                L.append(f"| {be} | " + " | ".join(
                    (f"{ft[c]['accuracy']:.0%}(n={ft[c]['n']})" if c in ft else "—") for c in cats) + " |")
            L.append("")
            # false-premise
            L.append("**false-premise(需识破假前提才算对):**")
            L.append("")
            L.append("| backend | " + " | ".join("假前提=" + k for k in ("True", "False")) + " |")
            L.append("|---|--:|--:|")
            for be in sorted(fm, key=lambda b: -fm[b]["accuracy"]):
                fp = fm[be]["breakdowns"]["false_premise"]
                L.append(f"| {be} | " + " | ".join(
                    (f"{fp[k]['accuracy']:.0%}(n={fp[k]['n']})" if k in fp else "—") for k in ("True", "False")) + " |")
            L.append("")

    if not sqa and not fresh:
        L.append("_(该运行没有 SimpleQA/FreshQA 记录;自建集报告请用 scripts/gen_report.py)_")

    L.append("## 附录")
    L.append("- SimpleQA:openai/simple-evals A/B/C 协议;F1 = 调和均值(correct-rate, CGA)。")
    L.append("- FreshQA:freshllms/freshqa FreshEval;多答案 ' | ' 拼接;strict 模式不容忍幻觉/过期。")
    L.append("- 搜索次数/改写率 = agent 端到端行为(所有 backend 用同一 agent+预算)。")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="agent_eval run dir (contains agent_eval.jsonl)")
    ap.add_argument("--out", default=None, help="defaults to <run>/BENCHMARK_REPORT.md")
    args = ap.parse_args()
    run = Path(args.run)
    md = build_report(run)
    out = Path(args.out) if args.out else run / "BENCHMARK_REPORT.md"
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n[written to {out}]")


if __name__ == "__main__":
    main()
