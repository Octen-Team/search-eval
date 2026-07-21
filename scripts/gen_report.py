"""Assemble a Markdown eval report from a run directory's artifacts.

Reads pairwise.jsonl / anchors.jsonl / responses.jsonl / triage.jsonl (+ an optional agent-eval
run) and emits a single reproducible Markdown file. Reuses the same aggregation helpers as
src/report.py so the numbers cannot drift from the console report.

Usage:
  python -m scripts.gen_report --run results/run_20260708_full \\
      --queries data/main_queries.jsonl data/realtime_20260708.jsonl \\
      [--agent-run results/agent_run_20260707_full] [--baseline results/run_XXXX] \\
      [--out results/run_20260708_full/20260708_general_REPORT.md]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.common import require_pairwise_meta
from src.report import (SLICE_KEYS, DIMENSIONS, CONFLICT_TRUST_THRESHOLD,
                        anchor_hit_rates, conflict_rate, load_jsonl, percentile,
                        wilson_ci, wtl)


def _wr_row(recs: list[dict], ours: str) -> tuple[int, int, int, float, tuple[float, float]]:
    w, t, l = wtl(recs, ours)  # shared counting — cannot drift from src.report
    dec = w + l
    return w, t, l, (w / dec if dec else 0.0), wilson_ci(w, dec)


def md_escape(s: str) -> str:
    # &/</> too: queries and evidence flow into raw HTML (<summary>) where a literal
    # '<something>' silently swallows content or breaks the <details> block
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace("|", "\\|").replace("\n", " "))


def by_comp_pre(pairs: list[dict]) -> dict[str, list[dict]]:
    d = defaultdict(list)
    for p in pairs:
        d[p["system_y"]].append(p)
    return d


def _top_urls(resp_by: dict, qid: str, backend: str, n: int = 3) -> list[str]:
    rec = resp_by.get((qid, backend))
    if not rec:
        return []
    out = []
    for r in rec.get("results", [])[:n]:
        title = (r.get("title") or "").strip()[:60]
        out.append(f"#{r['rank']} {title} — {r.get('url', '')}")
    return out


def select_cases(pairs, triage_by_qid, qmeta, ours, per_mode=4, resolver=None) -> dict[str, list[dict]]:
    """Pick representative, high-signal failure cases per mode, spread across verticals.

    Prefer high/medium-confidence, non-conflict losses (a conflict means the judge was unsure,
    so it is a poor illustration). Round-robin across verticals so one vertical doesn't dominate.
    """
    losses = [p for p in pairs
              if p["winner"] not in (ours, "tie") and not p.get("position_conflict")
              and p.get("confidence") != "low"]  # per the contract above: low-confidence verdicts are poor illustrations
    # strongest signal first: biggest weighted-score gap against us
    losses.sort(key=lambda p: p.get("weighted_y", 0) - p.get("weighted_x", 0), reverse=True)
    picked: dict[str, list[dict]] = defaultdict(list)
    seen_qids: dict[str, set] = defaultdict(set)
    vert_count: dict[tuple, int] = defaultdict(int)
    for p in losses:
        t = resolver(p) if resolver else triage_by_qid.get(p["qid"])
        if not t:
            continue
        mode = t["mode"]
        if mode not in ("INDEX_MISS_PROBED", "PENDING_L2_L4"):
            continue
        if len(picked[mode]) >= per_mode or p["qid"] in seen_qids[mode]:
            continue
        vert = qmeta.get(p["qid"], {}).get("vertical", "?")
        # cap 2 per vertical per mode for spread
        if vert_count[(mode, vert)] >= 2:
            continue
        picked[mode].append({"loss": p, "triage": t})
        seen_qids[mode].add(p["qid"])
        vert_count[(mode, vert)] += 1
    return picked


def _dom(url: str) -> str:
    from urllib.parse import urlparse
    try:
        d = urlparse(url).netloc.lower()
    except ValueError:
        return ""
    return d[4:] if d.startswith("www.") else d


# Findings thresholds — fixed rules so the section is reproducible, not editorial
IDX_MISS_MIN_SHARE = 0.4      # index coverage becomes the headline finding
RANK_MISS_MIN_SHARE = 0.3     # indexed-but-unranked (PENDING_L2_L4) becomes a headline finding
DIM_GAP_MINOR = -0.3          # freshness / snippet_quality call-out
DIM_GAP_MAJOR = -0.5          # authority / diversity call-out
WEAK_SLICE_WR = 0.25          # slice cells at or below this win rate (n>=20 decisive)
ANCHOR_GAP_PP = 0.10          # hit@1 gap vs best competitor
AGENT_TRAP_RATIO = 1.5        # ours trap-hit rate vs best competitor
LATENCY_RATIO = 1.5           # ours P50 vs best competitor


def build_findings(ours: str, pairs: list, by_comp: dict, qmeta: dict, triage: list,
                   resp_by: dict, anchors: list, agent_by_be: dict) -> list[str]:
    """Auto-derived problem statements + iteration recommendations, every one citing evidence
    from this run's artifacts. Deterministic rules — same artifacts, same findings."""
    findings = []  # (priority, title, evidence: list[str], recs: list[str])
    losses = [p for p in pairs if p["winner"] not in (ours, "tie")]

    gaps = {}
    for comp, recs in by_comp.items():
        gaps[comp] = {d: sum(r["dim_scores_x"][d] - r["dim_scores_y"][d] for r in recs) / max(len(recs), 1)
                      for d in DIMENSIONS}

    def worst(dim: str) -> tuple[str, float]:
        comp = min(gaps, key=lambda c: gaps[c][dim])
        return comp, gaps[comp][dim]

    # -- index coverage ------------------------------------------------------------------
    confirmed = [t for t in triage if t["mode"] == "INDEX_MISS_PROBED"]
    suspected = [t for t in triage if t["mode"] == "PENDING_INDEX_CHECK"]
    if triage and (len(confirmed) + len(suspected)) / len(triage) >= IDX_MISS_MIN_SHARE:
        c_share = len(confirmed) / len(triage)
        s_share = len(suspected) / len(triage)
        by_vert = Counter(qmeta.get(t["qid"], {}).get("vertical", "?") for t in confirmed + suspected)
        seen_doms, samples = set(), []
        for t in confirmed:
            for u in (t.get("detail") or {}).get("missing_urls", []):
                if _dom(u) not in seen_doms:
                    seen_doms.add(_dom(u))
                    samples.append(u)
            if len(samples) >= 5:
                break
        ev = [f"Index/recall failures: {c_share:.0%} probe-confirmed (n={len(confirmed)})"
              + (f" + {s_share:.0%} suspected-unprobed (PENDING_INDEX_CHECK, n={len(suspected)})"
                 if suspected else "") + f" of {len(triage)} triaged failures.",
              "Worst verticals: " + ", ".join(f"{v} ({n})" for v, n in by_vert.most_common(3)) + "."]
        if samples:
            ev.append("Sample missing URLs (direct-URL probe returned nothing):")
            ev.extend(f"  - `{u}`" for u in samples[:5])
        recs = ["Prioritize ingestion speed/coverage for the verticals above — for news/realtime "
                "content: news-sitemap + RSS polling and hot-signal-triggered crawl; for stable "
                "verticals: authoritative-domain seed lists.",
                f"Fixing retrieval addresses up to {c_share + s_share:.0%} of current losses; "
                "ranking work cannot recover documents the index does not hold."]
        if suspected:
            recs.append("Re-run `python -m src.triage --run <run> --probe` to confirm the "
                        "suspected-unprobed share before sizing the ingestion work.")
        findings.append((
            "P0", "Index coverage: competitors' top results are not retrievable from our index",
            ev, recs))

    # -- indexed but unranked --------------------------------------------------------------
    l24 = [t for t in triage if t["mode"] == "PENDING_L2_L4"]
    if triage and len(l24) / len(triage) >= RANK_MISS_MIN_SHARE:
        share = len(l24) / len(triage)
        seen_doms, samples = set(), []
        for t in l24:
            for u in (t.get("detail") or {}).get("suspect_gold_urls", []):
                if _dom(u) not in seen_doms:
                    seen_doms.add(_dom(u))
                    samples.append(u)
            if len(samples) >= 5:
                break
        findings.append((
            "P0", "Retrieval/ranking: documents ARE in the index but fail to surface",
            [f"{share:.0%} of triaged failures (n={len(l24)}/{len(triage)}) lost although every "
             "suspected gold URL was verified present in the index — crawling more will not fix these.",
             "Sample indexed-but-lost URLs:"]
            + [f"  - `{u}`" for u in samples[:5]],
            ["Engine-side reproduction per case: is the document in the recall candidate set for the "
             "original query? If absent → recall/matching (query analysis, CJK/tokenization, semantic "
             "recall); if present but low → ranking features (freshness/authority weights).",
             "This bucket is invisible to probe-based triage; it needs engine-internal "
             "reproduction to measure."]))

    # -- authority -----------------------------------------------------------------------
    comp_a, gap_a = worst("authority")
    if gap_a <= DIM_GAP_MAJOR:
        cnt = Counter()
        for p in losses:
            comp = p["system_y"] if p["system_x"] == ours else p["system_x"]
            ours_doms = {_dom(r["url"]) for r in resp_by.get((p["qid"], ours), {}).get("results", [])}
            for r in resp_by.get((p["qid"], comp), {}).get("results", [])[:3]:
                d = _dom(r["url"])
                if d and d not in ours_doms:
                    cnt[d] += 1
        findings.append((
            "P1", "Authority: losing to higher-authority sources we never surface",
            [f"Authority score gap {gap_a:+.2f} vs {comp_a} (worst dimension pairing).",
             "Domains most often in the competitor's winning top-3 while absent from our top-10:",
             "  " + ", ".join(f"`{d}`×{n}" for d, n in cnt.most_common(8))],
            ["Check the listed domains first: if absent from the index it is a coverage/seed problem; "
             "if indexed but unranked, add source-authority ranking signals.",
             "Re-measure via the authority column in §3 after the fix — the domain list above is the "
             "regression watchlist."]))

    # -- freshness -----------------------------------------------------------------------
    comp_f, gap_f = worst("freshness")
    if gap_f <= DIM_GAP_MINOR:
        findings.append((
            "P1" if gap_f <= DIM_GAP_MAJOR else "P2",
            "Freshness: results are older than the competition's",
            [f"Freshness score gap {gap_f:+.2f} vs {comp_f}."],
            ["Raise recrawl frequency for time-sensitive sources; expose crawl-to-serve latency as an "
             "engine metric. For realtime queries, prefer routing to the news/topical index (measured: "
             "topic=news closed most of the freshness gap on the news set)."]))

    # -- diversity -----------------------------------------------------------------------
    comp_d, gap_d = worst("diversity")
    if gap_d <= DIM_GAP_MAJOR:
        uniq = {}
        for be in (ours, comp_d):
            per_q = [len({_dom(r["url"]) for r in rec.get("results", [])[:10]})
                     for (qid, b), rec in resp_by.items() if b == be and rec.get("results")]
            uniq[be] = sum(per_q) / max(len(per_q), 1)
        # only cite domain-uniqueness when it actually explains the gap; otherwise the gap is
        # content/angle diversity, which domain counting cannot see
        dom_ev = (f"Unique domains in top-10: {ours} {uniq.get(ours, 0):.1f} vs {comp_d} "
                  f"{uniq.get(comp_d, 0):.1f}."
                  if uniq.get(comp_d, 0) - uniq.get(ours, 0) >= 0.5 else
                  "Domain-uniqueness is comparable — the gap is content/angle diversity "
                  "(same story from near-identical outlets), which domain counting cannot see.")
        findings.append((
            "P2", "Diversity: near-duplicate results crowd out coverage",
            [f"Diversity score gap {gap_d:+.2f} vs {comp_d}.", dom_ev],
            ["Apply per-domain caps / URL-variant dedup (e.g. homepage vs .aspx variants) at ranking "
             "time; judge evidence repeatedly cites duplicate homepage variants."]))

    # -- snippet quality -----------------------------------------------------------------
    comp_s, gap_s = worst("snippet_quality")
    if gap_s <= DIM_GAP_MINOR:
        findings.append((
            "P2", "Snippet quality: extraction lags the competition",
            [f"Snippet score gap {gap_s:+.2f} vs {comp_s}."],
            ["Improve highlight/extraction (query-relevant passages instead of page-head text); "
             "snippets drive both the judge dimension and agent-lane answer grounding."]))

    # -- weak slices ---------------------------------------------------------------------
    weak = []
    for key in SLICE_KEYS:
        buckets = defaultdict(list)
        for p in pairs:
            buckets[qmeta.get(p["qid"], {}).get(key, "?")].append(p)
        for val, recs in buckets.items():
            w = sum(1 for r in recs if r["winner"] == ours)
            l = sum(1 for r in recs if r["winner"] not in (ours, "tie"))
            if w + l >= 20 and w / max(w + l, 1) <= WEAK_SLICE_WR:
                weak.append((w / (w + l), f"{key}={val}", w + l))
    if weak:
        weak.sort()
        findings.append((
            "P2", "Weak slices: concentrated losses in specific query classes",
            [f"Win-rate ≤{WEAK_SLICE_WR:.0%} cells (n≥20 decisive): "
             + ", ".join(f"**{name}** {wr:.0%} (n={n})" for wr, name, n in weak[:4]) + "."],
            ["Read §9 cases for these slices specifically; slice-level losses usually share one "
             "root cause (e.g. navigational → URL/entity matching, zh → CJK analysis)."]))

    # -- anchors ------------------------------------------------------------------------
    if anchors:
        by_be = defaultdict(list)
        for a in anchors:
            by_be[a["backend"]].append(a)
        if ours in by_be:
            ours_h1 = anchor_hit_rates(by_be[ours])[0]
            best_c, best_h1 = None, 0.0
            for be, recs in by_be.items():
                if be != ours and anchor_hit_rates(recs)[0] > best_h1:
                    best_c, best_h1 = be, anchor_hit_rates(recs)[0]
            if best_c and best_h1 - ours_h1 >= ANCHOR_GAP_PP:
                findings.append((
                    "P1", "Anchor set: objective navigational gap",
                    [f"hit@1 {ours_h1:.0%} vs {best_c} {best_h1:.0%} on gold-URL anchor queries "
                     "(judge-free measurement)."],
                    ["Fix exact/official-page ranking for anchor queries first — objective, "
                     "regression-gated (scripts/anchor_gate.py), and immune to judge noise."]))

    # -- agent lane ----------------------------------------------------------------------
    if agent_by_be and ours in agent_by_be:
        def trap(be):
            rs = [r for r in agent_by_be[be] if r.get("rubric_score") is not None]
            return sum(1 for r in rs if r["disqualifiers_hit"]) / max(len(rs), 1)
        comp_traps = {be: trap(be) for be in agent_by_be if be != ours}
        if comp_traps:
            best_be = min(comp_traps, key=comp_traps.get)
            if trap(ours) >= AGENT_TRAP_RATIO * max(comp_traps[best_be], 0.01):
                findings.append((
                    "P1", "Agent lane: our results mislead agent answers into traps",
                    [f"Trap-hit rate (answers containing stale/wrong-premise content): {ours} "
                     f"{trap(ours):.0%} vs {best_be} {comp_traps[best_be]:.0%}.",
                     "Trap hits are answer-level damage — worse than an empty result, the agent "
                     "asserts outdated facts confidently."],
                    ["Downrank stale versions of updated stories (crawl-date vs story-development "
                     "signals); this is the freshness/coverage gap surfacing at answer level."]))

    # -- latency (only if we are the slow one) --- API-returned server latency only; backends
    #    that report none are skipped (no e2e fallback), so this fires only on reported figures
    by_lat = defaultdict(list)
    for (qid, be), rec in resp_by.items():
        if not rec.get("error") and rec.get("reported_latency_ms") is not None:
            by_lat[be].append(rec["reported_latency_ms"])
    if ours in by_lat and len(by_lat) > 1:
        p50 = {be: percentile(sorted(v), 0.5) for be, v in by_lat.items()}
        best = min(p50[b] for b in p50 if b != ours)
        if p50[ours] >= LATENCY_RATIO * best:
            findings.append((
                "P2", "Latency: serving is materially slower than the competition",
                [f"P50 {p50[ours]:.0f}ms vs best competitor {best:.0f}ms."],
                ["Profile the serving path; latency compounds in agent loops (2+ searches/episode)."]))

    if not findings:
        return []
    order = {"P0": 0, "P1": 1, "P2": 2}
    findings.sort(key=lambda f: order[f[0]])
    L = ["## Findings & iteration recommendations (auto-derived)", "",
         "> Derived from this run's artifacts by fixed rules (thresholds in `scripts/gen_report.py`); "
         "every finding cites its evidence. **P0** = majority failure mode, **P1** = major gap, "
         "**P2** = localized weakness.", ""]
    for i, (prio, title, ev, recs) in enumerate(findings, 1):
        L.append(f"### {i}. [{prio}] {title}")
        L.append("")
        L.append("**Evidence:**")
        # lines already indented are sub-items of the previous bullet — no extra "- " prefix
        L.extend(e if e.startswith("  ") else f"- {e}" for e in ev)
        L.append("")
        L.append("**Recommendation:**")
        L.extend(f"- {r}" for r in recs)
        L.append("")
    return L


def build(run: Path, qmeta: dict, baseline: Path | None, agent_run: Path | None,
          case_budget: int = 4) -> str:
    meta = require_pairwise_meta(run)
    ours = meta["ours"]
    pairs = [p for p in load_jsonl(run / "pairwise.jsonl") if "winner" in p]
    anchors = load_jsonl(run / "anchors.jsonl")
    responses = load_jsonl(run / "responses.jsonl")
    triage = load_jsonl(run / "triage.jsonl")
    resp_by = {(r["qid"], r["backend"]): r for r in responses}
    by_comp = by_comp_pre(pairs)  # computed ONCE — five ad-hoc regroupings used to drift-risk this

    # precompute headline figures for the executive summary
    comp_wr = {}
    for comp, recs in by_comp.items():
        w, t, l, wr, ci = _wr_row(recs, ours)
        comp_wr[comp] = (wr, ci)
    anchor_ours = None
    if anchors:
        our_a = [a for a in anchors if a["backend"] == ours]
        if our_a:
            h1, hk, _ = anchor_hit_rates(our_a)
            anchor_ours = (h1, hk)
    idx_miss = sum(1 for r in triage if r["mode"] == "INDEX_MISS_PROBED") / max(len(triage), 1) if triage else None

    L = []
    L.append(f"# Search Eval Report — `{run.name}`")
    L.append("")
    L.append(f"- **System under test (ours):** `{ours}`  vs  {', '.join(f'`{c}`' for c in meta['competitors'])}")
    L.append(f"- **Timestamp:** {meta['timestamp']}")
    L.append(f"- **Queries:** {meta.get('n_queries')} (`{'`, `'.join(meta['queries_file']) if isinstance(meta['queries_file'], list) else meta['queries_file']}`), k={meta['k']}, query-set sha `{meta.get('queries_sha')}`")
    L.append(f"- **Verdicts:** {len(pairs)}")
    L.append("")
    L.append("## Executive summary")
    L.append("")
    wr_bits = "; ".join(f"vs {c}: **{wr:.0%}** (CI {ci[0]:.0%}–{ci[1]:.0%})" for c, (wr, ci) in sorted(comp_wr.items()))
    L.append(f"- **Pairwise win-rate** (excl. tie): {wr_bits}.")
    if anchor_ours:
        L.append(f"- **Anchor coverage:** `{ours}` hit@1 {anchor_ours[0]:.0%} / hit@k {anchor_ours[1]:.0%} "
                 f"(see §4 for the competitor comparison) — the objective, CI-free regression signal.")
    if idx_miss is not None:
        L.append(f"- **Root cause (triage):** {idx_miss:.0%} of failures are index/recall misses (INDEX_MISS_PROBED), "
                 f"the rest are rank/presentation (L2-L4). See §6 for the per-vertical split that says where to point the engine first.")
    worst_dim = None
    if by_comp:
        first_comp = sorted(by_comp)[0]
        recs = by_comp[first_comp]
        diffs = {d: sum(r["dim_scores_x"][d] - r["dim_scores_y"][d] for r in recs) / max(len(recs), 1) for d in DIMENSIONS}
        worst_dim = min(diffs, key=diffs.get)
        L.append(f"- **Weakest dimension:** `{worst_dim}` ({diffs[worst_dim]:+.2f} vs {first_comp}); see §3.")
    L.append("")

    # agent records load early: both the findings section and §8 consume them
    agent_by_be: dict[str, list] = defaultdict(list)
    if agent_run and (agent_run / "agent_eval.jsonl").exists():
        for r in {(r["qid"], r["backend"]): r for r in load_jsonl(agent_run / "agent_eval.jsonl")}.values():
            agent_by_be[r["backend"]].append(r)

    L.extend(build_findings(ours, pairs, by_comp, qmeta, triage, resp_by, anchors, agent_by_be))

    # 1. overall win rate
    L.append("## 1. Overall win rate (ours vs competitors)")
    L.append("")
    L.append("| Competitor | W | T | L | Win-rate (excl. tie) | 95% CI | Conflict rate |")
    L.append("|---|---|---|---|---|---|---|")
    for comp, recs in sorted(by_comp.items()):
        w, t, l, wr, (lo, hi) = _wr_row(recs, ours)
        cr = conflict_rate(recs)
        flag = " ⚠LOW-TRUST" if cr > CONFLICT_TRUST_THRESHOLD else ""
        L.append(f"| {comp} | {w} | {t} | {l} | **{wr:.0%}** | {lo:.0%}–{hi:.0%} | {cr:.0%}{flag} |")
    L.append("")
    L.append(f"> Win-rate excludes ties. CI is the 95% Wilson interval over decisive (non-tie) verdicts. "
             f"A conflict rate above {CONFLICT_TRUST_THRESHOLD:.0%} flags the comparison as low-trust "
             f"(judge near coin-flipping — fix the rubric before trusting it).")
    L.append("")

    # 2. slice win rates
    L.append("## 2. Slice win rates (weakness map, win-rate excl. tie)")
    L.append("")
    for key in SLICE_KEYS:
        buckets = defaultdict(list)
        for p in pairs:
            buckets[qmeta.get(p["qid"], {}).get(key, "?")].append(p)
        L.append(f"**{key}**")
        L.append("")
        L.append("| Value | Win-rate | n | Conflict |")
        L.append("|---|---|---|---|")
        for val, recs in sorted(buckets.items(), key=lambda x: -len(x[1])):
            w = sum(1 for r in recs if r["winner"] == ours)
            l = sum(1 for r in recs if r["winner"] not in (ours, "tie"))
            cr = conflict_rate(recs)
            flag = " ⚠" if cr > CONFLICT_TRUST_THRESHOLD else ""
            small = " *(n<20)*" if (w + l) < 20 else ""
            L.append(f"| {md_escape(str(val))} | {w / max(w + l, 1):.0%}{small} | {len(recs)} | {cr:.0%}{flag} |")
        L.append("")

    # 3. dimension gaps
    L.append("## 3. Average dimension score gap (ours − competitor; negative = weakness)")
    L.append("")
    L.append("| Competitor | " + " | ".join(DIMENSIONS) + " |")
    L.append("|---|" + "---|" * len(DIMENSIONS))
    for comp, recs in sorted(by_comp.items()):
        diffs = {d: sum(r["dim_scores_x"][d] - r["dim_scores_y"][d] for r in recs) / max(len(recs), 1) for d in DIMENSIONS}
        L.append(f"| {comp} | " + " | ".join(f"{diffs[d]:+.2f}" for d in DIMENSIONS) + " |")
    L.append("")

    # 4. anchors
    if anchors:
        L.append("## 4. Anchor set (objective navigational queries)")
        L.append("")
        L.append("| Backend | hit@1 | hit@k | n |")
        L.append("|---|---|---|---|")
        by_be = defaultdict(list)
        for a in anchors:
            by_be[a["backend"]].append(a)
        for be, recs in sorted(by_be.items()):
            h1, hk, n = anchor_hit_rates(recs)
            bold = "**" if be == ours else ""
            L.append(f"| {bold}{be}{bold} | {h1:.0%} | {hk:.0%} | {n} |")
        L.append("")

    # 5. latency — API-returned SERVER latency only (octen/exa/tavily). Backends that report no
    #    server time (parallel/brave/perplexity) are left blank; no e2e round-trip substitution.
    if responses:
        L.append("## 5. Latency P50/P90 (ms) — API-returned server latency (blank if the backend reports none)")
        L.append("")
        L.append("| Backend | latency P50 | latency P90 | n |")
        L.append("|---|--:|--:|--:|")
        srv = defaultdict(list)
        backends_seen = set()
        for r in responses:
            if r.get("error"):
                continue
            backends_seen.add(r["backend"])
            if r.get("reported_latency_ms") is not None:
                srv[r["backend"]].append(r["reported_latency_ms"])
        for be in sorted(backends_seen):
            s = sorted(srv.get(be, []))
            if s:
                L.append(f"| {be} | {percentile(s, 0.5):.0f} | {percentile(s, 0.9):.0f} | {len(s)} |")
            else:
                L.append(f"| {be} | — | — | 0 |")
        L.append("")

    # 6. triage
    if triage:
        L.append("## 6. Failure attribution (triage)")
        L.append("")
        dist = Counter(r["mode"] for r in triage)
        L.append("| Mode | Count | Share |")
        L.append("|---|---|---|")
        for mode, n in dist.most_common():
            L.append(f"| {mode} | {n} | {n / len(triage):.0%} |")
        L.append("")
        # mode by vertical (unique failing queries; index-miss wins if any competitor-loss probed a miss)
        byq = defaultdict(list)
        for t in triage:
            byq[t["qid"]].append(t["mode"])
        qmode = {q: ("INDEX_MISS_PROBED" if "INDEX_MISS_PROBED" in ms else ms[0]) for q, ms in byq.items()}
        vt = defaultdict(Counter)
        for q, m in qmode.items():
            vt[qmeta.get(q, {}).get("vertical", "?")][m] += 1
        L.append("**Failure mode by vertical** (unique failing queries; where to point the engine first):")
        L.append("")
        L.append("| Vertical | Failing | Index-miss | Rank/present (L2-L4) |")
        L.append("|---|---|---|---|")
        for v, c in sorted(vt.items(), key=lambda x: -sum(x[1].values())):
            tot = sum(c.values())
            L.append(f"| {v} | {tot} | {c.get('INDEX_MISS_PROBED', 0) / tot:.0%} | {c.get('PENDING_L2_L4', 0) / tot:.0%} |")
        L.append("")

    # 7. baseline diff
    if baseline:
        base_pairs = {(p["qid"], p["system_y"]): p for p in load_jsonl(baseline / "pairwise.jsonl") if "winner" in p}
        reg, imp = [], []
        for p in pairs:
            b = base_pairs.get((p["qid"], p["system_y"]))
            if not b:
                continue
            was, now = b["winner"] == ours, p["winner"] == ours
            if was and not now:
                reg.append(p)
            elif not was and now:
                imp.append(p)
        L.append(f"## 7. Regression vs baseline `{baseline.name}`")
        L.append("")
        L.append(f"- Regressions (won→lost): **{len(reg)}**")
        L.append(f"- Improvements (lost→won): **{len(imp)}**")
        L.append("")

    # 8. agent eval
    if agent_by_be:
        by_be = agent_by_be
        L.append(f"## 8. End-to-end agent eval (`{agent_run.name}`)")
        L.append("")
        L.append("| Backend | QA accuracy (gold) | Rubric score | Intent addressed | Trap-hit | Avg searches |")
        L.append("|---|---|---|---|---|---|")
        for be, rs in sorted(by_be.items()):
            gold = [r for r in rs if r.get("grade")]
            acc = sum(1 for r in gold if r["grade"] == "CORRECT") / max(len(gold), 1)
            rub = [r for r in rs if r.get("rubric_score") is not None]
            rmean = sum(r["rubric_score"] for r in rub) / max(len(rub), 1)
            intent = sum(r["intent_addressed"] for r in rub) / max(len(rub), 1)
            trap = sum(1 for r in rub if r["disqualifiers_hit"]) / max(len(rub), 1)
            ok = [r for r in rs if "error" not in r]
            avs = sum(r["n_searches"] for r in ok) / max(len(ok), 1)
            L.append(f"| {be} | {acc:.0%} (n={len(gold)}) | {rmean:.2f} (n={len(rub)}) | {intent:.0%} | {trap:.0%} | {avs:.1f} |")
        L.append("")

    # 9. typical failure cases (for engine investigation)
    if triage and case_budget > 0:
        # pair each loss with the triage record for the SAME competitor — falling back to
        # any-competitor only when an exact match is absent (older triage files without the
        # competitor field). Mixing used to print 'Lost to brave' with exa's missing URLs.
        triage_exact = {(t["qid"], t.get("competitor")): t for t in triage}
        triage_any: dict[str, dict] = {}
        for t in triage:
            cur = triage_any.get(t["qid"])
            if cur is None or (t["mode"] == "INDEX_MISS_PROBED" and cur["mode"] != "INDEX_MISS_PROBED"):
                triage_any[t["qid"]] = t

        def resolve(loss: dict) -> dict | None:
            return triage_exact.get((loss["qid"], loss.get("system_y"))) or triage_any.get(loss["qid"])

        cases = select_cases(pairs, triage_any, qmeta, ours, per_mode=case_budget, resolver=resolve)
        MODE_TITLE = {
            "INDEX_MISS_PROBED": "Index / recall misses — the competitor's top results are not retrievable from our backend",
            "PENDING_L2_L4": "Retrieved but out-ranked / poorly presented — the doc is reachable but loses on ranking or snippet",
        }
        if any(cases.values()):
            L.append("## 9. Typical failure cases (for engine investigation)")
            L.append("")
            for mode in ("INDEX_MISS_PROBED", "PENDING_L2_L4"):
                if not cases.get(mode):
                    continue
                L.append(f"### {mode} — {MODE_TITLE[mode]}")
                L.append("")
                for c in cases[mode]:
                    p, t = c["loss"], c["triage"]
                    q = qmeta.get(p["qid"], {})
                    L.append(f"<details><summary><b>{p['qid']}</b> — {md_escape(q.get('query', ''))[:110]}</summary>")
                    L.append("")
                    L.append(f"- **Labels:** vertical={q.get('vertical')} · form={q.get('form')} · "
                             f"difficulty={q.get('difficulty')} · intent={q.get('intent')}")
                    L.append(f"- **Lost to `{p['system_y']}`** ({p.get('confidence')} confidence); "
                             f"weighted score {p.get('weighted_x')} vs {p.get('weighted_y')}")
                    L.append(f"- **Judge evidence:** {md_escape(p.get('evidence', ''))}")
                    if mode == "INDEX_MISS_PROBED":
                        miss = t.get("detail", {}).get("missing_urls", [])
                        L.append(f"- **Not retrievable from `{ours}` (probe missed):**")
                        for u in miss[:3]:
                            L.append(f"    - {u}")
                    else:
                        gap = t.get("detail", {}).get("dim_gap", {})
                        if gap:
                            worst = sorted(gap.items(), key=lambda x: x[1])[:2]
                            L.append("- **Largest dimension gaps:** " + ", ".join(f"{d} {v:+.1f}" for d, v in worst))
                    ours_top = _top_urls(resp_by, p["qid"], ours)
                    comp_top = _top_urls(resp_by, p["qid"], p["system_y"])
                    L.append(f"- **`{ours}` top-3:**")
                    for u in ours_top:
                        L.append(f"    - {md_escape(u)}")
                    L.append(f"- **`{p['system_y']}` top-3:**")
                    for u in comp_top:
                        L.append(f"    - {md_escape(u)}")
                    L.append("")
                    L.append("</details>")
                    L.append("")

    # failure case pointer
    L.append("---")
    L.append(f"*Generated by `scripts/gen_report.py` from `{run.name}` artifacts. "
             f"Numbers reproduce `python -m src.report`. Raw data: `pairwise.jsonl`, `anchors.jsonl`, "
             f"`triage.jsonl`, `losses.jsonl`.*")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--queries", required=True, nargs="+")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--agent-run", default=None)
    ap.add_argument("--cases", type=int, default=4, help="typical failure cases to include per mode (0 = omit)")
    ap.add_argument("--out", default=None, help="defaults to <run>/20260708_general_REPORT.md")
    args = ap.parse_args()

    run = Path(args.run)
    qmeta = {q["qid"]: q for f in args.queries for q in load_jsonl(Path(f))}
    md = build(run, qmeta, Path(args.baseline) if args.baseline else None,
               Path(args.agent_run) if args.agent_run else None, case_budget=args.cases)
    out = Path(args.out) if args.out else run / "20260708_general_REPORT.md"
    out.write_text(md + "\n", encoding="utf-8")
    print(f"→ {out} ({len(md.splitlines())} lines)")


if __name__ == "__main__":
    main()
