# search-eval — a web-search evaluation framework

Measure how good a web-search backend is with a re-runnable closed loop: **labeled queries →
multi-backend fetch → LLM pairwise judge + objective anchors + end-to-end agent eval → sliced reports
→ regression diff.** Everything is JSONL and git-versioned, so any two runs are directly comparable.

Two ways to use it, mixable:

1. **Build your own in-house benchmark** — synthesize/ingest queries, auto-resolve gold answers & URLs,
   generate SERP-grounded rubrics, and score your backend against competitors with a blind,
   position-debiased judge. Answers "*where* do we trail, and *why*", on your own traffic shape.
2. **Plug in open-source benchmarks** — SimpleQA & FreshQA run through the same agent-eval harness; a
   `meta.benchmark` tag auto-selects the official grader (SimpleQA A/B/C → F1; FreshEval → accuracy).
   Answers with standardized, comparable-to-the-literature numbers.

[What it measures](#what-it-measures) · [Results we've run](#results-weve-run) · [Requirements](#requirements) · [Usage](#usage)

## What it measures

**Object under test:** a web-search backend (`--ours`, default `octen`) vs competitors (`exa`, `brave`,
`tavily`, `parallel`, `perplexity`, …). The framework answers three questions with three methods that
**cross-check each other** — a conclusion is trusted only when all three agree:

| Question | Method | Signal | LLM? |
|---|---|---|---|
| Is our **result list** better than a competitor's? | **Pairwise judge** (blind, position-swapped, rubric-grounded) | win-rate + per-dimension gaps | yes |
| Do we return the **known-correct page**? | **Objective anchors** | hit@1 / hit@k (exact URL match) | no |
| Does an **agent answer correctly** using our results? | **End-to-end agent eval** | answer accuracy / rubric coverage / official grade | yes |

## Results we've run

Fast-tier bakeoff on the open benchmarks (2026-07-14), scored with the official graders in
`src/grading.py`. Generator + judge = `minimax/minimax-m2.5`; single-shot (1 search/query); fast-tier
configs (`exa`=instant · `tavily`=ultra-fast · `parallel`=turbo · `octen`=default).

| backend | SimpleQA (4326) | FreshQA (600) |
|---|--:|--:|
| **octen** | **95.2%** (F1 96.5) | **55.8%** |
| exa-instant | 89.2% (F1 92.0) | 52.8% |
| parallel-turbo | 88.6% (F1 91.3) | 52.8% |
| tavily-ultrafast | 69.9% (F1 79.5) | 43.2% |

Full tables, exact setup, and reproduce steps: **[`docs/benchmarks/`](docs/benchmarks/README.md)**.
Every number regenerates from run artifacts via `scripts/benchmark_report` — nothing is hand-entered.

## Requirements

- **Python 3.10+** (CI runs 3.12). One runtime dependency — `requests`:
  ```bash
  pip install -r requirements.txt   # requests>=2.31 (+ pytest>=8 for the mocked test suite)
  ```
- **API keys** for the stages you run, in `.env` at the repo root (git-ignored, auto-loaded). Report
  generation and the anchor CI gate need no keys.

  | Group | Keys | Needed by |
  |---|---|---|
  | **LLM provider** (one) | `OPENROUTER_API_KEY` *(required for the cross-family rubric-review panel)* or `ANTHROPIC_API_KEY` | every LLM stage |
  | **Search backends** | `OCTEN_API_KEY` + one per competitor: `EXA_API_KEY` / `BRAVE_API_KEY` / `TAVILY_API_KEY` / `PARALLEL_API_KEY` / `PERPLEXITY_API_KEY` (fast tiers reuse these) | `run_eval`, `agent_eval`, `triage --probe` |
  | **SERP oracle** (eval-set build only) | `SERPAPI_API_KEY` or `FIRECRAWL_API_KEY` | `rubric_gen`, `rubric_review`, `gold_resolver`, `realtime_synth` |

  Model selectors (`AGENT_MODEL` / `GRADER_MODEL` / `JUDGE_MODEL` / `RUBRIC_MODEL`) default to
  `claude-sonnet-4-6`; benchmark sets default to `minimax/minimax-m2.5`. Copy `.env.example` → `.env`.

## Usage

All commands run from the repo root. Start with the fully-mocked test suite (no keys needed):

```bash
python -m pytest tests/ -q
```

**Run an open-source benchmark** (fastest path to a number; `meta.benchmark` auto-routes the grader):

```bash
# Build the selectable sets from the bundled CSVs (data/datasets/):
#   data/simpleqa_verified.jsonl (1000) · data/simpleqa_full.jsonl (4326) · data/freshqa.jsonl (600)
python -m scripts.load_benchmarks

# Run. Benchmark sets DEFAULT to the single-shot scaffold — the reproducible standard protocol:
# raw question verbatim → ONE search → evidence-classification answer prompt → official grader
# (temp=0, max_tokens=2048; full-length octen highlights + uncapped parallel excerpts are
# auto-enabled). Numbers are reproducible run-over-run within ±3pp.
python -m src.agent_eval --queries data/simpleqa_verified.jsonl \
    --backends octen exa-instant parallel-turbo tavily-ultrafast \
    --k 10 --limit 100 --out results/simpleqa_$(date +%Y%m%d) --yes

# Optional: agent scaffold (multi-turn query-rewriting agent, ≤3 searches) — measures the
# end-to-end value to an agent instead of raw retrieval. NOT comparable with single-shot
# numbers; agent iteration compensates weak retrieval and compresses backend gaps.
python -m src.agent_eval --queries data/freshqa.jsonl \
    --backends octen exa-instant parallel-turbo tavily-ultrafast \
    --scaffold agent --k 8 --max-searches 3 --out results/freshqa_agent_$(date +%Y%m%d) --yes

# Report: SimpleQA F1/CGA table; FreshQA accuracy + CI + per-category breakdowns
python -m scripts.benchmark_report --run results/simpleqa_$(date +%Y%m%d)
```

Scaffold rules of thumb: **single** (benchmark default) = raw natural-question retrieval quality,
externally comparable, quote this one; **agent** (self-built-set default) = agent-in-the-loop
end-to-end. Always report which scaffold produced a number — they differ by 10-20pp.
> Override models per run with `--agent-model` / `--grader-model` (flags beat `$AGENT_MODEL`/`$GRADER_MODEL` beat per-set default). Chosen models are printed and recorded in `run_meta.json`.

**Build & grow an in-house set** (a 200-query set already ships in `data/main_queries.jsonl`; order
matters: synth → gold → rubrics → review → merge):

```bash
python -m src.query_synth --topic "vector databases" --n 15 --vertical tech_code \
    --out data/synth_gen.jsonl --dedup-against data/main_queries.jsonl   # or query_intake for raw lists
python -m src.gold_resolver  --queries data/synth_gen.jsonl --concurrency 4
python -m src.rubric_gen     --queries data/synth_gen.jsonl --concurrency 4
python -m src.rubric_review  --queries data/synth_gen.jsonl --concurrency 4 --yes
cat data/synth_gen.jsonl >> data/main_queries.jsonl
```

**Run the eval & read the report:**

```bash
# Full run (drop --skip-judge for the judge; add it to smoke-test backend field mappings first)
python -m src.run_eval --queries data/main_queries.jsonl \
    --ours octen --competitors exa brave --k 10 --out results/run_$(date +%Y%m%d)
# Weakness map (win-rate + Wilson CI + slices + dimension gaps); add --baseline for a regression diff
python -m src.report --run results/run_20260721 --queries data/main_queries.jsonl
# Failure attribution; then a shareable Markdown report
python -m src.triage --run results/run_20260721 --probe --concurrency 8
python -m scripts.gen_report --run results/run_20260721 --queries data/main_queries.jsonl --cases 5
# End-to-end agent eval (the only variable across backends is the search service)
python -m src.agent_eval --queries data/benchmark_queries.jsonl --backends octen exa \
    --k 8 --max-searches 3 --out results/agent_run_$(date +%Y%m%d) --limit 10
```

## License

[MIT](LICENSE) © 2026 Octen.
