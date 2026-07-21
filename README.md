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

Install, then smoke-test with the fully-mocked suite (no keys needed); commands run from the repo root:

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
cp .env.example .env          # then fill in keys for the stages you run
```

### 1 · Score a backend on an open benchmark

`agent_eval` runs each query through a backend and grades the answer; the `meta.benchmark` tag
auto-selects the official grader. **These are the exact commands behind
[Results we've run](#results-weve-run)** — benchmark sets default to the single-shot scaffold +
`minimax/minimax-m2.5` for generation and grading, so a fresh clone reproduces the table:

```bash
# Build the sets from the bundled CSVs → data/{simpleqa_verified,simpleqa_full,freshqa}.jsonl
python -m scripts.load_benchmarks

# SimpleQA full (4326 q) + FreshQA (600 q) across the four fast tiers.
# The full 4326 × 4 backends is large — add --limit N for a quick sample.
python -m src.agent_eval --queries data/simpleqa_full.jsonl \
    --backends octen exa-instant parallel-turbo tavily-ultrafast \
    --k 10 --out results/simpleqa_full --yes
python -m src.agent_eval --queries data/freshqa.jsonl \
    --backends octen exa-instant parallel-turbo tavily-ultrafast \
    --k 10 --out results/freshqa --yes

# Reports → SimpleQA F1/CGA · FreshQA accuracy + CI + per-category breakdowns · server latency P50/P95
python -m scripts.benchmark_report --run results/simpleqa_full
python -m scripts.benchmark_report --run results/freshqa
```

**Scaffold** (`--scaffold`, default is per-set): `single` (benchmark default) = raw question → 1
search → answer; measures *retrieval quality* and is externally comparable — quote this one. `agent`
(self-built default) = multi-turn query-rewriting agent; measures *end-to-end* value and compresses
backend gaps. They differ by 10–20pp, so always state which. Override models with `--agent-model` /
`--grader-model` (flag > `$AGENT_MODEL`/`$GRADER_MODEL` > per-set default; recorded in `run_meta.json`).

### 2 · Build your own query set

A 200-query set already ships in `data/main_queries.jsonl`. To grow it, the order is fixed —
**synth → gold → rubrics → review → merge**:

```bash
python -m src.query_synth --topic "vector databases" --n 15 --vertical tech_code \
    --out data/synth_gen.jsonl --dedup-against data/main_queries.jsonl   # or query_intake for raw lists
python -m src.gold_resolver --queries data/synth_gen.jsonl --concurrency 4
python -m src.rubric_gen    --queries data/synth_gen.jsonl --concurrency 4
python -m src.rubric_review --queries data/synth_gen.jsonl --concurrency 4 --yes
cat data/synth_gen.jsonl >> data/main_queries.jsonl
```

### 3 · Evaluate on your own set & read the report

Two independent lanes over the same set: **SERP-level pairwise** (`run_eval`) and **end-to-end agent**
(`agent_eval`).

```bash
RUN=results/run_$(date +%Y%m%d)      # reused across the pairwise-lane commands below

# Pairwise lane: fetch every backend, then a blind position-swapped judge
# (add --skip-judge first to smoke-test backend field mappings without spending judge tokens)
python -m src.run_eval --queries data/main_queries.jsonl \
    --ours octen --competitors exa brave --k 10 --out "$RUN"

# Weakness map: win-rate + Wilson CI + slice + dimension gaps (add --baseline for a regression diff)
python -m src.report --run "$RUN" --queries data/main_queries.jsonl

# Attribute failures, then assemble a shareable Markdown report
python -m src.triage --run "$RUN" --probe --concurrency 8
python -m scripts.gen_report --run "$RUN" --queries data/main_queries.jsonl --cases 5

# Agent lane (end-to-end; defaults to --scaffold agent on self-built sets)
python -m src.agent_eval --queries data/main_queries.jsonl --backends octen exa \
    --k 8 --max-searches 3 --out results/agent_$(date +%Y%m%d) --limit 10
```

## License

[MIT](LICENSE) © 2026 Octen.
