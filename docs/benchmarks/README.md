# Open-benchmark results

Versioned record of the **SimpleQA** and **FreshQA** fast-tier benchmark runs (2026-07-14).

> **Setup**
> - Generator + judge model: `openrouter/minimax/minimax-m2.5`
> - Pipeline: **single-shot** — exactly **1 search per query**, no agent loop
>   (so "avg searches" = 1.0 and "rewrite rate" is N/A).
> - Fast-tier configs: `exa`=instant · `tavily`=ultra-fast · `parallel`=turbo · `octen`=default
> - Scored with the official graders in `src/grading.py` (SimpleQA A/B/C, FreshEval strict).
> - Server latency = API-returned server time (search-only probe over the full sets; P50/P90 ms),
>   measured per set. It is the time the backend's API reports, not the client round-trip;
>   `parallel-turbo` returns none → blank.
> - Runs: SimpleQA `20260714_021958_760895` (4326 q) · FreshQA `20260714_042106_196d52` (600 q).
> - Per-question records live under `results/` (gitignored); regenerate the tables below with
>   `scripts/benchmark_report`.

## Headline (correct-rate / accuracy)

| backend | SimpleQA (4326) | SimpleQA lat (ms) | FreshQA (600) | FreshQA lat (ms) |
|---|--:|--:|--:|--:|
| **octen** | **95.2%** (F1 96.5) | **77 / 97** | **55.8%** | **71 / 88** |
| exa-instant | 89.2% (F1 92.0) | 277 / 338 | 52.8% | 286 / 326 |
| parallel-turbo | 88.6% (F1 91.3) | — | 52.8% | — |
| tavily-ultrafast | 69.9% (F1 79.5) | 130 / 200 | 43.2% | 110 / 160 |

Full per-metric tables (F1/CGA/not-attempted; FreshQA fact_type & false-premise breakdowns):
- [`simpleqa_full_20260714.md`](simpleqa_full_20260714.md)
- [`freshqa_20260714.md`](freshqa_20260714.md)

## Reproduce in this repo
```bash
python -m scripts.load_benchmarks                       # build data/simpleqa_full.jsonl, freshqa.jsonl
python -m src.agent_eval --queries data/simpleqa_full.jsonl \
    --backends octen exa-instant parallel-turbo tavily-ultrafast --out results/simpleqa --yes
python -m scripts.benchmark_report --run results/simpleqa
```
(benchmark sets default to `minimax/minimax-m2.5` for generator + judge, matching the above.)
