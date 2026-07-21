# 开源基准评测报告 — simpleqa_full_20260714

- 运行目录:`results/simpleqa_full_20260714` ｜ backends:exa, octen, parallel, tavily
- 记录:17304 episode(错误 0;已按 (qid,backend) 去重取最新)

## SimpleQA（full，4326 题）
> 官方指标:correct-rate=正确率;CGA=作答中正确率;**F1=二者调和均值(主排名指标)**。

| 排名 | backend | **F1** | 正确率 | CGA | 正确/错误/未答 | 搜索次数/改写率 | 接口耗时P50/P90(ms) |
|--:|---|--:|--:|--:|:--|:--|--:|
| 1 | octen | **96.5%** | 95.2% | 97.8% | 4120/94/112 | 1.0 / 0% | 77 / 97 |
| 2 | exa | **92.0%** | 89.2% | 95.0% | 3859/202/265 | 1.0 / 0% | 277 / 338 |
| 3 | parallel | **91.3%** | 88.6% | 94.1% | 3831/239/256 | 1.0 / 0% | — |
| 4 | tavily | **79.5%** | 69.9% | 92.2% | 3022/256/1048 | 1.0 / 0% | 130 / 200 |

## 附录
- SimpleQA:openai/simple-evals A/B/C 协议;F1 = 调和均值(correct-rate, CGA)。
- FreshQA:freshllms/freshqa FreshEval;多答案 ' | ' 拼接;strict 模式不容忍幻觉/过期。
- 搜索次数/改写率 = agent 端到端行为(所有 backend 用同一 agent+预算)。
- 接口耗时 = 后端接口返回的服务端耗时 P50/P90(reported_latency_ms);留空(—)= 该接口未返回耗时(如 parallel-turbo)或该 run 早于耗时采集。仅记录接口返回耗时,不回退端到端。