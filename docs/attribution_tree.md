# Failure Attribution Decision Tree (v1)

For every failure case (lost a pairwise / missed an anchor / wrong QA answer), diagnose in the order below;
the first node that holds is the case's failure mode.

```
failure case
│
├─ [L0] Service-layer failure? (timeout / error / empty results)
│        → mode: SERVICE_{TIMEOUT|ERROR|EMPTY}
│
├─ [L1] Is the correct page in the index?
│   │    Verify: --probe (URL-targeted retrieval against our own backend); or site:domain + exact-phrase query
│   ├─ never crawled          → mode: INDEX_NEVER_CRAWLED   (discovery / seeding / crawl-scheduling problem)
│   ├─ crawled but not stored → mode: INDEX_DROPPED         (parse failure / dedup false kill / quality-filter false kill)
│   ├─ stored but stale       → mode: INDEX_STALE           (recrawl-frequency problem, common for realtime queries)
│   └─ in the index ↓ continue
│
├─ [L2] In the index but missing from the recall candidate set?
│   │    Verify: offline repro — run the query through the production recall chain, check whether the gold doc is in the top-N candidates
│   ├─ query rewrite drifted the intent                     → mode: RECALL_REWRITE_DRIFT
│   ├─ tokenization problem (common for Chinese / mixed language) → mode: RECALL_TOKENIZATION
│   ├─ embedding missed the semantics (long-tail entities / proper nouns) → mode: RECALL_EMBEDDING_MISS
│   └─ recalled ↓ continue
│
├─ [L3] Recalled but failed to rank up?
│   │    Verify: dump the doc's ranking feature scores within the candidate set and compare with the docs ranked above it
│   ├─ authority signal missing/wrong                       → mode: RANK_AUTHORITY_SIGNAL
│   ├─ freshness weight inappropriate                       → mode: RANK_FRESHNESS_WEIGHT
│   ├─ squeezed out by low-quality pages (spam ranked above) → mode: RANK_SPAM_ABOVE
│   └─ ranked up ↓ continue
│
└─ [L4] Ranked up but presented poorly?
    ├─ snippet truncated / boilerplate / mojibake → mode: SNIPPET_EXTRACTION
    └─ title wrong / missing                      → mode: SNIPPET_TITLE
```

## Usage discipline

1. **Each failure case gets exactly one primary cause** (the most upstream one). When the index has no coverage, don't bother looking at ranking.
2. After each eval batch, compute each mode's share → that ranking IS the iteration priority order.
3. After a fix ships, rerun the same batch of failure cases; verify that mode's share drops and no new mode pops up.
4. L1 automation runs via `triage --probe`: URL-targeted queries against our own backend; misses
   are confidence=low because a probe cannot separate "not indexed" from "catastrophically
   unranked". L2/L3 confirmation needs engine-internal signals (recall candidates + ranking
   feature scores), so those remain manual tickets.
5. If the judge calls a loss but human review finds the judge wrong → mode: JUDGE_ERROR; feed it back into the rubric calibration set.

## Failure mode taxonomy summary

| mode | layer | owning module | typical fix |
|---|---|---|---|
| SERVICE_* | L0 | service / gateway | capacity, timeout config, retries |
| INDEX_NEVER_CRAWLED | L1 | crawl scheduling | seed expansion, discovery strategy, vertical-targeted crawling |
| INDEX_DROPPED | L1 | parsing / filtering | parser fixes, quality-filter threshold review |
| INDEX_STALE | L1 | recrawl strategy | tune recrawl frequency per site / vertical |
| RECALL_REWRITE_DRIFT | L2 | query understanding | feed rewrite-model bad cases back into training |
| RECALL_TOKENIZATION | L2 | tokenization | dictionaries / tokenizer, dedicated mixed Chinese-English work |
| RECALL_EMBEDDING_MISS | L2 | vector recall | embedding fine-tuning, hybrid recall with BM25 |
| RANK_AUTHORITY_SIGNAL | L3 | ranking features | site authority scoring system |
| RANK_FRESHNESS_WEIGHT | L3 | ranking model | couple freshness features with query freshness intent |
| RANK_SPAM_ABOVE | L3 | anti-spam | spam classifier |
| SNIPPET_EXTRACTION | L4 | content extraction | extractor (can hook up a WCXB-style extraction eval) |
| SNIPPET_TITLE | L4 | content extraction | title extraction rules |
| JUDGE_ERROR | — | the eval system itself | rubric revision, calibration-set expansion |
