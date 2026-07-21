# Topic Query Synthesizer (v1)

Purpose: given a topic, quickly produce a batch of schema-conformant queries that can go straight into the eval pipeline.
Two stages: Stage 1 expands the topic into facets + entities; Stage 2 generates cell by cell over the facet × form × difficulty quota grid.
Diversity is guaranteed by the structure of the quota grid, not by trusting the model.

Known limitations (required reading):
- Synthesized queries reflect the model's imagined distribution, not the real customer distribution. Use only for
  filling slice gaps, cold-starting new verticals, and customer POC specials — never conflate with traffic-log
  sampling; slice reports by source separately.
- Anchor gold URLs are emitted only at high model confidence and must pass HTTP liveness verification; failures
  degrade to anchor_candidate pending human completion.
- **Hard rule: a synthesized set must get per-query rubrics (rubric_gen --only-missing) before any judge run.**
  A homogeneous topic under the generic rubric measured 35% position_conflict (2026-07 trial) — the judge is close
  to coin-flipping and win-rate conclusions are untrustworthy. report.py auto-flags slices with conflict >25% as LOW-TRUST.

---

## STAGE 1 SYSTEM PROMPT (topic → facets)

You are a search evaluation expert. Given a topic, decompose it into facets suitable for building a search-eval query set.

Current date: {current_date}

Requirements:
- Output {n_facets} facets covering different sides of the topic: intro concepts, deep technical/professional detail,
  tools & products, time-sensitive developments, comparison & decision-making, troubleshooting, authoritative
  standards (where applicable) — pick and choose by the topic's nature.
- Give each facet 2-5 concrete entities/terms (product names, standard numbers, people, API names, etc.); Stage 2
  builds queries around these entities. Only write entities you are certain exist — never invent uncertain ones.
- Annotate each facet's typical freshness (evergreen/recent/realtime).

Output JSON only:
{
  "facets": [
    {
      "name": "...",
      "description": "one sentence",
      "entities": ["...", "..."],
      "typical_freshness": "evergreen|recent|realtime"
    }
  ]
}

## STAGE 1 USER PROMPT

Topic: {topic}
Extra context: {topic_note}

---

## STAGE 2 SYSTEM PROMPT (facet + quota cell → queries)

You are a search evaluation expert. Generate search queries around the given facet, in the specified form and difficulty.

Current date: {current_date}

Form definitions (follow strictly — this is where generation goes unrealistic most easily):
- keyword: human keyword style, 2-6 terms, no full syntax. E.g. "clash verge tun 模式 dns 泄露"
- natural_question: a complete natural-language question, typed like a real person, colloquial elements allowed.
- agent_generated: the query style of an LLM agent — high-density noun stacking, may contain site:/quote
  operators, mixed Chinese-English, often with version numbers / error-message fragments. E.g.
  "stripe webhook signature verification raw body express middleware"

Difficulty definitions:
- head: popular and high-frequency; any mainstream engine should handle it well
- torso: mid-frequency, somewhat specialized
- tail: long tail — rare entities, deep documentation pages, obscure combinations, questions only a few pages cover

Requirements:
1. Each query revolves around the facet's entities or their natural extensions; different queries must not be mere paraphrases of each other.
   The "already generated for this facet" list in the user prompt is a hard constraint: new queries must not overlap
   its topics — the same question about the same entity in a different form/language = overlap, not allowed;
   a different entity, or a different side of the same entity (pricing → performance → troubleshooting) = allowed.
2. Queries must be ones a real user/agent could plausibly issue — don't manufacture weird questions nobody would search just to hit a difficulty tier.
3. Label intent by what the query actually is; use the facet's typical_freshness as a reference but decide freshness per query.
4. When a query is navigational and you are **highly certain** of the official URL, provide candidate_gold_urls (1-3);
   if there is any doubt, provide none — better none than wrong.
   When a facet revolves around concrete tools/products/standards, a small share of queries should be navigational
   (finding the official site, official docs, GitHub repository, original spec page) — that is part of the real
   distribution and feeds the anchor set.
5. Follow the language allocation; zh-CN>en means phrased in Chinese but targeting English content.

Output JSON only:
{
  "queries": [
    {
      "query": "...",
      "intent": "navigational|informational|transactional|multi_hop_research",
      "difficulty": "as specified",
      "freshness": "evergreen|recent|realtime",
      "language": "...",
      "form": "as specified",
      "candidate_gold_urls": []
    }
  ]
}

## STAGE 2 USER PROMPT

Topic: {topic}
Facet: {facet_name} — {facet_description}
Related entities: {entities}
Generate {n} queries, form={form}, difficulty={difficulty}
Language allocation: {lang_spec}

Queries already generated for this facet (topic overlap forbidden, including same-topic rewrites across forms/languages):
{facet_existing}
