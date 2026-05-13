# ORA Intensity Levels 4 and 5 Design

## Overview

ORA currently supports levels 1-3 with a linear scale of queries and scrapes. Levels 4 (Deep) and 5 (Exhaustive) are defined in the type system but fall back to Level 3. This design implements them, aligned with the comprehensive-research skill's intensity scale.

## Goals

- Implement research behavior for levels 4 and 5.
- Scale search queries, scraping depth, content length, and research rounds per level.
- Enforce minimum source counts per level with fallback rounds and interim labeling.
- Enable the adversarial reviewer gate for levels 4-5.

## Level Scaling

Levels 1-3 unchanged. Levels 1-3 remain without reviewer gate.

| Dimension | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| Min sources | 3 | 8 | 15 | 50 | 100 |
| Max rounds | 1 | 1+gap fill | 2 | 3 | 4 |
| Queries/round | 1 | 3 | 5 | varies | varies |
| URLs attempted/query | 3 | 3 | 5 | 8 | 10 |
| Successful scrapes/query (cap) | 2 | 2 | 3 | 4 | 5 |
| Content max chars | 8000 | 8000 | 10000 | 12000 | 16000 |
| Reviewer gate | off | off | off | on | on |
| Gap-targeted queries per shortfall | 0 | 2 | 3 | 5 | 7 |

Levels 4 and 5 use the full query set from `generate_search_queries` for round 1 (7 queries for level 3's template, topped up with additional angles for L4-L5). Subsequent rounds use gap-targeted queries.

## Per-Round Logic

After each round:
1. Check `len(sources) >= minimum`.
2. If yes → proceed to writer.
3. If no and rounds remain → generate gap-targeted queries and run another round.
4. If no and max rounds exhausted → label report **Interim Findings** with a header noting the shortfall.

## Interim vs Final

- If the minimum source count is met → **Final Report**.
- If the minimum is not met after max rounds → **Interim Findings**. The report header is replaced with `**Interim Report** -- minimum source threshold (X) not met after Y rounds (Z sources found)`. The shortfall is also noted in Evidence Gaps.
- If the reviewer gate returns REVISIONS REQUIRED and issues cannot be resolved → **Interim Findings**.

## Reviewer Gate (L4-L5)

The adversarial reviewer (currently disabled) is re-enabled for levels 4-5. It reviews the draft report and returns PASS or REVISIONS REQUIRED. If revisions are required, the researcher re-runs targeted queries to address reviewer concerns, and the writer produces an updated draft. The review loop runs up to 3 times.

## Gap-Targeted Queries

When a round fails to hit the source minimum, gap-targeted queries are generated using templates like:

- "{query} detailed analysis"
- "{query} expert review"
- "key aspects of {query}"
- "recent developments in {query}"
- "opposing view on {query}"

For L5 specifically, additional angles may be pulled from the reviewer's feedback.

## Fallbacks

- If Firecrawl returns fewer URLs than needed, the researcher logs the shortfall and continues with what's available.
- If all scrape attempts fail, the raw search result text is used as a fallback finding (existing behavior).
- Domain skip-list filtering may reduce available URLs; the researcher walks the full URL list per query.

## Testing

- Extend `generate_search_queries` tests for L4 and L5 query sets.
- Add tests for gap-targeted query generation.
- Add tests for per-round source-count checking and round continuation.
- Add tests for interim vs final labeling.
- Update CLI tests for the enabled reviewer gate at L4-L5.
- Smoke-test L4 and L5 end-to-end with the fake harness.
