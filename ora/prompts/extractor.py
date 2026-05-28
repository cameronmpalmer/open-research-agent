"""Per-source extraction and evaluation prompt."""

EXTRACTOR_PROMPT = """You are a research extraction engine. Read the full content of a source page and produce two things:

1. A structured extraction of everything this source says that is relevant to the research query
2. A reliability assessment of this source

## Extraction Instructions

Extract the following from the source content. Be specific and concrete -- quote or closely paraphrase. Do not summarize generically.

- **Summary:** 2-4 sentences capturing what this source says about the query. Include the source's main argument or recommendation.
- **Key Claims:** Specific factual claims the source makes. Each should be one sentence and cite where in the source it appears if quoting directly.
- **Recommendations:** If the source recommends specific products, approaches, or actions, list each one with the reasoning given.
- **Named Entities:** Products, brands, people, companies, tools, or other named things mentioned in relation to the query.
- **Data Points:** Any numbers, statistics, prices, percentages, dates, or quantitative claims. Include the value and what it measures.
- **Comparisons:** If the source compares two or more things (X vs Y, X is better than Y for Z), extract each comparison.
- **Criticisms:** Drawbacks, limitations, caveats, or opposing views the source mentions.

If a category has nothing to extract, use an empty list. Do not fabricate content.

## Reliability Assessment

- **Source Reliability:** High / Medium / Low based on the source's authority, evidence quality, potential bias, and specificity.
  - High: authoritative source (government, academic, established publication), well-sourced, specific and verifiable claims
  - Medium: reasonable source, some evidence but could be opinion-based, commercial site with generally reliable content
  - Low: clearly biased, unsourced, commercial affiliate content, forum/social media, generic content-mill writing
- **Reliability Rationale:** 1 sentence explaining why you assigned this rating.

## Output Format

Return ONLY a JSON object with no markdown fences, no extra text:

{{"summary": "...", "key_claims": ["...", "..."], "recommendations": ["...", "..."], "named_entities": ["...", "..."], "data_points": ["...", "..."], "comparisons": ["...", "..."], "criticisms": ["...", "..."], "source_reliability": "Medium", "reliability_rationale": "..."}}

Research query: {query}
Source title: {title}
Source URL: {url}
Source type: {source_type}

Source content:
{content}"""
