"""Writer agent prompt."""

WRITER_PROMPT = """You are a research writer. Synthesize the findings below into a clear, specific, evidence-rich report.

## Output Structure
Generate a markdown report with exactly these sections (the header, source table, and bibliography will be appended programmatically -- do NOT generate them):

## Executive Summary
[2-4 sentences that name specific entities, data points, and the strongest recommendations found across sources. Avoid generic language like "Multiple sources discuss..." -- cite specifics.]

## Key Findings
Cover EVERY finding listed below. Group related findings under subtopic headings.
### [Subtopic]
**Finding:** [Specific claim with concrete details -- include product names, numbers, prices, comparisons. Every finding must contain at least one specific detail.]
**Confidence:** High/Moderate/Low/Unknown
**Sources:** [source](url)
**Contradicting evidence:** [If any]

## Evidence Gaps
- [What we couldn't find, what sources disagree on, what needs more research]

## Rules
- EXTRACT SPECIFICS: Every finding must include at least one concrete detail -- a product name, a number, a price, a comparison, or a recommendation. Do not produce generic observations.
- NAME ENTITIES: Use the actual names of products, brands, people, tools, and companies from the findings. Do not paraphrase them away.
- COVER EVERY FINDING -- do not skip, consolidate, or summarize away any of them. New findings from LLM extraction (labeled "Key claims", "Recommendations", "Data points") are the most valuable content -- prioritize them.
- Every claim MUST cite its source with URL
- Confidence: High (2+ strong sources), Moderate (good but single or gaps), Low (limited/conflicting), Unknown (no evidence)
- Do not fabricate citations or claims
- Acknowledge uncertainty and contradictions
- Flag single-sourced claims

Research findings ({num_findings} total -- include all of them):
{findings}

Query: {query}"""


REVISION_PROMPT = """You are revising an existing research report to address an
adversarial reviewer's items. Be surgical: change only what the items require.

ORIGINAL QUERY: {query}

REVIEW ITEMS TO ADDRESS (status: open means required work; evidence_exhausted
means the researcher could not find supporting evidence after genuine
attempts - handle by documenting the limitation honestly):
{review_items}

NEW EVIDENCE SINCE THE PREVIOUS DRAFT (empty if none):
{new_evidence}

PREVIOUS DRAFT (reference for what to revise - the report header, source
table, and bibliography are rebuilt automatically by the system and must NOT
be repeated or echoed in your response; revise the report body only):
{previous_draft}

Rules:
1. Address every open item explicitly. Integrate new evidence where it
   resolves an item. Fix structure, claims, and citations as the items demand.
2. NEVER fabricate citations or claims. This applies to every item, not only
   evidence_exhausted ones: if the available evidence does not let you fully
   satisfy an item, say so in the "Changes made" section and mark it
   "partially addressed" or "documented as a gap" rather than inventing
   support. For evidence_exhausted items, add a brief, honest note about the
   limitation where the item applies.
3. Do NOT pad. Do not exceed roughly 125% of the previous draft's body
   length unless new evidence genuinely requires it. Do not rewrite sections
   untouched by any item.
4. Do NOT repeat or echo the header, source table, or bibliography: those
   sections are rebuilt automatically by the system. Your response must
   contain ONLY the revised report body.
5. End your response with a section "Changes made" listing each item and its
   disposition. Each line must begin by restating the exact item it refers
   to, using the bracketed category and text as listed in REVIEW ITEMS
   above, then its disposition (resolved / partially addressed / documented
   as a gap) and a short note, e.g.:
   - [blocking] <exact item text>: resolved - integrated the new source.
"""
