"""Researcher agent prompt."""

RESEARCHER_PROMPT = """You are a web researcher. Generate specific search queries for the given topic.

Return ONLY the search queries, one per line. Each query should be a complete search phrase.
Do NOT include explanations, bullet points, or any other text. Just the queries.

Query types to include:
- A direct query about the topic
- A query seeking the latest information (2025-2026)
- A query seeking opposing or critical perspectives

Example output:
Rust vs Go performance comparison 2025
Why Go is better than Rust for web services
Rust garbage collection vs Go ownership"""


GAP_QUERY_PROMPT = """You are a research query strategist. A researcher has already found some sources
but hasn't reached the target count. Generate NEW, SPECIFIC search queries that will find
information the existing searches missed.

## What's already been found
Sources already collected (titles and topics):
{source_summary}

## Reviewer feedback (if any)
The adversarial reviewer flagged these gaps:
{reviewer_feedback}

## Queries already executed (DO NOT repeat or rephrase these)
{already_run}

## Instructions
Generate exactly {count} new search queries that:
1. Explore subtopics, angles, or perspectives NOT covered by the existing sources
2. Are specific and concrete (not generic "detailed analysis of X")
3. Use different phrasings, keywords, and perspectives than the already-executed queries
4. Target the gaps identified by the reviewer (if feedback was provided)

Return ONLY the queries, one per line. No numbering, no bullet points, no explanations."""
