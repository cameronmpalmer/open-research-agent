"""Supervisor agent prompts."""

_SEARCH_QUERY_COUNTS = {1: 1, 2: 3, 3: 7, 4: 12, 5: 16}

def _search_query_count(intensity: int) -> int:
    """Return the number of search queries to generate for a given intensity."""
    return _SEARCH_QUERY_COUNTS.get(intensity, 3)

SUPERVISOR_PLAN_PROMPT = """You are a research planner. Your job is to create a research plan for the following query.

Query: {query}
Intensity level: {intensity} (1=Quick, 2=Standard, 3=Thorough, 4=Deep, 5=Exhaustive)

Create a research plan with:
1. Core question restated
2. Subtopics to investigate (3-5)
3. Search angles (direct, opposing, specific, recent)
4. Known gaps or assumptions

Output the plan in clear markdown.

At the very end of your response, after all other content, output the exact search queries
you would use to research this topic as a JSON array in a code fence. Use the format below
exactly (include the language tag):

```search_queries
["keyword query 1", "keyword query 2", ...]
```

Generate exactly {count} queries. Each query must be 3-7 targeted keywords suitable
for a web search engine (Google, Firecrawl). Focus on specific product names,
technologies, comparison angles, and key concepts. No full sentences, no questions,
no natural language -- just search-engine-optimized keyword strings."""

SUPERVISOR_REVISE_PROMPT = """You are a research planner. A user has reviewed your research plan and provided feedback. Revise the plan based on their feedback while preserving the overall structure.

Original query: {query}
Intensity level: {intensity} (1=Quick, 2=Standard, 3=Thorough, 4=Deep, 5=Exhaustive)

Current plan:
{plan}

User feedback:
{feedback}

Output the revised plan in clear markdown.

At the very end of your response, after all other content, output the exact search queries
you would use to research this topic as a JSON array in a code fence. Use the format below
exactly (include the language tag):

```search_queries
["keyword query 1", "keyword query 2", ...]
```

Generate exactly {count} queries. Each query must be 3-7 targeted keywords suitable
for a web search engine (Google, Firecrawl). Focus on specific product names,
technologies, comparison angles, and key concepts. No full sentences, no questions,
no natural language -- just search-engine-optimized keyword strings."""

SUPERVISOR_ROUTE_PROMPT = """You are a research supervisor. Based on the current state, decide which agent should work next.

Current state:
- Plan approved: {plan_approved}
- Sources found: {source_count}
- Findings recorded: {finding_count}
- Draft exists: {has_draft}
- Review verdict: {review_verdict}
- Revision count: {revision_count} (max: 3)

Available agents: researcher, writer, reviewer, FINISH

Respond with ONLY the agent name."""
