"""Researcher agent node for LangGraph."""
import re
from typing import Any
from langchain_core.runnables import RunnableConfig
from ora.state import ResearchState, Source, Finding
from ora.prompts import RESEARCHER_PROMPT
from ora.config import load_config, get_researcher_model, get_llm


def generate_search_queries(query: str, intensity: int) -> list[str]:
    """Generate search queries based on intensity level."""
    angles = {
        1: ["{query}"],
        2: ["{query}", "{query} latest", 'opposing view on {query}'],
        3: [
            "{query}",
            "{query} latest research 2025 2026",
            'critique of {query}',
            '"{query}" expert analysis',
            '{query} vs alternatives',
            '{query} best practices',
            'problems with {query}',
        ],
    }
    templates = angles.get(intensity, angles[2])
    return [t.format(query=query) for t in templates]


async def researcher_node(
    state: ResearchState, config: RunnableConfig = None
) -> dict[str, Any]:
    """Researcher: LLM generates queries, then tools execute searches/scrapes."""
    settings = load_config()
    model_name = get_researcher_model(settings)
    llm = get_llm(model_name, temperature=0)

    query = state.get("query", "")
    intensity = state.get("intensity", 2)
    query_count = {1: 1, 2: 3, 3: 5}.get(intensity, 3)

    # Generate queries via LLM
    response = llm.invoke(RESEARCHER_PROMPT + f"\n\nTopic: {query}")
    content = response.content if hasattr(response, 'content') else str(response)
    llm_queries = [q.strip().lstrip("- *1234567890. ") for q in content.split("\n") if q.strip() and len(q.strip()) > 10]
    template_queries = generate_search_queries(query, intensity)
    seen = set()
    all_queries = []
    for q in llm_queries + template_queries:
        ql = q.lower().strip()
        if ql not in seen:
            seen.add(ql)
            all_queries.append(q)
    queries = all_queries[:query_count]

    # Execute searches and scrapes
    from ora.tools.search import web_search
    from ora.tools.scrape import scrape_page
    from ora.tools.evaluate import evaluate_source

    sources, findings = [], []

    for q in queries:
        r = web_search.invoke({"query": q})
        urls = re.findall(r'https?://[^\s<>"\')\]]+', r)
        for url in urls[:1]:
            c = scrape_page.invoke({"url": url})
            if c and "error" not in c[:50].lower():
                source = evaluate_source(url=url, content=c, source_type="unknown")
                sources.append(source)
                findings.append(Finding(claim=c[:500], supporting_sources=[url]))

    # Fallback
    if not findings:
        findings.append(Finding(
            claim=f"No results for '{query}'. Check Firecrawl configuration.",
            confidence="Unknown",
        ))

    return {
        "search_queries": queries,
        "sources": sources,
        "findings": findings,
        "messages": [f"Searched {len(queries)} queries → {len(sources)} sources, {len(findings)} findings."],
    }
