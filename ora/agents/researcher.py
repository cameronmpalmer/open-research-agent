"""Researcher agent node for LangGraph."""
import re
from typing import Any
from langchain_core.runnables import RunnableConfig
from ora.state import ResearchState, Source, Finding
from ora.config import load_config


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
    """Researcher: template-based queries, programmatic search/scrape."""
    settings = load_config()

    query = state.get("query", "")
    intensity = state.get("intensity", 2)
    query_count = {1: 1, 2: 3, 3: 5}.get(intensity, 3)

    queries = generate_search_queries(query, intensity)[:query_count]

    from ora.tools.search import web_search
    from ora.tools.scrape import scrape_page
    from ora.tools.evaluate import evaluate_source

    sources, findings = [], []
    log: list[str] = []

    for q in queries:
        log.append(f"Search: {q}")
        r = web_search.invoke({"query": q})
        log.append(f"  Result: {len(r)} chars")

        # Debug: show search result format
        if r:
            log.append(f"  Preview: {r[:200]}")

        urls = re.findall(r'https?://[^\s<>"\')\]]+', r)
        log.append(f"  URLs found: {len(urls)}")
        if urls:
            log.append(f"  First URL: {urls[0][:80]}")

        for url in urls[:1]:
            c = scrape_page.invoke({"url": url})
            log.append(f"  Scraped: {len(c)} chars from {url[:60]}")
            if c and "error" not in c[:50].lower() and "Error" not in c[:50]:
                source = evaluate_source(url=url, content=c, source_type="unknown")
                sources.append(source)
                findings.append(Finding(claim=c[:500], supporting_sources=[url]))

    if not findings:
        results_text = web_search.invoke({"query": query})
        findings.append(Finding(
            claim=f"No scraped content found. Raw search: {results_text[:300]}",
            confidence="Unknown",
        ))

    return {
        "search_queries": queries,
        "sources": sources,
        "findings": findings,
        "messages": ["\n".join(log)],
    }
