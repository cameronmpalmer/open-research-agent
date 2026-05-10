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
    """Researcher node. LLM generates queries, then tools execute them programmatically."""
    settings = load_config()
    model_name = get_researcher_model(settings)
    llm = get_llm(model_name, temperature=0)

    query = state.get("query", "")
    intensity = state.get("intensity", 2)
    query_count = {1: 2, 2: 4, 3: 7}.get(intensity, 4)

    # Step 1: Generate search queries using LLM
    topic_prompt = RESEARCHER_PROMPT + f"\n\nTopic: {query}"
    response = llm.invoke(topic_prompt)
    content = response.content if hasattr(response, 'content') else str(response)

    llm_queries = [
        q.strip().lstrip("- *1234567890. ")
        for q in content.split("\n")
        if q.strip() and len(q.strip()) > 10
    ]
    all_queries = llm_queries[:query_count]

    # Step 2: Execute searches and scrapes programmatically
    from ora.tools.search import web_search
    from ora.tools.scrape import scrape_page
    from ora.tools.evaluate import evaluate_source

    sources: list[Source] = []
    findings: list[Finding] = []

    for q in all_queries:
        results = web_search.invoke({"query": q})
        results_str = results if isinstance(results, str) else str(results)

        urls = re.findall(r'https?://[^\s<>"\')\]]+', results_str)
        for url in urls[:3]:
            content = scrape_page.invoke({"url": url})
            content_str = content if isinstance(content, str) else str(content)
            source = evaluate_source(url=url, content=content_str, source_type="unknown")
            sources.append(source)

            sentences = [s.strip() for s in content_str.split(". ") if len(s.strip()) > 20]
            for sentence in sentences[:3]:
                findings.append(Finding(claim=sentence[:500], supporting_sources=[url]))

    if not findings:
        findings.append(Finding(
            claim=f"No results found for query: {query}",
            confidence="Unknown",
        ))

    return {
        "search_queries": all_queries,
        "sources": sources,
        "findings": findings,
        "messages": [f"Searched {len(all_queries)} queries, found {len(sources)} sources."],
    }
