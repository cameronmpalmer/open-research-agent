"""Researcher agent node for LangGraph."""
from typing import Any
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from ora.state import ResearchState, Source, Finding
from ora.prompts import RESEARCHER_PROMPT
from ora.tools.search import web_search
from ora.tools.scrape import scrape_page
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
    """Researcher LangGraph node. Searches, scrapes, and evaluates sources."""
    settings = load_config()
    model_name = get_researcher_model(settings)

    llm = get_llm(model_name, temperature=0)
    tools = [web_search, scrape_page]

    intensity = state.get("intensity", 2)
    query = state.get("query", "")

    system_prompt = RESEARCHER_PROMPT.format(
        query=query,
        intensity=intensity,
    ) if "{query}" in RESEARCHER_PROMPT else RESEARCHER_PROMPT

    query_count = {1: 2, 2: 4, 3: 7}.get(intensity, 4)
    queries = generate_search_queries(query, intensity)

    graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )

    inputs = {
        "messages": [
            HumanMessage(
                content=(
                    f"Research the following query. Generate approximately {query_count} "
                    f"distinct search queries and run them. "
                    f"Search angles: {', '.join(queries[:query_count])}"
                ),
            )
        ]
    }

    result = await graph.ainvoke(inputs)
    messages = result.get("messages", [])
    output = messages[-1].content if messages else ""

    return {
        "search_queries": queries[:query_count],
        "sources": _extract_sources_from_output(output),
        "findings": _extract_findings_from_output(output),
        "messages": [output],
    }


def _extract_sources_from_output(output: str) -> list[Source]:
    """Parse sources from agent output."""
    import re
    from ora.tools.evaluate import evaluate_source

    url_pattern = r'https?://[^\s<>"\')]+'
    urls = list(set(re.findall(url_pattern, output)))
    sources = []
    for url in urls[:20]:
        source = evaluate_source(url=url, content=output[:500], source_type="unknown")
        sources.append(source)
    return sources


def _extract_findings_from_output(output: str) -> list[Finding]:
    """Parse findings from agent output."""
    findings = []
    lines = output.split("\n")
    current = ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("- ", "* ", "1.", "2.", "3.")) and len(stripped) > 10:
            if current:
                findings.append(Finding(claim=current[:500]))
            current = stripped.lstrip("- *1234567890. ")
        elif current:
            current += " " + stripped
    if current and len(current) > 10:
        findings.append(Finding(claim=current[:500]))
    if not findings:
        findings.append(Finding(claim=output[:500]))
    return findings
