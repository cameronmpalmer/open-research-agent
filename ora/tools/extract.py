"""Per-source LLM extraction and evaluation."""
import json
from typing import Optional
from langchain_core.runnables import RunnableConfig
from ora.state import Source, SourceExtraction
from ora.prompts import EXTRACTOR_PROMPT
from ora.config import load_config, get_llm
from ora.tools.evaluate import evaluate_source as evaluate_heuristic


def extract_and_evaluate(
    url: str,
    title: str = "",
    content: str = "",
    source_type: str = "unknown",
    query: str = "",
    config: Optional[RunnableConfig] = None,
) -> tuple[Source, SourceExtraction]:
    """Extract claims and evaluate reliability from a source using an LLM.

    Reads the full scraped content and produces both a structured
    extraction (key claims, entities, data, recommendations) and a
    reliability assessment. One LLM call does both jobs.

    Args:
        url: Source URL.
        title: Page title from search results.
        content: Full scraped page content.
        source_type: Source classification (academic_paper, news, etc.).
        query: The original research query for context.
        config: Optional RunnableConfig for progress events.

    Returns:
        A (Source, SourceExtraction) tuple. The Source contains CRAAP
        dimensions derived from the LLM's reliability rating, and the
        SourceExtraction contains all extracted content for the writer.
    """
    from ora.progress import emit_progress

    # Truncate content to a reasonable prompt budget.
    content_budget = content[:6000]

    settings = load_config()
    model_name = settings.models.researcher or settings.models.default

    prompt_text = EXTRACTOR_PROMPT.format(
        query=query,
        title=title,
        url=url,
        source_type=source_type,
        content=content_budget,
    )

    try:
        llm = get_llm(model_name, temperature=0.1)
        response = llm.invoke(prompt_text)
        text = response.content if hasattr(response, 'content') else str(response)
    except Exception:
        emit_progress(
            config,
            f"Extractor: LLM call failed for {url[:60]}, falling back to heuristic",
            kind="warning",
        )
        heuristic_source = evaluate_heuristic(
            url=url, title=title, content=content, source_type=source_type
        )
        return heuristic_source, SourceExtraction(
            summary=content[:500] if content else "",
        )

    # Parse the JSON output. Handle markdown code fences if present.
    try:
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            json_str = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            json_str = text[start:end].strip()
        else:
            json_str = text.strip()

        data = json.loads(json_str)

        extraction = SourceExtraction(
            summary=data.get("summary", ""),
            key_claims=data.get("key_claims", []),
            recommendations=data.get("recommendations", []),
            named_entities=data.get("named_entities", []),
            data_points=data.get("data_points", []),
            comparisons=data.get("comparisons", []),
            criticisms=data.get("criticisms", []),
            source_reliability=data.get("source_reliability", "Medium"),
            reliability_rationale=data.get("reliability_rationale", ""),
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        emit_progress(
            config,
            f"Extractor: JSON parse failed for {url[:60]}, using heuristic",
            kind="warning",
        )
        heuristic_source = evaluate_heuristic(
            url=url, title=title, content=content, source_type=source_type
        )
        return heuristic_source, SourceExtraction(
            summary=text[:1000] if text else (content[:500] if content else ""),
        )

    # Build Source from the extraction's reliability rating.
    rel = extraction.source_reliability
    if rel == "High":
        currency, relevance, authority, accuracy, purpose = 4, 5, 5, 5, 5
    elif rel == "Medium":
        currency, relevance, authority, accuracy, purpose = 3, 4, 3, 3, 3
    else:
        currency, relevance, authority, accuracy, purpose = 2, 3, 2, 2, 2

    source = Source(
        url=url,
        title=title,
        source_type=source_type,  # type: ignore[arg-type]
        currency=currency,
        relevance=relevance,
        authority=authority,
        accuracy=accuracy,
        purpose=purpose,
        overall_reliability=rel,  # type: ignore[arg-type]
        notes=extraction.reliability_rationale,
    )

    return source, extraction
