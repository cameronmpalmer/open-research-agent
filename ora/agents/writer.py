"""Writer agent node for LangGraph."""
from datetime import date
from typing import Any, Optional
from langchain_core.runnables import RunnableConfig
from ora.state import ResearchState
from ora.prompts import WRITER_PROMPT
from ora.config import load_config, get_researcher_model, get_llm
from ora.progress import emit_progress


def _format_findings_for_prompt(findings: list) -> str:
    """Format findings list into a string for the writer prompt.

    When extraction data is available (intensity 3+), includes the
    structured extraction fields. Otherwise, formats the basic claim.
    """
    if not findings:
        return "No findings available."

    lines = []
    for i, f in enumerate(findings, 1):
        # Handle both Pydantic model and dict (LangGraph serialization)
        if hasattr(f, 'claim'):
            claim = f.claim
            confidence = f.confidence
            sources = ", ".join(f.supporting_sources[:3]) if f.supporting_sources else "no sources"
            extraction = getattr(f, 'extraction', None)
        elif isinstance(f, dict):
            claim = f.get("claim", "")
            confidence = f.get("confidence", "Moderate")
            sources = ", ".join(f.get("supporting_sources", [])[:3])
            extraction = f.get("extraction")
        else:
            continue

        lines.append(f"{i}. [{confidence}] {claim}")
        lines.append(f"   Sources: {sources}")

        # If we have LLM-extracted details, include them.
        if extraction is not None:
            if hasattr(extraction, 'key_claims') and extraction.key_claims:
                lines.append("   Key claims:")
                for c in extraction.key_claims[:5]:  # cap at 5
                    lines.append(f"     - {c}")
            if hasattr(extraction, 'recommendations') and extraction.recommendations:
                lines.append("   Recommendations:")
                for r in extraction.recommendations[:5]:
                    lines.append(f"     - {r}")
            if hasattr(extraction, 'data_points') and extraction.data_points:
                lines.append("   Data points:")
                for d in extraction.data_points[:5]:
                    lines.append(f"     - {d}")
            if hasattr(extraction, 'named_entities') and extraction.named_entities:
                lines.append(f"   Named: {', '.join(extraction.named_entities[:10])}")
            if hasattr(extraction, 'comparisons') and extraction.comparisons:
                lines.append("   Comparisons:")
                for cmp in extraction.comparisons[:3]:
                    lines.append(f"     - {cmp}")

        lines.append("")
    return "\n".join(lines)


def _build_header(query: str, intensity: int, num_sources: int) -> str:
    """Build the report header with programmatic source count."""
    today = date.today().strftime("%Y-%m-%d")
    return (
        f"# Research: {query}\n"
        f"**Intensity:** Level {intensity} | **Sources:** {num_sources} | **Date:** {today}\n"
    )


def _build_source_table(sources: list) -> str:
    """Build a markdown source table from the sources list."""
    if not sources:
        return ""
    rows = []
    for i, s in enumerate(sources, 1):
        title = getattr(s, 'title', '') or ''
        url = getattr(s, 'url', '') or ''
        source_type = getattr(s, 'source_type', 'unknown') or 'unknown'
        reliability = getattr(s, 'overall_reliability', 'Unknown') or 'Unknown'
        rows.append(f"| {i} | {title} | {url} | {source_type} | {reliability} |")
    header = "| # | Title | URL | Type | Reliability |\n|---|-------|-----|------|-------------|\n"
    return "## Source Table\n" + header + "\n".join(rows) + "\n"


def _build_bibliography(sources: list) -> str:
    """Build a numbered bibliography from the sources list."""
    if not sources:
        return ""
    lines = ["## Bibliography"]
    for i, s in enumerate(sources, 1):
        title = getattr(s, 'title', '') or 'Untitled'
        url = getattr(s, 'url', '') or ''
        date_str = getattr(s, 'publication_date', '') or ''

        line = f"{i}. {title}"
        if date_str:
            line += f", {date_str}"
        line += f". [{url}]({url})"
        lines.append(line)
    return "\n".join(lines) + "\n"


def writer_node(
    state: ResearchState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """Writer LangGraph node. Synthesizes findings into a structured report.

    The header (source count), source table, and bibliography are generated
    programmatically to ensure completeness. The LLM handles the Executive
    Summary, Key Findings, and Evidence Gaps sections.
    """
    settings = load_config()
    model_name = get_researcher_model(settings)

    llm = get_llm(model_name, temperature=0.3)

    findings_raw = state.get("findings", [])
    sources_raw = state.get("sources", [])
    query = state.get("query", "")
    intensity = state.get("intensity", 2)

    num_findings = len(findings_raw)
    finding_label = "finding" if num_findings == 1 else "findings"
    emit_progress(
        config,
        f"Writer: synthesizing report from {num_findings} {finding_label}",
        kind="write",
    )
    findings_text = _format_findings_for_prompt(findings_raw)

    prompt_text = WRITER_PROMPT.format(
        query=query,
        findings=findings_text,
        num_findings=num_findings,
    )

    try:
        response = llm.invoke(prompt_text)
    except Exception as e:
        emit_progress(config, f"Writer: LLM call failed: {e}", kind="error")
        raise
    llm_body = response.content if hasattr(response, 'content') else str(response)

    # Assemble the final report: header + LLM body + programmatic sections
    header = _build_header(query, intensity, len(sources_raw))
    source_table = _build_source_table(sources_raw)
    bibliography = _build_bibliography(sources_raw)

    draft_report = header + "\n" + llm_body + "\n" + source_table + "\n" + bibliography

    emit_progress(config, f"Writer: draft generated, {len(draft_report)} chars", kind="success")

    return {
        "draft_report": draft_report,
        "messages": [draft_report],
    }
