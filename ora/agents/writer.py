"""Writer agent node for LangGraph."""
from typing import Any, Optional
from langchain_core.runnables import RunnableConfig
from ora.state import ResearchState
from ora.prompts import WRITER_PROMPT
from ora.config import load_config, get_researcher_model, get_llm
from ora.progress import emit_progress


def _format_findings_for_prompt(findings: list) -> str:
    """Format findings list into a string for the writer prompt."""
    if not findings:
        return "No findings available."
    lines = []
    for i, f in enumerate(findings, 1):
        if hasattr(f, 'claim'):
            claim = f.claim
            confidence = f.confidence
            sources = ", ".join(f.supporting_sources[:3]) if f.supporting_sources else "no sources"
        elif isinstance(f, dict):
            claim = f.get("claim", "")
            confidence = f.get("confidence", "Moderate")
            sources = ", ".join(f.get("supporting_sources", [])[:3])
        else:
            continue
        lines.append(f"{i}. [{confidence}] {claim}")
        lines.append(f"   Sources: {sources}")
        lines.append("")
    return "\n".join(lines)


def writer_node(
    state: ResearchState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """Writer LangGraph node. Synthesizes findings into a structured report."""
    settings = load_config()
    model_name = get_researcher_model(settings)

    llm = get_llm(model_name, temperature=0.3)

    findings_raw = state.get("findings", [])
    finding_label = "finding" if len(findings_raw) == 1 else "findings"
    emit_progress(
        config,
        f"Writer: synthesizing report from {len(findings_raw)} {finding_label}",
        kind="write",
    )
    findings_text = _format_findings_for_prompt(findings_raw)

    prompt_text = WRITER_PROMPT.format(
        query=state.get("query", ""),
        intensity=state.get("intensity", 2),
        findings=findings_text,
    )

    try:
        response = llm.invoke(prompt_text)
    except Exception as e:
        emit_progress(config, f"Writer: LLM call failed: {e}", kind="error")
        raise
    draft_report = response.content if hasattr(response, 'content') else str(response)

    emit_progress(config, f"Writer: draft generated, {len(draft_report)} chars", kind="success")

    return {
        "draft_report": draft_report,
        "messages": [draft_report],
    }
