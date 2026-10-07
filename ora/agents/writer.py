"""Writer agent node for LangGraph."""

from datetime import date
from typing import Any

from langchain_core.runnables import RunnableConfig

from ora.config import get_llm, get_writer_model, load_config
from ora.progress import emit_progress
from ora.prompts import REVISION_PROMPT, WRITER_PROMPT
from ora.state import ResearchState


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
        if hasattr(f, "claim"):
            claim = f.claim
            confidence = f.confidence
            sources = ", ".join(f.supporting_sources[:3]) if f.supporting_sources else "no sources"
            extraction = getattr(f, "extraction", None)
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
            # Handle both dict (LangGraph serialized) and Pydantic model.
            if isinstance(extraction, dict):
                kc = extraction.get("key_claims", [])
                recs = extraction.get("recommendations", [])
                dps = extraction.get("data_points", [])
                ents = extraction.get("named_entities", [])
                comps = extraction.get("comparisons", [])
            else:
                kc = getattr(extraction, "key_claims", [])
                recs = getattr(extraction, "recommendations", [])
                dps = getattr(extraction, "data_points", [])
                ents = getattr(extraction, "named_entities", [])
                comps = getattr(extraction, "comparisons", [])

            if kc:
                lines.append("   Key claims:")
                for c in kc[:5]:
                    lines.append(f"     - {c}")
            if recs:
                lines.append("   Recommendations:")
                for r in recs[:5]:
                    lines.append(f"     - {r}")
            if dps:
                lines.append("   Data points:")
                for d in dps[:5]:
                    lines.append(f"     - {d}")
            if ents:
                lines.append(f"   Named: {', '.join(ents[:10])}")
            if comps:
                lines.append("   Comparisons:")
                for cmp in comps[:3]:
                    lines.append(f"     - {cmp}")

        lines.append("")
    return "\n".join(lines)


# Programmatic section markers emitted by _build_source_table and
# _build_bibliography. The LLM body never legitimately contains them
# (WRITER_PROMPT/REVISION_PROMPT forbid generating them), so the first
# occurrence in a full report marks the start of the programmatic tail.
_SOURCE_TABLE_MARKER = "## Source Table"
_BIBLIOGRAPHY_MARKER = "## Bibliography"
_CHANGES_MADE_MARKER = "Changes made"

# Sections that are (re)built programmatically rather than written by the
# LLM. In revision mode the previous draft is passed to the LLM as revision
# context (body plus programmatic tail, so items about specific citations
# can be addressed) but any programmatic sections the response echoes back
# are stripped during assembly and rebuilt fresh from the current sources.
_PROGRAMMATIC_START_MARKERS = (_SOURCE_TABLE_MARKER, _BIBLIOGRAPHY_MARKER)


def _find_changes_made_marker(text: str) -> int:
    """Return the index of a 'Changes made' section heading, or -1.

    Matches the marker as its own heading line ("Changes made" optionally
    prefixed by markdown heading markers) so prose that merely contains the
    words "changes made" is not truncated.
    """
    if not text:
        return -1
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip("#").strip()
        if stripped.lower() == _CHANGES_MADE_MARKER.lower():
            return offset
        offset += len(line)
    return -1


def _extract_change_notes(body: str) -> str:
    """Return the writer's 'Changes made' section from an LLM body.

    Finds the section starting at the 'Changes made' marker (heading style
    agnostic) and returns the text from that marker to the end of the body
    (or to the next programmatic section, when the response echoes the full
    assembled report). Returns "" when the marker is absent.
    """
    if not body:
        return ""
    idx = _find_changes_made_marker(body)
    if idx == -1:
        return ""
    end = len(body)
    for marker in _PROGRAMMATIC_START_MARKERS:
        m_idx = body.find(marker, idx)
        if m_idx != -1:
            end = m_idx
            break
    return body[idx:end].strip()


def _strip_previous_change_notes(previous_draft: str) -> str:
    """Drop a prior 'Changes made' disposition block from the previous draft.

    A previous revision's assembled draft may contain a "Changes made"
    block (written by the LLM body). It must not be fed back as context for
    the next revision: each revision writes a fresh block for the current
    items, and stale dispositions would stack. Only the block itself is cut
    (from the marker to the start of the next section); the body and the
    source table/bibliography context stay for the LLM to reference.
    """
    marker_idx = _find_changes_made_marker(previous_draft)
    if marker_idx == -1:
        return previous_draft
    end = len(previous_draft)
    for marker in _PROGRAMMATIC_START_MARKERS:
        idx = previous_draft.find(marker, marker_idx)
        if idx != -1:
            end = idx
            break
    return previous_draft[:marker_idx].rstrip() + "\n" + previous_draft[end:]


def _strip_programmatic_sections(report: str) -> str:
    """Remove the assembled programmatic sections from a full report.

    Removes the programmatic header (the "# Research: ..." line plus the
    "**Intensity:** ..." metadata line) and cuts at the first programmatic
    section marker ("## Source Table" / "## Bibliography"). Markdown headings
    inside the LLM body (including a "Changes made" disposition block) are
    left untouched. Used when assembling so that programmatic sections the
    LLM echoes back in revision mode are discarded in favor of fresh ones.
    """
    if not report:
        return ""
    body = report
    for marker in _PROGRAMMATIC_START_MARKERS:
        idx = body.find(marker)
        if idx != -1:
            body = body[:idx]
            break
    lines = body.splitlines()
    # Drop only the assembled header lines ("# Research: ..." and the
    # "**Intensity:** ..." metadata line) when present.
    if lines and lines[0].startswith("# Research:"):
        lines.pop(0)
        if lines and lines[0].startswith("**Intensity:"):
            lines.pop(0)
    return "\n".join(lines).strip()


def _assemble_draft(header: str, llm_body: str, sources: list) -> str:
    """Assemble the final report from programmatic sections + LLM body.

    The LLM body may still contain programmatic sections if a revision-mode
    response echoed the previous draft; strip them so the source table and
    bibliography are always rebuilt fresh from the current sources. When the
    response is purely programmatic (the pre-marker body strips to ""), the
    body is left empty rather than falling back to the raw response: keeping
    the raw response would re-introduce the programmatic sections and
    duplicate the freshly built ones.
    """
    body = _strip_programmatic_sections(llm_body)
    source_table = _build_source_table(sources)
    bibliography = _build_bibliography(sources)
    return header + "\n" + body + "\n" + source_table + "\n" + bibliography


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
        title = getattr(s, "title", "") or ""
        url = getattr(s, "url", "") or ""
        source_type = getattr(s, "source_type", "unknown") or "unknown"
        reliability = getattr(s, "overall_reliability", "Unknown") or "Unknown"
        rows.append(f"| {i} | {title} | {url} | {source_type} | {reliability} |")
    header = "| # | Title | URL | Type | Reliability |\n|---|-------|-----|------|-------------|\n"
    return "## Source Table\n" + header + "\n".join(rows) + "\n"


def _build_bibliography(sources: list) -> str:
    """Build a numbered bibliography from the sources list."""
    if not sources:
        return ""
    lines = ["## Bibliography"]
    for i, s in enumerate(sources, 1):
        title = getattr(s, "title", "") or "Untitled"
        url = getattr(s, "url", "") or ""
        date_str = getattr(s, "publication_date", "") or ""

        line = f"{i}. {title}"
        if date_str:
            line += f", {date_str}"
        line += f". [{url}]({url})"
        lines.append(line)
    return "\n".join(lines) + "\n"


def writer_node(state: ResearchState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Writer LangGraph node. Synthesizes findings into a structured report.

    First pass: the LLM writes the report body from the findings and the
    header (source count), source table, and bibliography are generated
    programmatically to ensure completeness.

    Revision mode (state has review_items from a REVISE verdict and a
    previous draft_report): the LLM receives the review items with statuses,
    only the new findings/sources since the last audit, and the previous
    draft's LLM-written body, and revises surgically. Programmatic sections
    are always rebuilt fresh from the current sources. The returned
    writer_change_notes carries the LLM's per-item disposition.
    """
    settings = load_config()
    model_name = get_writer_model(settings)

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

    review_items = state.get("review_items") or []
    previous_draft = state.get("draft_report", "")

    if review_items and previous_draft:
        # Revision pass: surgical rewrite of the previous draft. The previous
        # draft (body plus its source table/bibliography context, so items
        # about specific citations can be addressed) is shown to the LLM; any
        # prior pass's "Changes made" block is dropped so the fresh block is
        # written for the current items. Only the new findings/sources from
        # the pass just audited count as new evidence. Programmatic sections
        # of the OUTPUT are always rebuilt fresh below from the current
        # sources.
        revision_mode = True
        revision_context = _strip_previous_change_notes(previous_draft)
        items_text = "\n".join(
            f"- [{i.get('category', 'required')}] ({i.get('status', 'open')}) {i.get('text', '')}"
            for i in review_items
        )
        new_findings_count = state.get("last_round_new_findings", 0)
        if new_findings_count:
            new_evidence_text = _format_findings_for_prompt(findings_raw[-new_findings_count:])
        else:
            new_evidence_text = "(none)"
        prompt_text = REVISION_PROMPT.format(
            query=query,
            review_items=items_text or "(no open items)",
            new_evidence=new_evidence_text,
            previous_draft=revision_context,
        )
    else:
        # First pass (or a draft-less continuation): write the body from all
        # findings as before.
        revision_mode = False
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
    llm_body = response.content if hasattr(response, "content") else str(response)

    # Capture the LLM's "Changes made" disposition (revision mode) before any
    # programmatic-section stripping below discards the end of the response.
    change_notes = _extract_change_notes(llm_body) if revision_mode else ""

    # Assemble the final report: header + LLM body + programmatic sections.
    # The header/source table/bibliography are always rebuilt fresh from the
    # current sources so a revision never carries forward stale sections.
    header = _build_header(query, intensity, len(sources_raw))
    draft_report = _assemble_draft(header, llm_body, sources_raw)

    mode_label = " (revision)" if revision_mode else ""
    emit_progress(
        config,
        f"Writer: draft generated, {len(draft_report)} chars{mode_label}",
        kind="success",
    )

    return {
        "draft_report": draft_report,
        "writer_change_notes": change_notes,
        "messages": [draft_report],
    }
