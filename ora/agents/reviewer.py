"""Adversarial reviewer agent node for LangGraph."""

import json
from typing import Any

from langchain_core.runnables import RunnableConfig

from ora.config import get_llm, get_reviewer_model, load_config
from ora.prompts import REVIEWER_PROMPT
from ora.state import ResearchState, ReviewVerdict


def parse_reviewer_output(output: str) -> ReviewVerdict:
    """Parse the reviewer's JSON output into a ReviewVerdict.

    Handles JSON in markdown code fences and malformed JSON.
    """
    try:
        # Extract JSON from markdown fences if present
        if "```json" in output:
            start = output.index("```json") + 7
            end = output.index("```", start)
            json_str = output[start:end].strip()
        elif "```" in output:
            start = output.index("```") + 3
            end = output.index("```", start)
            json_str = output[start:end].strip()
        else:
            json_str = output.strip()

        data = json.loads(json_str)
        return ReviewVerdict(
            verdict=data.get("verdict", "PASS"),
            blocking=data.get("blocking", []),
            required=data.get("required", []),
            suggested=data.get("suggested", []),
            contradicting_evidence_found=data.get("contradicting_evidence_found", []),
            confidence_recalibrations=data.get("confidence_recalibrations", {}),
            unresolvable_gaps=data.get("unresolvable_gaps", []),
        )
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        return ReviewVerdict(
            verdict="REVISE",
            blocking=[f"Reviewer output parsing failed: {e!s}. Raw: {output[:200]}"],
        )


def review_items_from_verdict(verdict) -> list[dict]:
    """Convert a ReviewVerdict (or its dict form from a checkpoint) into open
    review_items for the next pass.

    blocking and required become work items (status "open"); suggested and
    unresolvable_gaps are context only and are not routed to research.
    """
    if verdict is None:
        return []
    if isinstance(verdict, dict):
        blocking = verdict.get("blocking") or []
        required = verdict.get("required") or []
    else:
        blocking = getattr(verdict, "blocking", None) or []
        required = getattr(verdict, "required", None) or []
    items = [{"category": "blocking", "text": b, "status": "open"} for b in blocking]
    items += [{"category": "required", "text": r, "status": "open"} for r in required]
    return items


def reviewer_node(state: ResearchState, config: RunnableConfig = None) -> dict[str, Any]:
    """Adversarial reviewer LangGraph node.

    Receives the draft report and original query. Does NOT receive the
    researcher's intermediate findings or search queries. Uses a different
    model provider from the researcher to prevent correlated errors.

    On re-audits (state carries review_items from the previous audit) the
    prompt additionally receives the previous items with their statuses, the
    writer's change notes, and the count of new sources found since the last
    audit, so the reviewer can verify the writer's claimed dispositions
    against the updated report instead of re-raising resolved items.
    """
    from ora.progress import emit_progress

    settings = load_config()
    model_name = get_reviewer_model(settings)

    emit_progress(config, "Reviewer: auditing draft report...", kind="review")

    llm = get_llm(model_name, temperature=0.2)

    items = state.get("review_items", [])
    items_text = (
        "\n".join(f"- [{i.get('category')}] ({i.get('status')}) {i.get('text')}" for i in items)
        or "(first audit)"
    )
    prompt_text = REVIEWER_PROMPT.format(
        query=state.get("query", ""),
        report=state.get("draft_report", ""),
        review_items=items_text,
        writer_change_notes=state.get("writer_change_notes", "") or "(no revision notes)",
        new_sources_count=str(state.get("last_round_new_sources", 0)),
    )

    response = llm.invoke(prompt_text)
    output = response.content if hasattr(response, "content") else str(response)

    verdict = parse_reviewer_output(output)

    v = verdict.verdict if hasattr(verdict, "verdict") else "PASS"
    if v == "REVISE":
        emit_progress(
            config,
            "Reviewer: REVISE — restarting research to address gaps",
            kind="warning",
        )
    else:
        emit_progress(config, "Reviewer: PASS — report accepted", kind="success")

    return {
        "review_verdict": verdict,
        "review_verdict_raw": output,
        "revision_count": state.get("revision_count", 0) + 1,
        "review_items": review_items_from_verdict(verdict),
        # NOTE: last_round_new_sources/findings are NOT reset here on purpose.
        # route_after_reviewer runs immediately after this node and needs the
        # just-audited pass's deltas to decide whether REVISE should continue.
        # The researcher overwrites them on its next pass.
        "messages": [output],
    }
