"""Adversarial reviewer agent node for LangGraph."""

import json
from typing import Any

from langchain_core.runnables import RunnableConfig

from ora.agents.supervisor import MAX_REVISIONS
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


def _norm_item_text(text: str) -> str:
    """Normalize an item text for repeat-raise comparison."""
    return (text or "").strip().casefold()


def _fold_repeated_exhausted(verdict: ReviewVerdict, previous_items: list[dict]) -> ReviewVerdict:
    """Fold re-raised evidence_exhausted items out of blocking/required.

    Hard guard behind the reviewer prompt's no-repeat-raise rule: when the
    parsed verdict lists a blocking/required item whose normalized text was
    previously marked evidence_exhausted, the item must not become a fresh
    open work item (that would burn another revision on something already
    accepted as a gap). Its text is moved into unresolvable_gaps instead and
    removed from blocking/required. The verdict value itself is left alone:
    with nothing left in blocking/required the routing guard stops the loop
    (no open review_items), and the CLI surfaces the folded gaps.
    """
    if verdict is None or not previous_items:
        return verdict
    exhausted = {
        _norm_item_text(i.get("text"))
        for i in previous_items
        if i.get("status") == "evidence_exhausted"
    }
    if not exhausted:
        return verdict

    folded: list[str] = []
    kept_blocking: list[str] = []
    kept_required: list[str] = []
    for text in verdict.blocking or []:
        (folded if _norm_item_text(text) in exhausted else kept_blocking).append(text)
    for text in verdict.required or []:
        (folded if _norm_item_text(text) in exhausted else kept_required).append(text)
    if not folded:
        return verdict

    verdict.blocking = kept_blocking
    verdict.required = kept_required
    verdict.unresolvable_gaps = list(verdict.unresolvable_gaps or []) + folded
    return verdict


def reviewer_node(state: ResearchState, config: RunnableConfig = None) -> dict[str, Any]:
    """Adversarial reviewer LangGraph node.

    Receives the draft report and original query. Does NOT receive the
    researcher's intermediate findings or search queries. Uses a different
    model provider from the researcher to prevent correlated errors.

    On re-audits (state carries review_items from the previous audit) the
    prompt additionally receives the previous items with their statuses, the
    writer's change notes, and the count of new sources found since the last
    audit, so the reviewer can verify the writer's claimed dispositions
    against the updated report instead of re-raising resolved items. The
    audit number and the revision budget (state max_revisions when wired in,
    else the MAX_REVISIONS constant) are passed too, so the reviewer knows
    when it is at its final audit and folds residual addressed/accepted
    concerns into unresolvable_gaps instead of issuing a REVISE that the
    routing cap would discard. A hard guard folds re-raised
    evidence_exhausted items out of blocking/required into
    unresolvable_gaps.
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
    # The audit currently being performed corresponds to the count this node
    # is about to return (revision_count is incremented below), so the prompt
    # gets revision_count + 1 as the audit number and the run's revision
    # budget as the cap. The cap mirrors the routing budget used by
    # route_after_reviewer (state max_revisions when wired in by the CLI,
    # else the MAX_REVISIONS constant) so the reviewer knows when it is at
    # its FINAL audit and must not REVISE for residual addressed/accepted
    # items. When audit_number == cap the reviewer is at its final audit.
    cap = state.get("max_revisions") or MAX_REVISIONS
    audit_number = state.get("revision_count", 0) + 1
    prompt_text = REVIEWER_PROMPT.format(
        query=state.get("query", ""),
        report=state.get("draft_report", ""),
        audit_number=audit_number,
        max_audits=cap,
        review_items=items_text,
        writer_change_notes=state.get("writer_change_notes", "") or "(no revision notes)",
        new_sources_count=str(state.get("last_round_new_sources", 0)),
    )

    response = llm.invoke(prompt_text)
    output = response.content if hasattr(response, "content") else str(response)

    verdict = parse_reviewer_output(output)
    # Hard repeat-raise guard: if the model re-raises a blocking/required
    # item whose normalized text was already evidence_exhausted in a previous
    # audit, fold it into unresolvable_gaps instead of letting it become a
    # fresh open item that burns another revision round.
    verdict = _fold_repeated_exhausted(verdict, state.get("review_items", []))

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
        # NOTE: last_round_new_sources/findings are deliberately not reset
        # here: the routing guard reads review_items statuses, and the next
        # audit's NEW_SOURCES_SINCE_LAST_AUDIT context needs the researcher's
        # just-completed deltas to survive this node. The researcher
        # overwrites them on its next pass.
        "messages": [output],
    }
