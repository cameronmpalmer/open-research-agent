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


# Disposition vocabulary the writer uses in its "Changes made" notes
# (see REVISION_PROMPT). A previously-exhausted item counts as acknowledged
# when the writer restated the item next to one of these markers.
_ACKNOWLEDGED_MARKERS = ("resolved", "partially addressed", "documented as a gap")


def _notes_acknowledge_item(notes: str, text: str) -> bool:
    """True when the writer's change notes respond to the item.

    The writer's change notes restate the item text together with a
    disposition marker (resolved / partially addressed / documented as a
    gap) on the same line, for example
    "- [blocking] Add pricing details: resolved." An item whose text merely
    appears somewhere in the notes (a near-duplicate the writer actually
    addressed, say "Add pricing details" when the item is "Add pricing") is
    not acknowledgment, so the item text must co-occur with a marker on one
    line. Matching is normalized (strip + casefold).
    """
    if not notes or not text:
        return False
    norm_notes = _norm_item_text(notes)
    norm_text = _norm_item_text(text)
    if not norm_text:
        return False
    return any(
        norm_text in line and any(marker in line for marker in _ACKNOWLEDGED_MARKERS)
        for line in norm_notes.splitlines()
    )


def _should_fold_repeated_exhausted(
    text: str, writer_change_notes: str, is_final_audit: bool
) -> bool:
    """Decide whether a re-raised previously-exhausted item may be folded.

    Fold (accept into unresolvable_gaps) only when the writer acknowledged
    the item in its change notes (so the report documents the limitation), or
    when this is the final audit (no researcher pass will follow a REVISE).
    Otherwise the item stays in blocking/required so the reviewer's
    exhausted-but-undocumented REVISE can reach the writer.
    """
    return is_final_audit or _notes_acknowledge_item(writer_change_notes, text)


def _fold_repeated_exhausted(
    verdict: ReviewVerdict,
    previous_items: list[dict],
    writer_change_notes: str = "",
    is_final_audit: bool = False,
) -> ReviewVerdict:
    """Fold re-raised evidence_exhausted items out of blocking/required.

    Hard guard behind the reviewer prompt's no-repeat-raise rule: when the
    parsed verdict lists a blocking/required item whose normalized text was
    previously marked evidence_exhausted, the item must not become a fresh
    open work item that burns another revision on something the writer has
    already answered. It is folded out of blocking/required into
    unresolvable_gaps ONLY when the writer acknowledged it in the change
    notes (the report documents the limitation) or this is the final audit.
    Otherwise it stays in blocking/required so an exhausted-but-undocumented
    documentation-gap REVISE can reach the writer. The verdict value itself
    is left alone: with nothing left in blocking/required the routing guard
    stops the loop (no open review_items), and the CLI surfaces the gaps.
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
        norm = _norm_item_text(text)
        if norm in exhausted and _should_fold_repeated_exhausted(
            text, writer_change_notes, is_final_audit
        ):
            folded.append(text)
        else:
            kept_blocking.append(text)
    for text in verdict.required or []:
        norm = _norm_item_text(text)
        if norm in exhausted and _should_fold_repeated_exhausted(
            text, writer_change_notes, is_final_audit
        ):
            folded.append(text)
        else:
            kept_required.append(text)
    if not folded:
        return verdict

    verdict.blocking = kept_blocking
    verdict.required = kept_required
    # Dedupe gap text: the same text may appear in blocking AND required, and
    # may already be present in the model-supplied unresolvable_gaps.
    existing_gaps = {_norm_item_text(g) for g in (verdict.unresolvable_gaps or [])}
    for text in folded:
        norm = _norm_item_text(text)
        if norm not in existing_gaps:
            verdict.unresolvable_gaps.append(text)
            existing_gaps.add(norm)
    return verdict


def _carry_exhausted_items(
    verdict: ReviewVerdict,
    previous_items: list[dict],
    writer_change_notes: str = "",
) -> list[dict]:
    """Carry previously-exhausted items forward as evidence_exhausted records.

    Items the reviewer neither re-raised this audit (so the fold decision did
    not handle them) nor accepted into this verdict's unresolvable_gaps keep
    their evidence_exhausted record in review_items. That keeps the
    repeat-raise guard multi-shot (history survives into the next audit) and
    the record cannot silently disappear from structured state when the
    reviewer fails to re-raise or accept a previously-exhausted item. The
    writer acknowledging the item in its change notes is deliberately NOT
    enough to drop it: an acknowledged item the reviewer neither re-raised nor
    accepted would otherwise vanish from both review_items and
    unresolvable_gaps, losing the CLI's open/exhausted accounting and any
    downstream consumer of structured gaps. Routing already ignores non-open
    statuses, so a carried record never schedules another researcher pass.

    The ``writer_change_notes`` argument is retained so callers and tests can
    supply the writer's dispositions (the contract this function guards).
    """
    if not previous_items:
        return []
    re_raised = {_norm_item_text(t) for t in (verdict.blocking or []) + (verdict.required or [])}
    accepted = {_norm_item_text(g) for g in (verdict.unresolvable_gaps or [])}
    carried: list[dict] = []
    for item in previous_items:
        if item.get("status") != "evidence_exhausted":
            continue
        text = item.get("text") or ""
        norm = _norm_item_text(text)
        if not norm:
            continue
        if norm in re_raised or norm in accepted:
            continue
        carried.append(
            {
                "category": item.get("category", "blocking"),
                "text": text,
                "status": "evidence_exhausted",
            }
        )
    return carried


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
    unresolvable_gaps only when the writer acknowledged the item in its
    change notes or this is the final audit; previously-exhausted items the
    writer has not yet acknowledged are carried forward as evidence_exhausted
    records so history survives into the next audit and the writer sees them
    for documentation.
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
    # An explicit None check, not truthiness: a wired-in 0 or negative budget
    # must be honored as-is rather than silently raised to MAX_REVISIONS.
    max_revisions = state.get("max_revisions")
    cap = max_revisions if max_revisions is not None else MAX_REVISIONS
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

    previous_items = state.get("review_items", [])
    writer_change_notes = state.get("writer_change_notes", "") or ""
    is_final_audit = audit_number >= cap

    verdict = parse_reviewer_output(output)
    # Hard repeat-raise guard: if the model re-raises a blocking/required
    # item whose normalized text was already evidence_exhausted in a previous
    # audit, fold it into unresolvable_gaps instead of letting it become a
    # fresh open item that burns another revision round. Folding is limited
    # to items the writer acknowledged in its change notes (so the report
    # documents the limitation) or to the final audit; otherwise the item
    # stays in blocking/required so the exhausted-but-undocumented
    # documentation-gap REVISE can reach the writer.
    verdict = _fold_repeated_exhausted(verdict, previous_items, writer_change_notes, is_final_audit)
    # Carry previously-exhausted items the writer has not yet acknowledged
    # forward as evidence_exhausted records (I2): the guard keeps history and
    # the end state cannot silently drop the exhausted item's text.
    review_items = review_items_from_verdict(verdict) + _carry_exhausted_items(
        verdict, previous_items, writer_change_notes
    )

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
        "review_items": review_items,
        # NOTE: last_round_new_sources/findings are deliberately not reset
        # here: the routing guard reads review_items statuses, and the next
        # audit's NEW_SOURCES_SINCE_LAST_AUDIT context needs the researcher's
        # just-completed deltas to survive this node. The researcher
        # overwrites them on its next pass.
        "messages": [output],
    }
