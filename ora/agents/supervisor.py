"""Supervisor agent node for LangGraph."""

import ast
import json
import re
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig

from ora.config import get_llm, get_supervisor_model, load_config
from ora.progress import emit_progress
from ora.prompts import SUPERVISOR_PLAN_PROMPT, SUPERVISOR_REVISE_PROMPT, _search_query_count
from ora.state import ResearchState


def _invoke_supervisor(prompt: str) -> str:
    """Call the supervisor LLM and return the response text."""
    settings = load_config()
    model_name = get_supervisor_model(settings)
    llm = get_llm(model_name, temperature=0)
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


_FENCE_RE = re.compile(r"```\s*search_queries\s*\n(.*?)```", re.DOTALL)

# Hard cap on writer-reviewer revision cycles. Shared by the routing budget
# (route_after_reviewer) and the reviewer's audit context (audit_number /
# max_audits) so the reviewer knows when it is at its final audit.
MAX_REVISIONS = 3


def _verdict_value(state: ResearchState) -> str:
    """Return the review verdict string from state, tolerating both a
    ReviewVerdict model and its dict form (e.g. from a checkpoint).

    The value is normalized to upper case before comparison so a
    checkpointed lowercase "pass" is not misread as a revision request.

    Falls back to "REVISE" when the recorded verdict carries no readable
    value, matching the pre-existing routing assumption that a review object
    without a PASS verdict is a revision request.
    """
    verdict = state.get("review_verdict")
    if isinstance(verdict, dict):
        raw = verdict.get("verdict", "REVISE")
    else:
        raw = getattr(verdict, "verdict", "REVISE")
    return str(raw).upper() if raw else "REVISE"


def _search_queries_fence_found(plan_text: str) -> bool:
    """Return True if the plan text contains a search_queries code fence,
    regardless of whether the content can be parsed correctly."""
    return bool(_FENCE_RE.search(plan_text))


def _parse_query_list(text: str) -> list[str] | None:
    """Parse one text blob into a list of strings, or return None.

    Tries JSON first, then Python literal eval (LLMs sometimes use single
    quotes). Only a list whose items are all strings is accepted.
    """
    try:
        result = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        try:
            result = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None

    if isinstance(result, list) and all(isinstance(s, str) for s in result):
        return result
    return None


def _extract_search_queries(plan_text: str) -> list[str]:
    """Extract search queries from the supervisor response code fence.

    Returns empty list on any failure (no fence, unparseable block, wrong
    type). Accepts a single JSON/Python list spanning the whole block, and
    recovers the shape where the model emitted one independent array per
    line: every nonblank line must then be a list of strings, and the
    ordered queries are flattened. A malformed line rejects the whole
    block rather than salvaging a partial list.
    """
    m = _FENCE_RE.search(plan_text)
    if not m:
        return []

    block = m.group(1).strip()

    whole = _parse_query_list(block)
    if whole is not None:
        return whole

    queries: list[str] = []
    saw_line = False
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        saw_line = True
        parsed = _parse_query_list(stripped)
        if parsed is None:
            return []
        queries.extend(parsed)
    return queries if saw_line else []


def plan_node(state: ResearchState, config: RunnableConfig = None) -> dict[str, Any]:
    """Generate a research plan for user review."""
    intensity = state.get("intensity", 2)
    prompt = SUPERVISOR_PLAN_PROMPT.format(
        query=state.get("query", ""),
        intensity=intensity,
        count=_search_query_count(intensity),
    )
    response_text = _invoke_supervisor(prompt)
    search_queries = _extract_search_queries(response_text)
    if not search_queries and _search_queries_fence_found(response_text):
        emit_progress(
            config,
            "Supervisor provided a search_queries block but queries could not be parsed, "
            "falling back to template-generated queries",
            kind="warning",
        )
    return {
        "research_plan": response_text,
        "search_queries": search_queries,
        "messages": [response_text],
    }


def revise_plan_text(
    query: str,
    intensity: int,
    current_plan: str,
    feedback: str,
) -> tuple[str, list[str]]:
    """Revise a research plan based on user feedback.

    Args:
        query: The original research query.
        intensity: Research intensity level.
        current_plan: The current plan text to revise.
        feedback: Natural-language user feedback to incorporate.

    Returns:
        Tuple of (revised plan text, extracted search queries).
    """
    prompt = SUPERVISOR_REVISE_PROMPT.format(
        query=query,
        intensity=intensity,
        plan=current_plan,
        feedback=feedback,
        count=_search_query_count(intensity),
    )
    response = _invoke_supervisor(prompt)
    return response, _extract_search_queries(response)


def route_after_plan(state: ResearchState) -> Literal["researcher", "__end__"]:
    """Route after plan: researcher if approved, wait otherwise."""
    if state.get("plan_approved", False):
        return "researcher"
    return "__end__"


def route_after_researcher(state: ResearchState) -> Literal["writer", "__end__"]:
    """Route after research: writer if findings exist."""
    findings = state.get("findings", [])
    if findings:
        return "writer"
    return "__end__"


def route_after_writer(state: ResearchState) -> Literal["reviewer", "__end__"]:
    """Route after writer: reviewer if draft exists."""
    if state.get("draft_report"):
        return "reviewer"
    return "__end__"


def route_after_reviewer(state: ResearchState) -> Literal["researcher", "__end__"]:
    """Route after review: revise while open review items remain, end on
    PASS, on budget exhaustion, or when nothing actionable is left.

    The REVISE branch is a convergent guard, not a blind counter: it only
    sends execution back to the researcher while at least one open
    (non-exhausted) review item remains. Zero-progress passes still route
    back while an open item remains (letting per-item attempts accumulate
    across passes); when the researcher has exhausted every item, the next
    audit closes them as unresolvable gaps and the loop stops.
    """
    verdict = state.get("review_verdict")
    if verdict is None:
        return "__end__"

    v = _verdict_value(state)
    revision_count = state.get("revision_count", 0)
    # The routing budget is the state's max_revisions when the CLI/config
    # wired one in; otherwise the module constant (3). Both default to 3, so
    # a state without the key keeps the historical cap. An explicit None
    # check, not truthiness, so a wired-in 0 or negative budget is honored
    # as-is instead of being silently raised to MAX_REVISIONS.
    max_revisions = state.get("max_revisions")
    cap = max_revisions if max_revisions is not None else MAX_REVISIONS

    if v == "PASS":
        return "__end__"
    elif revision_count < cap:
        # Convergent loop: only keep revising while at least one open
        # (non-exhausted) review item remains. Zero-progress passes still
        # route back while an open item remains, letting per-item attempts
        # accumulate across passes; when all items are exhausted the next
        # audit closes them as unresolvable gaps.
        open_items = [i for i in state.get("review_items", []) if i.get("status") == "open"]
        if open_items:
            return "researcher"
        return "__end__"
    else:
        return "__end__"
