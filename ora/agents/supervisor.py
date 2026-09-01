"""Supervisor agent node for LangGraph."""

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


def _search_queries_fence_found(plan_text: str) -> bool:
    """Return True if the plan text contains a search_queries code fence,
    regardless of whether the content can be parsed correctly."""
    return bool(_FENCE_RE.search(plan_text))


def _extract_search_queries(plan_text: str) -> list[str]:
    """Extract JSON search queries from supervisor response code fence.

    Returns empty list on any failure (no fence, bad JSON, wrong type).
    """
    m = _FENCE_RE.search(plan_text)
    if not m:
        return []

    json_str = m.group(1).strip()

    # Try JSON first, then Python literal eval (LLMs sometimes use single quotes)
    try:
        result = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        try:
            import ast

            result = ast.literal_eval(json_str)
        except (ValueError, SyntaxError):
            return []

    if isinstance(result, list) and all(isinstance(s, str) for s in result):
        return result
    return []


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
    """Route after review: revise if needed, end if pass or max revisions."""
    verdict = state.get("review_verdict")
    if verdict is None:
        return "__end__"

    v = verdict.verdict if hasattr(verdict, "verdict") else "REVISE"
    revision_count = state.get("revision_count", 0)

    if v == "PASS":
        return "__end__"
    elif revision_count < 3:
        return "researcher"
    else:
        return "__end__"
