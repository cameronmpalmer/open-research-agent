"""Supervisor agent node for LangGraph."""
import json
import re
from typing import Any, Literal
from langchain_core.runnables import RunnableConfig
from ora.state import ResearchState
from ora.prompts import SUPERVISOR_PLAN_PROMPT, SUPERVISOR_REVISE_PROMPT, _search_query_count
from ora.config import load_config, get_llm, get_supervisor_model


def _invoke_supervisor(prompt: str) -> str:
    """Call the supervisor LLM and return the response text."""
    settings = load_config()
    model_name = get_supervisor_model(settings)
    llm = get_llm(model_name, temperature=0)
    response = llm.invoke(prompt)
    return response.content if hasattr(response, 'content') else str(response)


def _extract_search_queries(plan_text: str) -> list[str]:
    """Extract JSON search queries from supervisor response code fence.

    Returns empty list on any failure (no fence, bad JSON, wrong type).
    """
    m = re.search(r'```\s*search_queries\s*\n(.*?)```', plan_text, re.DOTALL)
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


def plan_node(
    state: ResearchState, config: RunnableConfig = None
) -> dict[str, Any]:
    """Generate a research plan for user review."""
    intensity = state.get("intensity", 2)
    prompt = SUPERVISOR_PLAN_PROMPT.format(
        query=state.get("query", ""),
        intensity=intensity,
        count=_search_query_count(intensity),
    )
    response_text = _invoke_supervisor(prompt)
    return {
        "research_plan": response_text,
        "search_queries": _extract_search_queries(response_text),
        "messages": [response_text],
    }


def revise_plan_text(
    query: str,
    intensity: int,
    current_plan: str,
    feedback: str,
) -> str:
    """Revise a research plan based on user feedback.

    Args:
        query: The original research query.
        intensity: Research intensity level.
        current_plan: The current plan text to revise.
        feedback: Natural-language user feedback to incorporate.

    Returns:
        Revised plan text in markdown.
    """
    prompt = SUPERVISOR_REVISE_PROMPT.format(
        query=query,
        intensity=intensity,
        plan=current_plan,
        feedback=feedback,
        count=_search_query_count(intensity),
    )
    return _invoke_supervisor(prompt)


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

    v = verdict.verdict if hasattr(verdict, 'verdict') else "REVISE"
    revision_count = state.get("revision_count", 0)

    if v == "PASS":
        return "__end__"
    elif revision_count < 3:
        return "researcher"
    else:
        return "__end__"
