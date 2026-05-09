"""Supervisor agent node for LangGraph."""
from typing import Any, Literal
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from ora.state import ResearchState
from ora.prompts import SUPERVISOR_PLAN_PROMPT
from ora.config import load_config


async def plan_node(
    state: ResearchState, config: RunnableConfig = None
) -> dict[str, Any]:
    """Generate a research plan for user review."""
    settings = load_config()
    llm = ChatOpenAI(
        model=settings.models.default.split(":")[-1],
        temperature=0,
    )

    prompt = SUPERVISOR_PLAN_PROMPT.format(
        query=state.get("query", ""),
        intensity=state.get("intensity", 2),
    )

    response = llm.invoke(prompt)
    return {
        "research_plan": response.content if hasattr(response, 'content') else str(response),
        "messages": [response.content if hasattr(response, 'content') else str(response)],
    }


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
