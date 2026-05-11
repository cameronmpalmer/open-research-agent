"""LangGraph StateGraph assembly for ORA."""
import warnings
warnings.filterwarnings("ignore", message=".*allowed_objects.*", module="langgraph")

from langgraph.graph import StateGraph, END
from ora.state import ResearchState
from ora.agents.supervisor import (
    plan_node,
    route_after_plan,
    route_after_researcher,
    route_after_writer,
    route_after_reviewer,
)
from ora.agents.researcher import researcher_node
from ora.agents.writer import writer_node
from ora.agents.reviewer import reviewer_node


def build_plan_graph() -> StateGraph:
    """Build a graph that only runs the plan node (for review)."""
    workflow = StateGraph(ResearchState)
    workflow.add_node("plan", plan_node)
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", END)
    return workflow.compile()


def build_research_graph() -> StateGraph:
    """Build the research graph (researcher -> writer). No reviewer for now."""
    workflow = StateGraph(ResearchState)

    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)

    workflow.set_entry_point("researcher")

    workflow.add_conditional_edges(
        "researcher", route_after_researcher,
        {"writer": "writer", "__end__": END},
    )
    workflow.add_conditional_edges(
        "writer", route_after_writer,
        {"reviewer": END, "__end__": END},
    )

    return workflow.compile()


def build_graph() -> StateGraph:
    """Build the full graph with reviewer (for backward compat / testing)."""
    workflow = StateGraph(ResearchState)

    workflow.add_node("plan", plan_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)

    workflow.set_entry_point("plan")

    workflow.add_conditional_edges(
        "plan", route_after_plan,
        {"researcher": "researcher", "__end__": END},
    )
    workflow.add_conditional_edges(
        "researcher", route_after_researcher,
        {"writer": "writer", "__end__": END},
    )
    workflow.add_conditional_edges(
        "writer", route_after_writer,
        {"reviewer": "reviewer", "__end__": END},
    )
    workflow.add_conditional_edges(
        "reviewer", route_after_reviewer,
        {"researcher": "researcher", "__end__": END},
    )

    return workflow.compile()
