"""LangGraph StateGraph assembly for ORA."""
import warnings

warnings.filterwarnings("ignore", message=".*allowed_objects.*", module="langgraph")

from langgraph.graph import END, StateGraph

from ora.agents.researcher import researcher_node
from ora.agents.reviewer import reviewer_node
from ora.agents.supervisor import (
    plan_node,
    route_after_plan,
    route_after_researcher,
    route_after_reviewer,
    route_after_writer,
)
from ora.agents.writer import writer_node
from ora.state import ResearchState


def build_plan_graph() -> StateGraph:
    """Build a graph that only runs the plan node (for review)."""
    workflow = StateGraph(ResearchState)
    workflow.add_node("plan", plan_node)
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", END)
    return workflow.compile()


def build_research_graph(intensity: int = 2, no_review: bool = False) -> StateGraph:
    """Build the research graph.

    For intensity >= 3, includes the adversarial reviewer node for
    multi-round revision. For lower intensities, researcher -> writer only.
    Set no_review=True to skip the reviewer even at intensity >= 3.
    """
    workflow = StateGraph(ResearchState)

    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)

    workflow.set_entry_point("researcher")

    if intensity >= 3 and not no_review:
        workflow.add_node("reviewer", reviewer_node)
        workflow.add_conditional_edges(
            "writer", route_after_writer,
            {"reviewer": "reviewer", "__end__": END},
        )
        workflow.add_conditional_edges(
            "reviewer", route_after_reviewer,
            {"researcher": "researcher", "__end__": END},
        )
    else:
        workflow.add_conditional_edges(
            "writer", route_after_writer,
            {"reviewer": END, "__end__": END},
        )

    workflow.add_conditional_edges(
        "researcher", route_after_researcher,
        {"writer": "writer", "__end__": END},
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
