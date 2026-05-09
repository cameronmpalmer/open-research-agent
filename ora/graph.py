"""LangGraph StateGraph assembly for ORA."""
import warnings
warnings.filterwarnings("ignore", message=".*allowed_objects.*", module="langgraph")

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
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


def build_graph() -> StateGraph:
    """Build and compile the ORA research graph.

    Graph: plan -> [HITL interrupt] -> researcher -> writer -> reviewer -> [loop|end]
    """
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

    memory = MemorySaver()
    return workflow.compile(
        checkpointer=memory,
        interrupt_before=["researcher"],
    )
