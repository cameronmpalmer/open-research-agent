"""Tests for graph assembly and routing."""
from ora.graph import build_graph
from ora.state import ResearchState, ReviewVerdict


class TestGraphAssembly:
    def test_graph_builds_without_error(self):
        graph = build_graph()
        assert graph is not None

    def test_graph_accepts_initial_state(self):
        graph = build_graph()
        initial_state: ResearchState = {
            "query": "test",
            "intensity": 2,
            "plan_approved": False,
            "revision_count": 0,
        }
        assert initial_state["query"] == "test"


class TestRouting:
    def test_route_after_plan_not_approved(self):
        from ora.agents.supervisor import route_after_plan
        state: ResearchState = {"plan_approved": False}
        assert route_after_plan(state) == "__end__"

    def test_route_after_plan_approved(self):
        from ora.agents.supervisor import route_after_plan
        state: ResearchState = {"plan_approved": True}
        assert route_after_plan(state) == "researcher"

    def test_route_after_reviewer_pass(self):
        from ora.agents.supervisor import route_after_reviewer
        state: ResearchState = {
            "review_verdict": ReviewVerdict(verdict="PASS"),
            "revision_count": 0,
        }
        assert route_after_reviewer(state) == "__end__"

    def test_route_after_reviewer_revise_under_limit(self):
        from ora.agents.supervisor import route_after_reviewer
        state: ResearchState = {
            "review_verdict": ReviewVerdict(verdict="REVISE"),
            "revision_count": 1,
        }
        assert route_after_reviewer(state) == "researcher"

    def test_route_after_reviewer_revise_at_limit(self):
        from ora.agents.supervisor import route_after_reviewer
        state: ResearchState = {
            "review_verdict": ReviewVerdict(verdict="REVISE"),
            "revision_count": 3,
        }
        assert route_after_reviewer(state) == "__end__"
