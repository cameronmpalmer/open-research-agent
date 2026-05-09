"""Integration smoke test for the full ORA pipeline."""
from ora.graph import build_graph
from ora.state import ResearchState, ReviewVerdict


class TestFullPipeline:
    def test_graph_end_to_end_mocked(self):
        """Verify the graph can be built and accepts initial state."""
        graph = build_graph()
        initial_state: ResearchState = {
            "query": "test query",
            "intensity": 1,
            "plan_approved": True,
            "revision_count": 0,
            "sources": [],
            "findings": [],
            "draft_report": "",
        }
        assert graph is not None
        assert initial_state["query"] == "test query"

    def test_reviewer_blocks_broken_urls(self):
        """Adversarial reviewer should catch broken URLs."""
        from ora.agents.reviewer import parse_reviewer_output
        import json
        output = json.dumps({
            "verdict": "REVISE",
            "blocking": ["URL https://example.com/fake returns 404"],
            "required": [],
            "suggested": [],
            "contradicting_evidence_found": [],
            "confidence_recalibrations": {},
        })
        verdict = parse_reviewer_output(output)
        assert verdict.verdict == "REVISE"
        assert len(verdict.blocking) == 1
        assert "404" in verdict.blocking[0]

    def test_reviewer_passes_clean_report(self):
        """Adversarial reviewer should pass a clean report."""
        from ora.agents.reviewer import parse_reviewer_output
        import json
        output = json.dumps({
            "verdict": "PASS",
            "blocking": [],
            "required": ["minor: add one more source"],
            "suggested": ["add examples"],
            "contradicting_evidence_found": [],
            "confidence_recalibrations": {},
        })
        verdict = parse_reviewer_output(output)
        assert verdict.verdict == "PASS"
        assert len(verdict.blocking) == 0

    def test_complete_state_flow(self):
        """Verify routing through all states."""
        from ora.agents.supervisor import (
            route_after_plan,
            route_after_researcher,
            route_after_writer,
            route_after_reviewer,
        )

        # Plan -> Researcher
        assert route_after_plan({"plan_approved": True}) == "researcher"
        assert route_after_plan({"plan_approved": False}) == "__end__"

        # Researcher -> Writer
        assert route_after_researcher({"findings": [{}]}) == "writer"
        assert route_after_researcher({"findings": []}) == "__end__"

        # Writer -> Reviewer
        assert route_after_writer({"draft_report": "content"}) == "reviewer"
        assert route_after_writer({"draft_report": ""}) == "__end__"

        # Reviewer -> End or Loop
        assert route_after_reviewer({"review_verdict": ReviewVerdict(verdict="PASS"), "revision_count": 0}) == "__end__"
        rev_verdict = ReviewVerdict(verdict="REVISE")
        assert route_after_reviewer({"review_verdict": rev_verdict, "revision_count": 0}) == "researcher"
        assert route_after_reviewer({"review_verdict": rev_verdict, "revision_count": 3}) == "__end__"
