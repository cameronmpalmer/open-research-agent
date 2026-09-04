"""Tests for graph assembly and routing."""

from langgraph.graph import END, StateGraph

from ora.agents.supervisor import (
    route_after_researcher,
    route_after_reviewer,
    route_after_writer,
)
from ora.graph import build_graph, build_research_graph
from ora.state import Finding, ResearchState, ReviewVerdict


class TestGraphAssembly:
    def test_graph_builds_without_error(self):
        graph = build_graph()
        assert graph is not None

    def test_no_review_skips_reviewer_at_high_intensity(self):
        """With no_review=True, graph should omit the reviewer node at intensity 4."""
        graph = build_research_graph(intensity=4, no_review=True)
        assert graph is not None
        assert "reviewer" not in graph.nodes

    def test_no_review_is_noop_at_low_intensity(self):
        """no_review=True at intensity 1 should compile (no reviewer anyway)."""
        graph = build_research_graph(intensity=1, no_review=True)
        assert graph is not None
        assert "reviewer" not in graph.nodes

    def test_default_includes_reviewer_at_high_intensity(self):
        """Without no_review, intensity 4 should include the reviewer."""
        graph = build_research_graph(intensity=4)
        assert "reviewer" in graph.nodes

    def test_graph_accepts_initial_state(self):
        build_graph()
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

    def test_route_after_reviewer_revise_with_open_item_routes(self):
        """REVISE with budget remaining and an open (non-exhausted) item
        keeps the convergent loop going."""
        from ora.agents.supervisor import route_after_reviewer

        state: ResearchState = {
            "review_verdict": ReviewVerdict(verdict="REVISE"),
            "revision_count": 1,
            "review_items": [{"category": "blocking", "text": "add pricing", "status": "open"}],
        }
        assert route_after_reviewer(state) == "researcher"

    def test_route_after_reviewer_revise_open_alongside_exhausted_routes(self):
        """Open items remaining alongside exhausted ones still route back:
        the exhausted ones wait for the next audit to be closed as gaps."""
        from ora.agents.supervisor import route_after_reviewer

        state: ResearchState = {
            "review_verdict": ReviewVerdict(verdict="REVISE"),
            "revision_count": 2,
            "review_items": [
                {"category": "blocking", "text": "add pricing", "status": "open"},
                {
                    "category": "required",
                    "text": "2026 outlook unavailable",
                    "status": "evidence_exhausted",
                },
            ],
        }
        assert route_after_reviewer(state) == "researcher"

    def test_route_after_reviewer_revise_all_items_exhausted_ends(self):
        """REVISE with only evidence_exhausted items stops: the next audit
        is expected to close them as unresolvable gaps, so no researcher
        round is scheduled."""
        from ora.agents.supervisor import route_after_reviewer

        state: ResearchState = {
            "review_verdict": ReviewVerdict(verdict="REVISE"),
            "revision_count": 1,
            "review_items": [
                {
                    "category": "required",
                    "text": "2026 outlook unavailable",
                    "status": "evidence_exhausted",
                }
            ],
        }
        assert route_after_reviewer(state) == "__end__"

    def test_route_after_reviewer_revise_no_review_items_ends(self):
        """Legacy states without a review_items key have no open items:
        a REVISE verdict without anything actionable ends."""
        from ora.agents.supervisor import route_after_reviewer

        state: ResearchState = {
            "review_verdict": ReviewVerdict(verdict="REVISE"),
            "revision_count": 1,
        }
        assert route_after_reviewer(state) == "__end__"

    def test_route_after_reviewer_revise_at_limit_ends_even_with_open_items(self):
        """revision_count >= 3 remains the hard safety cap: even an open
        item cannot schedule another researcher round."""
        from ora.agents.supervisor import route_after_reviewer

        state: ResearchState = {
            "review_verdict": ReviewVerdict(verdict="REVISE"),
            "revision_count": 3,
            "review_items": [{"category": "blocking", "text": "add pricing", "status": "open"}],
        }
        assert route_after_reviewer(state) == "__end__"


class TestReviewLoopIntegration:
    """Compose the review loop (researcher -> writer -> reviewer) from fake
    nodes and the REAL route functions, then drive the convergent loop.

    Fake nodes stand in for the LLM-backed agents; the routers under test
    (route_after_researcher/writer/reviewer) are the production ones, and
    the reviewer fake mirrors reviewer_node's state contract (review_items
    derived from the verdict, revision_count incremented).
    """

    FIRST_DRAFT = "# Research: test query\n\n## Key Findings\nInitial claim.\n"
    # Revision grows the draft only by the addressed evidence / documented gap.
    REVISED_TAIL = "\n\n## Evidence gaps\n2026 market data is not publicly available.\n"

    def _build_loop(self, researcher, writer, reviewer):
        workflow = StateGraph(ResearchState)
        workflow.add_node("researcher", researcher)
        workflow.add_node("writer", writer)
        workflow.add_node("reviewer", reviewer)
        workflow.set_entry_point("researcher")
        workflow.add_conditional_edges(
            "researcher",
            route_after_researcher,
            {"writer": "writer", "__end__": END},
        )
        workflow.add_conditional_edges(
            "writer",
            route_after_writer,
            {"reviewer": "reviewer", "__end__": END},
        )
        workflow.add_conditional_edges(
            "reviewer",
            route_after_reviewer,
            {"researcher": "researcher", "__end__": END},
        )
        return workflow.compile()

    def _reviewer_fake(self, calls, scripts):
        def reviewer(state):
            calls["reviewer"] += 1
            step = calls["reviewer"] - 1
            verdict, items = scripts[step]
            return {
                "review_verdict": verdict,
                "review_items": items,
                "revision_count": state.get("revision_count", 0) + 1,
            }

        return reviewer

    def test_items_resolved_in_two_passes_end_with_pass(self):
        """Item-resolving loop: initial REVISE -> researcher targets the open
        item -> writer integrates it and documents the exhausted gap ->
        re-audit PASSes with the gap in unresolvable_gaps. No third pass."""
        calls = {"researcher": 0, "writer": 0, "reviewer": 0}

        def researcher(state):
            calls["researcher"] += 1
            if calls["researcher"] == 1:
                return {"findings": [Finding(claim="initial claim")]}
            # REVISE round: pricing evidence found (item stays open for the
            # writer to integrate), outlook exhausted after zero yield.
            return {
                "review_items": [
                    {"category": "blocking", "text": "Add pricing details", "status": "open"},
                    {
                        "category": "required",
                        "text": "2026 outlook unavailable in sources",
                        "status": "evidence_exhausted",
                    },
                ],
                "last_round_new_sources": 1,
                "last_round_new_findings": 1,
            }

        def writer(state):
            calls["writer"] += 1
            if calls["writer"] == 1:
                return {"draft_report": self.FIRST_DRAFT}
            return {
                "draft_report": self.FIRST_DRAFT + self.REVISED_TAIL,
                "writer_change_notes": (
                    "## Changes made\n"
                    "- [blocking] Add pricing details: resolved with new source.\n"
                    "- [required] 2026 outlook unavailable in sources: documented as a gap.\n"
                ),
            }

        reviewer = self._reviewer_fake(
            calls,
            [
                (
                    ReviewVerdict(
                        verdict="REVISE",
                        blocking=["Add pricing details"],
                        required=["2026 outlook unavailable in sources"],
                    ),
                    [
                        {"category": "blocking", "text": "Add pricing details", "status": "open"},
                        {
                            "category": "required",
                            "text": "2026 outlook unavailable in sources",
                            "status": "open",
                        },
                    ],
                ),
                (
                    ReviewVerdict(
                        verdict="PASS",
                        unresolvable_gaps=["2026 outlook unavailable in sources"],
                    ),
                    [],
                ),
            ],
        )

        graph = self._build_loop(researcher, writer, reviewer)
        final = graph.invoke({"query": "test query", "intensity": 3, "revision_count": 0})

        assert calls["researcher"] == 2
        assert calls["reviewer"] == 2
        assert final["revision_count"] == 2
        assert final["review_verdict"].verdict == "PASS"
        assert final["review_verdict"].unresolvable_gaps == ["2026 outlook unavailable in sources"]
        assert final["review_items"] == []
        assert "Evidence gaps" in final["draft_report"]

    def test_all_items_exhausted_finalizes_without_draft_ballooning(self):
        """Exhausted-items path: the single open item yields no evidence, the
        researcher exhausts it, the writer documents the limitation, and the
        re-audit PASSes with an unresolvable gap - after only 2 revision
        passes and with no draft length explosion."""
        calls = {"researcher": 0, "writer": 0, "reviewer": 0}

        def researcher(state):
            calls["researcher"] += 1
            if calls["researcher"] == 1:
                return {"findings": [Finding(claim="initial claim")]}
            # Zero-yield item-targeted pass: exhaust the only open item.
            return {
                "review_items": [
                    {
                        "category": "blocking",
                        "text": "Find a 2026 market forecast",
                        "status": "evidence_exhausted",
                    }
                ],
                "last_round_new_sources": 0,
                "last_round_new_findings": 0,
            }

        def writer(state):
            calls["writer"] += 1
            if calls["writer"] == 1:
                return {"draft_report": self.FIRST_DRAFT}
            return {
                "draft_report": self.FIRST_DRAFT + self.REVISED_TAIL,
                "writer_change_notes": (
                    "## Changes made\n"
                    "- [blocking] Find a 2026 market forecast: documented as a gap.\n"
                ),
            }

        reviewer = self._reviewer_fake(
            calls,
            [
                (
                    ReviewVerdict(
                        verdict="REVISE",
                        blocking=["Find a 2026 market forecast"],
                    ),
                    [
                        {
                            "category": "blocking",
                            "text": "Find a 2026 market forecast",
                            "status": "open",
                        }
                    ],
                ),
                (
                    ReviewVerdict(
                        verdict="PASS",
                        unresolvable_gaps=["Find a 2026 market forecast"],
                    ),
                    [],
                ),
            ],
        )

        graph = self._build_loop(researcher, writer, reviewer)
        final = graph.invoke({"query": "test query", "intensity": 3, "revision_count": 0})

        assert calls["reviewer"] == 2
        assert calls["researcher"] == 2
        assert final["revision_count"] == 2
        assert final["review_verdict"].verdict == "PASS"
        assert final["review_verdict"].unresolvable_gaps == ["Find a 2026 market forecast"]
        assert final["review_items"] == []
        # No ballooning: the revised draft only appends the documented gap.
        assert len(final["draft_report"]) <= len(self.FIRST_DRAFT) + 200
