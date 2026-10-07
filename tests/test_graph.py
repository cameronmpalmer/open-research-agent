"""Tests for graph assembly and routing."""

import pytest
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

    def test_route_after_reviewer_dict_form_pass_ends(self):
        """Checkpointed dict-form verdicts read their own 'verdict' key: a
        dict PASS must end the loop, not be misread as a REVISE."""
        from ora.agents.supervisor import route_after_reviewer

        state: ResearchState = {
            "review_verdict": {"verdict": "PASS"},
            "revision_count": 1,
            "review_items": [{"category": "blocking", "text": "add pricing", "status": "open"}],
        }
        assert route_after_reviewer(state) == "__end__"

    def test_route_after_reviewer_dict_form_revise_with_open_item_routes(self):
        """Dict-form REVISE verdicts with an open item keep looping."""
        from ora.agents.supervisor import route_after_reviewer

        state: ResearchState = {
            "review_verdict": {"verdict": "REVISE"},
            "revision_count": 1,
            "review_items": [{"category": "blocking", "text": "add pricing", "status": "open"}],
        }
        assert route_after_reviewer(state) == "researcher"

    def test_route_after_reviewer_respects_state_budget_beyond_default(self):
        """A state max_revisions of 5 lets the loop continue past the default
        3-revision cap: revision_count 4 with an open item still routes."""
        from ora.agents.supervisor import route_after_reviewer

        state: ResearchState = {
            "review_verdict": ReviewVerdict(verdict="REVISE"),
            "revision_count": 4,
            "max_revisions": 5,
            "review_items": [{"category": "blocking", "text": "add pricing", "status": "open"}],
        }
        assert route_after_reviewer(state) == "researcher"

    def test_route_after_reviewer_state_budget_caps_at_configured_value(self):
        """The state budget is the cap: revision_count 5 with max_revisions 5
        ends even with an open item."""
        from ora.agents.supervisor import route_after_reviewer

        state: ResearchState = {
            "review_verdict": ReviewVerdict(verdict="REVISE"),
            "revision_count": 5,
            "max_revisions": 5,
            "review_items": [{"category": "blocking", "text": "add pricing", "status": "open"}],
        }
        assert route_after_reviewer(state) == "__end__"

    @pytest.mark.parametrize("cap", [0, 1, 2])
    def test_route_after_reviewer_honors_state_budget_exactly(self, cap):
        """The wired-in budget is used as-is: at revision_count == cap the
        loop ends, and below it an open item still routes. An explicit
        absence check is required because ``cap or MAX_REVISIONS`` would
        silently raise a wired-in 0 to 3."""
        from ora.agents.supervisor import route_after_reviewer

        base: ResearchState = {
            "review_verdict": ReviewVerdict(verdict="REVISE"),
            "max_revisions": cap,
            "review_items": [{"category": "blocking", "text": "add pricing", "status": "open"}],
        }
        assert route_after_reviewer({**base, "revision_count": cap}) == "__end__"
        if cap > 0:
            assert route_after_reviewer({**base, "revision_count": cap - 1}) == "researcher"

    def test_route_after_reviewer_dict_form_lowercase_pass_ends(self):
        """A checkpointed lowercase dict verdict is normalized before
        comparison: "pass" must end the loop, not be treated as a REVISE."""
        from ora.agents.supervisor import route_after_reviewer

        state: ResearchState = {
            "review_verdict": {"verdict": "pass"},
            "revision_count": 1,
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

    def test_budget_capped_with_open_items_ends_at_revision_limit(self):
        """A persistently open item that never exhausts and never resolves is
        capped by the revision budget: three REVISE audits, each followed by
        a writer pass that honestly documents its attempt, then the loop
        stops at revision_count == MAX_REVISIONS with the REVISE verdict and
        the open item retained in the end state."""
        from ora.agents.supervisor import MAX_REVISIONS

        calls = {"researcher": 0, "writer": 0, "reviewer": 0}
        item_text = "Find a 2026 market forecast"

        def researcher(state):
            calls["researcher"] += 1
            if calls["researcher"] == 1:
                return {"findings": [Finding(claim="initial claim")]}
            # One fresh query per pass, zero yield; the item is never
            # exhausted (only 1 attempt) and never resolved.
            return {
                "review_items": [{"category": "blocking", "text": item_text, "status": "open"}],
                "last_round_new_sources": 0,
                "last_round_new_findings": 0,
            }

        def writer(state):
            calls["writer"] += 1
            if calls["writer"] == 1:
                return {"draft_report": self.FIRST_DRAFT}
            # Every revision honestly records the attempt in a "Changes made"
            # block (mirrors the real writer keeping dispositions in the body).
            return {
                "draft_report": (
                    self.FIRST_DRAFT
                    + "\n## Changes made\n"
                    + f"- [blocking] {item_text}: attempted, no public data found.\n"
                ),
                "writer_change_notes": (
                    "## Changes made\n"
                    + f"- [blocking] {item_text}: attempted, no public data found.\n"
                ),
            }

        reviewer = self._reviewer_fake(
            calls,
            [
                (
                    ReviewVerdict(verdict="REVISE", blocking=[item_text]),
                    [{"category": "blocking", "text": item_text, "status": "open"}],
                ),
                (
                    ReviewVerdict(verdict="REVISE", blocking=[item_text]),
                    [{"category": "blocking", "text": item_text, "status": "open"}],
                ),
                (
                    ReviewVerdict(verdict="REVISE", blocking=[item_text]),
                    [{"category": "blocking", "text": item_text, "status": "open"}],
                ),
            ],
        )

        graph = self._build_loop(researcher, writer, reviewer)
        final = graph.invoke({"query": "test query", "intensity": 3, "revision_count": 0})

        assert calls["reviewer"] == MAX_REVISIONS
        assert calls["researcher"] == MAX_REVISIONS
        assert calls["writer"] == MAX_REVISIONS
        assert final["revision_count"] == MAX_REVISIONS
        # Capped on REVISE, not silently PASSed: the open item and the
        # REVISE verdict survive so the CLI can surface the unresolved end.
        assert final["review_verdict"].verdict == "REVISE"
        assert final["review_items"] == [
            {"category": "blocking", "text": item_text, "status": "open"}
        ]
        # The last writer pass left an honest disposition block in the draft.
        assert "## Changes made" in final["draft_report"]
        assert item_text in final["draft_report"]

    def test_budget_one_is_a_single_audit_with_no_revision_passes(self):
        """max_revisions=1 is the total audit budget, so the first reviewer
        audit ends the loop even on REVISE: the writer runs once (the initial
        draft) and the researcher runs once (the initial research), with no
        revision pass."""
        calls = {"researcher": 0, "writer": 0, "reviewer": 0}
        item_text = "Add pricing details"

        def researcher(state):
            calls["researcher"] += 1
            return {"findings": [Finding(claim="initial claim")]}

        def writer(state):
            calls["writer"] += 1
            return {"draft_report": self.FIRST_DRAFT}

        reviewer = self._reviewer_fake(
            calls,
            [
                (
                    ReviewVerdict(verdict="REVISE", blocking=[item_text]),
                    [{"category": "blocking", "text": item_text, "status": "open"}],
                ),
            ],
        )

        graph = self._build_loop(researcher, writer, reviewer)
        final = graph.invoke(
            {"query": "test query", "intensity": 3, "revision_count": 0, "max_revisions": 1}
        )

        assert calls == {"researcher": 1, "writer": 1, "reviewer": 1}
        assert final["revision_count"] == 1
        assert final["review_verdict"].verdict == "REVISE"

    def test_final_audit_new_issue_retained_in_end_state(self):
        """A NEW issue raised on the final (budget-capped) audit cannot get a
        researcher pass, but it must survive in the end state (REVISE verdict
        + the new open item) so the CLI warning path can surface it."""
        from ora.agents.supervisor import MAX_REVISIONS

        calls = {"researcher": 0, "writer": 0, "reviewer": 0}
        old_item = "Find a 2026 market forecast"
        new_item = "Fabricated statistic on line 12 contradicts its cited source"

        def researcher(state):
            calls["researcher"] += 1
            if calls["researcher"] == 1:
                return {"findings": [Finding(claim="initial claim")]}
            return {
                "review_items": [{"category": "blocking", "text": old_item, "status": "open"}],
                "last_round_new_sources": 0,
                "last_round_new_findings": 0,
            }

        def writer(state):
            calls["writer"] += 1
            if calls["writer"] == 1:
                return {"draft_report": self.FIRST_DRAFT}
            return {
                "draft_report": (
                    self.FIRST_DRAFT
                    + "\n## Changes made\n"
                    + f"- [blocking] {old_item}: attempted, no public data found.\n"
                ),
                "writer_change_notes": (
                    "## Changes made\n"
                    + f"- [blocking] {old_item}: attempted, no public data found.\n"
                ),
            }

        reviewer = self._reviewer_fake(
            calls,
            [
                (
                    ReviewVerdict(verdict="REVISE", blocking=[old_item]),
                    [{"category": "blocking", "text": old_item, "status": "open"}],
                ),
                (
                    ReviewVerdict(verdict="REVISE", blocking=[old_item]),
                    [{"category": "blocking", "text": old_item, "status": "open"}],
                ),
                # Audit 3 (final): reviewer spots a NEW material flaw.
                (
                    ReviewVerdict(verdict="REVISE", blocking=[new_item]),
                    [{"category": "blocking", "text": new_item, "status": "open"}],
                ),
            ],
        )

        graph = self._build_loop(researcher, writer, reviewer)
        final = graph.invoke({"query": "test query", "intensity": 3, "revision_count": 0})

        assert calls["reviewer"] == MAX_REVISIONS
        assert final["revision_count"] == MAX_REVISIONS
        assert final["review_verdict"].verdict == "REVISE"
        assert final["review_verdict"].blocking == [new_item]
        # New issue item is retained as an open review_item in the end state.
        assert final["review_items"] == [
            {"category": "blocking", "text": new_item, "status": "open"}
        ]
