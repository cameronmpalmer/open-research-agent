"""Tests for per-item gap queries and evidence exhaustion in the researcher.

Covers generate_gap_queries_for_items (dedup, shared parser preservation,
no-open-items/LLM-failure fallbacks) and researcher_node revise-pass
behavior: min-met revise rounds search up to three fresh queries; whole-pass
zero-new-sources marks open items evidence_exhausted only after a genuine
multi-query attempt (>= 2 queries searched); single-query rounds keep items
open; progress keeps them open; failed searches still count as attempts.
"""

import types

from ora.agents.researcher import (
    generate_gap_queries,
    generate_gap_queries_for_items,
    researcher_node,
)
from ora.state import Finding, ResearchState, ReviewVerdict, Source, SourceExtraction

OPEN_ITEMS = [
    {"category": "blocking", "text": "Cover pricing details", "status": "open"},
    {"category": "required", "text": "Add 2026 outlook", "status": "open"},
]


class _FakeLLM:
    """Minimal LLM double whose invoke() returns text content."""

    def __init__(self, content: str):
        self.content = content

    def invoke(self, prompt_text):
        return types.SimpleNamespace(content=self.content)


class _FakeTool:
    def __init__(self, value):
        self.value = value

    def invoke(self, _args):
        return self.value


def _revise_state(**overrides) -> ResearchState:
    """State with min_sources already met (15 sources at intensity 3) so a
    REVISE pass runs exactly one forced round, plus two open review items."""
    existing = [Source(url=f"https://example.com/{i}", title=f"Source {i}") for i in range(15)]
    state = ResearchState(
        query="test query",
        intensity=3,
        sources=existing,
        findings=[Finding(claim="placeholder")],
        messages=[],
        review_verdict=ReviewVerdict(
            verdict="REVISE",
            blocking=[OPEN_ITEMS[0]["text"]],
            required=[OPEN_ITEMS[1]["text"]],
        ),
        review_items=[dict(i) for i in OPEN_ITEMS],
        revision_count=1,
        search_queries=["first pass query"],
        executed_queries=["first pass query"],
    )
    state.update(overrides)
    return state


class TestGenerateGapQueriesForItems:
    def test_returns_fresh_per_item_queries_and_dedupes(self, monkeypatch):
        """Parses the LLM's per-item variants, drops executed queries, and
        keeps the list unique."""
        monkeypatch.setattr(
            "ora.agents.researcher.get_llm",
            lambda *a, **kw: _FakeLLM(
                "1. enterprise pricing breakdown\n"
                "- industry outlook report for 2026\n"
                "already executed query\n"
                "dup\n"
                "enterprise pricing breakdown\n"
            ),
        )
        executed = {"already executed query", "older query"}
        result = generate_gap_queries_for_items(
            query="test query",
            intensity=3,
            items=OPEN_ITEMS,
            executed_queries=executed,
            config=None,
        )
        assert "enterprise pricing breakdown" in result
        assert "industry outlook report for 2026" in result
        assert "already executed query" not in result
        assert len(result) == len(set(result))

    def test_parse_preserves_content_leading_digits_and_parenthesis(self, monkeypatch):
        """The shared parser strips list markers but keeps content-leading
        digits (e.g. a year) and strips '1)' numbering to its content."""
        monkeypatch.setattr(
            "ora.agents.researcher.get_llm",
            lambda *a, **kw: _FakeLLM(
                "2026 outlook for the AI market\n1) enterprise pricing\n- supplier cost breakdown\n"
            ),
        )
        result = generate_gap_queries_for_items(
            query="test query",
            intensity=3,
            items=OPEN_ITEMS,
            executed_queries=set(),
            config=None,
        )
        assert "2026 outlook for the AI market" in result
        assert "enterprise pricing" in result
        assert "supplier cost breakdown" in result

    def test_falls_back_to_dynamic_when_no_open_items(self, monkeypatch):
        """With no open items the helper delegates to the flat dynamic
        generator (first-pass behavior)."""
        calls = []

        def fake_dynamic(query, intensity, sources, reviewer_feedback, executed, config=None):
            calls.append((query, intensity, sources, reviewer_feedback, config))
            return ["dynamic fallback query"]

        monkeypatch.setattr("ora.agents.researcher.generate_gap_queries_dynamic", fake_dynamic)
        exhausted_items = [
            {
                "category": "blocking",
                "text": "Cover pricing details",
                "status": "evidence_exhausted",
            }
        ]
        result = generate_gap_queries_for_items(
            query="test query",
            intensity=3,
            items=exhausted_items,
            executed_queries=set(),
            config=None,
        )
        assert result == ["dynamic fallback query"]
        assert len(calls) == 1

    def test_falls_back_when_llm_fails(self, monkeypatch):
        """An LLM failure inside the helper cascades to the dynamic
        generator, whose own LLM call also fails, ending in templates."""
        monkeypatch.setattr(
            "ora.agents.researcher.get_llm",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("LLM down")),
        )
        result = generate_gap_queries_for_items(
            query="test query",
            intensity=3,
            items=OPEN_ITEMS,
            executed_queries=set(),
            config=None,
        )
        templates = generate_gap_queries("test query", intensity=3)
        assert result == templates

    def test_low_intensity_skips_llm(self, monkeypatch):
        """Below intensity 3 the helper never calls the LLM."""
        called = []

        def fail_if_called(*a, **kw):
            called.append(True)
            raise AssertionError("LLM should not be called below intensity 3")

        monkeypatch.setattr("ora.agents.researcher.get_llm", fail_if_called)
        result = generate_gap_queries_for_items(
            query="test query",
            intensity=2,
            items=OPEN_ITEMS,
            executed_queries=set(),
            config=None,
        )
        templates = generate_gap_queries("test query", intensity=2)
        assert result == templates
        assert not called


class TestResearcherNodeExhaustion:
    def test_revise_zero_yield_with_multiple_queries_marks_open_items_exhausted(self, monkeypatch):
        """A revise pass that genuinely searched at least two queries and
        found nothing marks every open item evidence_exhausted and reports
        zero deltas."""
        # min_sources (15 at intensity 3) is NOT met, so the round does not
        # break after one query: two distinct fresh round-1 queries are both
        # searched and both yield nothing, a genuine >= 2-query attempt.
        state = _revise_state(
            sources=[Source(url="https://example.com/0", title="Source 0")],
            search_queries=["plan query one", "plan query two"],
            executed_queries=[],
        )

        # Every search returns zero candidate URLs.
        monkeypatch.setattr(
            "ora.tools.search.web_search",
            _FakeTool("No results matched your search."),
        )
        # Gap rounds only ever re-propose an already-executed query, so the
        # loop terminates right after the two real searches.
        monkeypatch.setattr(
            "ora.agents.researcher.get_llm",
            lambda *a, **kw: _FakeLLM("plan query one\n"),
        )
        # Template fallback is not under test; force it empty so the loop
        # ends cleanly once per-item regeneration finds nothing fresh.
        monkeypatch.setattr(
            "ora.agents.researcher.generate_gap_queries", lambda query, intensity: []
        )

        result = researcher_node(state)

        assert result["last_round_new_sources"] == 0
        assert result["last_round_new_findings"] == 0
        assert result["review_items"]
        assert all(item["status"] == "evidence_exhausted" for item in result["review_items"])

    def test_exhaustion_returns_copies_without_mutating_caller_state(self, monkeypatch):
        """The exhaustion pass returns updated dicts; the caller's shared
        review_items (checkpointed graph state) must not be mutated in place."""
        state = _revise_state(
            sources=[Source(url="https://example.com/0", title="Source 0")],
            search_queries=["plan query one", "plan query two"],
            executed_queries=[],
        )
        monkeypatch.setattr(
            "ora.tools.search.web_search",
            _FakeTool("No results matched your search."),
        )
        monkeypatch.setattr(
            "ora.agents.researcher.get_llm",
            lambda *a, **kw: _FakeLLM("plan query one\n"),
        )
        monkeypatch.setattr(
            "ora.agents.researcher.generate_gap_queries", lambda query, intensity: []
        )

        result = researcher_node(state)

        assert all(item["status"] == "evidence_exhausted" for item in result["review_items"])
        # The caller's objects are untouched: only the returned copies changed.
        assert all(item["status"] == "open" for item in state["review_items"])

    def test_revise_single_item_query_round_keeps_items_open(self, monkeypatch):
        """Regression guard: with only ONE fresh item-targeted query to
        search, the >= 2 exhaustion floor is not reached, so a zero-yield
        single search must NOT exhaust the open items (fewer than two
        attempts this pass; a future REVISE pass should try other variants)."""
        state = _revise_state(
            search_queries=["leftover plan query one"],
            executed_queries=[],
        )
        monkeypatch.setattr(
            "ora.agents.researcher.get_llm",
            lambda *a, **kw: _FakeLLM("item query one\n"),
        )
        searched = []

        def fake_search(args):
            searched.append(args["query"])
            return "No results matched your search."

        monkeypatch.setattr(
            "ora.tools.search.web_search", types.SimpleNamespace(invoke=fake_search)
        )

        result = researcher_node(state)

        # Only the single item query ran; leftover plan queries are skipped.
        assert searched == ["item query one"]
        assert "leftover plan query one" not in searched
        assert result["last_round_new_sources"] == 0
        assert result["review_items"]
        assert all(item["status"] == "open" for item in result["review_items"])

    def test_revise_skips_leftover_plan_queries_and_uses_item_queries(self, monkeypatch):
        """Regression guard (I-1 re-review): on a min-met revise pass with
        open items, leftover untargeted plan search_queries must NOT be
        searched on round 1. The per-item generator runs instead, and items
        exhaust only after >= 2 zero-yield ITEM searches."""
        state = _revise_state(
            search_queries=["leftover plan query one", "leftover plan query two"],
            executed_queries=[],
        )
        llm_calls = []

        def fake_get_llm(*a, **kw):
            llm_calls.append(True)
            return _FakeLLM("item query one\nitem query two\n")

        monkeypatch.setattr("ora.agents.researcher.get_llm", fake_get_llm)

        searched = []

        def fake_search(args):
            searched.append(args["query"])
            return "No results matched your search."

        monkeypatch.setattr(
            "ora.tools.search.web_search", types.SimpleNamespace(invoke=fake_search)
        )

        result = researcher_node(state)

        assert llm_calls, "per-item LLM generator should have run"
        assert len(searched) >= 2
        assert all(q.startswith("item query") for q in searched), (
            f"searches must come from per-item output, not leftover plan queries: {searched}"
        )
        assert all("leftover plan query" not in q for q in searched)
        assert result["last_round_new_sources"] == 0
        assert result["review_items"]
        assert all(item["status"] == "evidence_exhausted" for item in result["review_items"])

    def test_revise_search_failures_count_as_attempts(self, monkeypatch):
        """A failed search is still a genuine attempt: two failed ITEM
        searches on a min-met revise round with open items reach the >= 2
        floor and exhaust the items (zero new sources)."""
        state = _revise_state(
            search_queries=["leftover plan query one", "leftover plan query two"],
            executed_queries=[],
        )
        monkeypatch.setattr(
            "ora.agents.researcher.get_llm",
            lambda *a, **kw: _FakeLLM("item query one\nitem query two\n"),
        )
        # Both searches fail at the tool level (no candidate URLs either).
        monkeypatch.setattr(
            "ora.tools.search.web_search",
            _FakeTool("Search error: rate limited"),
        )

        result = researcher_node(state)

        assert result["last_round_new_sources"] == 0
        assert result["review_items"]
        assert all(item["status"] == "evidence_exhausted" for item in result["review_items"])

    def test_revise_duplicate_regeneration_breaks_without_templates(self, monkeypatch):
        """When even per-item regeneration only repeats executed queries, the
        researcher breaks instead of falling back to generic templates; with
        zero queries executed the open items are NOT exhausted (no genuine
        attempt this pass)."""
        state = _revise_state()

        def fail_if_templates_used(query, intensity):
            raise AssertionError("template fallback must not run on a revise pass with open items")

        monkeypatch.setattr("ora.agents.researcher.generate_gap_queries", fail_if_templates_used)

        # The per-item LLM only echoes the already-executed query.
        monkeypatch.setattr(
            "ora.agents.researcher.get_llm",
            lambda *a, **kw: _FakeLLM("first pass query"),
        )

        def fail_if_searched(_args):
            raise AssertionError("no search should run after the regeneration break")

        monkeypatch.setattr(
            "ora.tools.search.web_search", types.SimpleNamespace(invoke=fail_if_searched)
        )

        result = researcher_node(state)

        assert result["last_round_new_sources"] == 0
        assert result["review_items"]
        assert all(item["status"] == "open" for item in result["review_items"])


class TestResearcherNodeProgress:
    def test_revise_with_new_source_keeps_items_open(self, monkeypatch):
        """A revise pass that finds a new source reports positive deltas and
        leaves the review items open (no exhaustion)."""
        state = _revise_state(
            search_queries=["leftover plan query one"],
            executed_queries=[],
        )
        # Round 1 targets items, not leftover plan queries; the one item
        # query finds a source.
        monkeypatch.setattr(
            "ora.agents.researcher.get_llm",
            lambda *a, **kw: _FakeLLM("enterprise pricing details 2026\n"),
        )

        monkeypatch.setattr(
            "ora.tools.search.web_search",
            _FakeTool("[Pricing source](https://example.com/pricing-2026)\n  snippet"),
        )
        monkeypatch.setattr(
            "ora.tools.scrape.scrape_page",
            _FakeTool("Pricing content that resolves the item."),
        )

        def fake_extract_and_evaluate(*args, **kwargs):
            return (
                Source(url=kwargs.get("url", "https://example.com"), title="Pricing source"),
                SourceExtraction(summary="Enterprise pricing content."),
            )

        monkeypatch.setattr("ora.tools.extract.extract_and_evaluate", fake_extract_and_evaluate)

        result = researcher_node(state)

        assert result["last_round_new_sources"] == 1
        assert result["last_round_new_findings"] == 1
        assert result["review_items"]
        assert all(item["status"] == "open" for item in result["review_items"])
