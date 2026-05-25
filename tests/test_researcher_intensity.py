"""Tests for intensity level 4-5 behavior."""
from ora.agents.researcher import (
    generate_search_queries,
    generate_gap_queries,
    generate_gap_queries_dynamic,
    _format_reviewer_feedback,
    _normalize_url_for_dedupe,
    researcher_node,
    LEVEL_PARAMS,
)
from ora.state import ReviewVerdict


class TestLevelParams:
    def test_level_1_params(self):
        p = LEVEL_PARAMS[1]
        assert p["min_sources"] == 3
        assert p["max_rounds"] == 5
        assert p["urls_per_query"] == 5
        assert p["scrapes_per_query"] == 3
        assert p["max_content_chars"] == 8000

    def test_level_4_params(self):
        p = LEVEL_PARAMS[4]
        assert p["min_sources"] == 50
        assert p["max_rounds"] == 10
        assert p["urls_per_query"] == 8
        assert p["scrapes_per_query"] == 4
        assert p["max_content_chars"] == 12000

    def test_level_5_params(self):
        p = LEVEL_PARAMS[5]
        assert p["min_sources"] == 100
        assert p["max_rounds"] == 10
        assert p["urls_per_query"] == 10
        assert p["scrapes_per_query"] == 5
        assert p["max_content_chars"] == 16000

    def test_level_4_queries_exist(self):
        queries = generate_search_queries("test", intensity=4)
        assert len(queries) >= 8
        assert "detailed analysis" in " ".join(queries).lower()

    def test_level_5_queries_exist(self):
        queries = generate_search_queries("test", intensity=5)
        assert len(queries) >= 12


class TestGapQueries:
    def test_generates_at_least_min_gap_queries(self):
        queries = generate_gap_queries("AI safety", intensity=4)
        assert len(queries) >= 5

    def test_gap_queries_include_varied_angles(self):
        queries = generate_gap_queries("AI safety", intensity=4)
        joined = " ".join(queries).lower()
        assert any(phrase in joined for phrase in ["detailed", "expert", "recent"])

    def test_gap_queries_level_5_produces_more(self):
        q4 = generate_gap_queries("AI safety", intensity=4)
        q5 = generate_gap_queries("AI safety", intensity=5)
        assert len(q5) >= len(q4)


class TestGapQueriesDynamic:
    """Tests for the LLM-powered dynamic gap query generator."""

    def test_falls_back_to_templates_at_low_intensity(self):
        """At intensity < 3, dynamic should delegate to template-based."""
        dynamic = generate_gap_queries_dynamic(
            query="test query",
            intensity=1,
            sources=[],
            reviewer_feedback="",
            executed_queries=set(),
        )
        templates = generate_gap_queries("test query", intensity=1)
        assert dynamic == templates

    def test_falls_back_to_templates_at_intensity_2(self):
        dynamic = generate_gap_queries_dynamic(
            query="test query",
            intensity=2,
            sources=[],
            reviewer_feedback="",
            executed_queries=set(),
        )
        templates = generate_gap_queries("test query", intensity=2)
        assert dynamic == templates
        assert len(dynamic) >= 2

    def test_falls_back_on_llm_error(self, monkeypatch):
        """When the LLM call fails, fall back to template queries."""
        # At intensity 3, the LLM path would normally be triggered.
        # Simulate an LLM failure.
        monkeypatch.setattr(
            "ora.agents.researcher.get_llm",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("LLM down")),
        )
        queries = generate_gap_queries_dynamic(
            query="test query",
            intensity=3,
            sources=[],
            reviewer_feedback="",
            executed_queries=set(),
        )
        templates = generate_gap_queries("test query", intensity=3)
        assert queries == templates
        assert len(queries) == 3


class TestFormatReviewerFeedback:
    """Tests for _format_reviewer_feedback helper."""

    def test_returns_empty_for_no_verdict(self):
        result = _format_reviewer_feedback({})
        assert result == ""

    def test_formats_blocking_issues(self):
        verdict = ReviewVerdict(
            verdict="REVISE",
            blocking=["Missing data on pricing", "No source for claim X"],
        )
        result = _format_reviewer_feedback({"review_verdict": verdict})
        assert "BLOCKING issues:" in result
        assert "Missing data on pricing" in result
        assert "No source for claim X" in result

    def test_formats_required_and_suggested(self):
        verdict = ReviewVerdict(
            verdict="REVISE",
            required=["Add competitor analysis"],
            suggested=["Consider adding charts"],
        )
        result = _format_reviewer_feedback({"review_verdict": verdict})
        assert "REQUIRED improvements:" in result
        assert "Add competitor analysis" in result
        assert "SUGGESTED improvements:" in result
        assert "Consider adding charts" in result

    def test_skips_empty_fields(self):
        verdict = ReviewVerdict(
            verdict="PASS",
            blocking=[],
            required=[],
            suggested=[],
        )
        result = _format_reviewer_feedback({"review_verdict": verdict})
        assert result == ""


class TestQueryDeduplication:
    """Tests for query deduplication across researcher invocations."""

    def test_skips_already_executed_queries(self, monkeypatch):
        """When executed_queries is in state, round-1 queries in those
        should be skipped and counted as duplicates."""
        from ora.state import Source
        events = []

        class FakeTool:
            def __init__(self, value):
                self.value = value

            def invoke(self, _args):
                return self.value

        monkeypatch.setattr(
            "ora.tools.search.web_search",
            FakeTool("1. [Example](https://example.com/article)\n   snippet"),
        )
        monkeypatch.setattr(
            "ora.tools.scrape.scrape_page",
            FakeTool("Scraped content about the topic."),
        )
        monkeypatch.setattr(
            "ora.tools.evaluate.evaluate_source",
            lambda url, title, content, source_type: Source(
                url=url, title=title or "Example", source_type=source_type
            ),
        )

        # Simulate a reviewer re-entry: researcher was called before and
        # already executed these queries.
        executed = ["test query", "test query latest"]
        state = {
            "query": "test query",
            "intensity": 2,
            "executed_queries": executed,
        }

        result = researcher_node(
            state,
            {"configurable": {"progress_callback": events.append}},
        )

        messages = [event["message"] for event in events]
        # The executed queries should be logged as duplicates.
        assert any("duplicates skipped" in message for message in messages)

        # executed_queries should still include the originals plus any new ones.
        assert "test query" in result["executed_queries"]
        assert "test query latest" in result["executed_queries"]

    def test_executed_queries_persist_in_output(self, monkeypatch):
        """The node output must include executed_queries so they
        accumulate across invocations via the _list_reducer."""
        from ora.state import Source
        events = []

        class FakeTool:
            def __init__(self, value):
                self.value = value

            def invoke(self, _args):
                return self.value

        monkeypatch.setattr(
            "ora.tools.search.web_search",
            FakeTool("1. [Example](https://example.com/article)\n   snippet"),
        )
        monkeypatch.setattr(
            "ora.tools.scrape.scrape_page",
            FakeTool("Scraped content."),
        )
        monkeypatch.setattr(
            "ora.tools.evaluate.evaluate_source",
            lambda url, title, content, source_type: Source(
                url=url, title="Example", source_type=source_type
            ),
        )

        result = researcher_node(
            {"query": "test query", "intensity": 1},
            {"configurable": {"progress_callback": events.append}},
        )

        assert "executed_queries" in result
        assert isinstance(result["executed_queries"], list)
        assert len(result["executed_queries"]) > 0
        # The query itself should be in there.
        assert any("test query" in q for q in result["executed_queries"])
