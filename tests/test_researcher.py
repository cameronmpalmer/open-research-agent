"""Tests for researcher agent."""

from ora.agents.researcher import _normalize_url_for_dedupe, generate_search_queries
from ora.state import Finding, ResearchState, Source, SourceExtraction


class TestGenerateSearchQueries:
    def test_intensity_1_generates_fewer_queries(self):
        queries = generate_search_queries("test", intensity=1)
        assert 1 <= len(queries) <= 2

    def test_intensity_2_generates_medium_queries(self):
        queries = generate_search_queries("test query", intensity=2)
        assert 3 <= len(queries) <= 4

    def test_intensity_3_generates_more_queries(self):
        queries = generate_search_queries("test", intensity=3)
        assert len(queries) >= 5

    def test_query_included_in_generated_queries(self):
        queries = generate_search_queries("AI safety", intensity=2)
        assert any("AI safety" in q for q in queries)


class TestNormalizeUrlForDedupe:
    def test_normalizes_http_and_https_to_same_key(self):
        assert _normalize_url_for_dedupe("http://example.com/article") == _normalize_url_for_dedupe(
            "https://example.com/article"
        )

    def test_ignores_fragments_and_trailing_slashes(self):
        assert _normalize_url_for_dedupe(
            "https://example.com/article/#section"
        ) == _normalize_url_for_dedupe("https://example.com/article")


class TestResearcherUsesSearchQueries:
    def test_researcher_prefers_search_queries_over_templates(self, monkeypatch):
        """Round 1 should use state.search_queries when available, not templates."""
        from ora.agents.researcher import researcher_node

        template_calls = []

        def fake_generate_search_queries(query, intensity):
            template_calls.append((query, intensity))
            return ["template_q1", "template_q2"]

        monkeypatch.setattr(
            "ora.agents.researcher.generate_search_queries",
            fake_generate_search_queries,
        )

        def fake_web_search_invoke(input_dict):
            return "[Plan Queries Test](https://example.com/sp1)\n  test snippet\n[Plan Queries 2](https://example.com/sp2)\n  snippet 2"

        import types

        mock_web_search = types.SimpleNamespace(invoke=fake_web_search_invoke)
        # Patch at source module: function-level import in researcher_node
        # does `from ora.tools.search import web_search`
        monkeypatch.setattr("ora.tools.search.web_search", mock_web_search)

        def fake_scrape_page_invoke(input_dict):
            return "Scraped content for " + input_dict["url"]

        mock_scrape = types.SimpleNamespace(invoke=fake_scrape_page_invoke)
        # Patch at source module: function-level import in _scrape_and_collect
        # does `from ora.tools.scrape import scrape_page`
        monkeypatch.setattr("ora.tools.scrape.scrape_page", mock_scrape)

        def fake_extract_and_evaluate(*args, **kwargs):
            return (
                Source(url=kwargs.get("url", "https://example.com"), title=""),
                SourceExtraction(summary="test"),
            )

        # Patch at source module: function-level import in _scrape_and_collect
        monkeypatch.setattr("ora.tools.extract.extract_and_evaluate", fake_extract_and_evaluate)

        state = ResearchState(
            query="test query",
            intensity=3,
            sources=[],
            findings=[],
            messages=[],
        )
        state["search_queries"] = ["plan query one", "plan query two"]

        researcher_node(state)

        assert len(template_calls) == 0

    def test_researcher_does_not_overwrite_search_queries(self, monkeypatch):
        """Researcher return dict must not include search_queries key."""
        import types

        from ora.agents.researcher import researcher_node

        def fake_web_search_invoke(input_dict):
            return "[Test](https://example.com/t1)\n  snippet"

        mock_web_search = types.SimpleNamespace(invoke=fake_web_search_invoke)
        monkeypatch.setattr("ora.tools.search.web_search", mock_web_search)

        def fake_scrape_page_invoke(input_dict):
            return "Scraped content"

        mock_scrape = types.SimpleNamespace(invoke=fake_scrape_page_invoke)
        monkeypatch.setattr("ora.tools.scrape.scrape_page", mock_scrape)

        def fake_extract_and_evaluate(*args, **kwargs):
            return (
                Source(url=kwargs.get("url", "https://example.com"), title=""),
                SourceExtraction(summary="test"),
            )

        monkeypatch.setattr("ora.tools.extract.extract_and_evaluate", fake_extract_and_evaluate)

        original_queries = ["important query one", "important query two"]
        state = ResearchState(
            query="test",
            intensity=3,
            sources=[],
            findings=[],
            messages=[],
            search_queries=original_queries,
        )

        result = researcher_node(state)

        assert "search_queries" not in result


class TestResearcherReviseLoop:
    def test_researcher_runs_gap_queries_on_revise(self, monkeypatch):
        """When min_sources is met but reviewer has issued REVISE,
        the researcher should run gap queries and actually collect new sources.

        The revise_round flag relaxes both the for-loop and _scrape_and_collect
        gates so that sources are collected even though min_sources is already met.
        """
        import types

        from ora.agents.researcher import researcher_node
        from ora.state import ReviewVerdict

        existing_sources = [
            Source(url=f"https://example.com/{i}", title=f"Source {i}") for i in range(15)
        ]
        verdict = ReviewVerdict(
            verdict="REVISE",
            blocking=["Need source on topic X"],
            required=["Add coverage of Y"],
        )

        state = ResearchState(
            query="test query",
            intensity=3,
            sources=existing_sources,
            findings=[Finding(claim="placeholder")],
            messages=[],
            review_verdict=verdict,
            revision_count=1,
            executed_queries=["already searched this"],
        )
        # intensity 3 has min_sources=15. With 15 existing sources, min_sources is met.
        # The while loop must still run because of the REVISE verdict.

        def fake_web_search_invoke(input_dict):
            return "[New Source](https://example.com/new)\n  snippet about topic X"

        mock_web_search = types.SimpleNamespace(invoke=fake_web_search_invoke)
        monkeypatch.setattr("ora.tools.search.web_search", mock_web_search)

        def fake_scrape_page_invoke(input_dict):
            return "Scraped content about " + input_dict["url"]

        mock_scrape = types.SimpleNamespace(invoke=fake_scrape_page_invoke)
        monkeypatch.setattr("ora.tools.scrape.scrape_page", mock_scrape)

        def fake_extract_and_evaluate(*args, **kwargs):
            return (
                Source(url=kwargs.get("url", "https://example.com"), title=""),
                SourceExtraction(summary="test"),
            )

        monkeypatch.setattr("ora.tools.extract.extract_and_evaluate", fake_extract_and_evaluate)
        monkeypatch.setattr(
            "ora.tools.evaluate.evaluate_source", lambda u, t, c, cc: Source(url=u, title=t)
        )

        template_calls = []

        def fake_generate_search_queries(query, intensity):
            template_calls.append((query, intensity))
            return ["fresh_query_1", "fresh_query_2"]

        monkeypatch.setattr(
            "ora.agents.researcher.generate_search_queries",
            fake_generate_search_queries,
        )

        result = researcher_node(state)

        new_sources = result.get("sources", [])
        assert len(new_sources) > 0, (
            "Expected at least one new source on REVISE round when gates are relaxed"
        )

    def test_revise_without_verdict_runs_normally(self, monkeypatch):
        """Without a REVISE verdict, normal behavior: no bypass of min_sources."""
        from ora.agents.researcher import researcher_node

        existing_sources = [
            Source(url=f"https://example.com/{i}", title=f"Source {i}") for i in range(15)
        ]

        state = ResearchState(
            query="test query",
            intensity=3,
            sources=existing_sources,
            findings=[Finding(claim="placeholder")],
            messages=[],
            executed_queries=[],
        )
        # No review_verdict -- normal behavior. min_sources=15 is met.

        result = researcher_node(state)

        new_sources = result.get("sources", [])
        assert len(new_sources) == 0, "Expected zero new sources when no REVISE and min_sources met"

    def test_gap_regeneration_fires_round_one_on_revise(self, monkeypatch):
        """When round 1 queries are all deduped and reviewer feedback exists,
        the gap query regeneration should fire even on round 1."""
        import types

        from ora.agents.researcher import researcher_node
        from ora.state import ReviewVerdict

        verdict = ReviewVerdict(
            verdict="REVISE",
            required=["Find evidence about Z"],
        )

        state = ResearchState(
            query="test query",
            intensity=3,
            sources=[Source(url="https://example.com/1", title="Source 1")],
            findings=[Finding(claim="placeholder")],
            messages=[],
            review_verdict=verdict,
            revision_count=1,
            search_queries=["gap query"],
            executed_queries=["gap query"],
        )

        gap_dynamic_calls = []

        def fake_web_search_invoke(input_dict):
            return "[Regen Source](https://example.com/new)\n  snippet"

        mock_web_search = types.SimpleNamespace(invoke=fake_web_search_invoke)
        monkeypatch.setattr("ora.tools.search.web_search", mock_web_search)

        def fake_scrape_page_invoke(input_dict):
            return "Content"

        mock_scrape = types.SimpleNamespace(invoke=fake_scrape_page_invoke)
        monkeypatch.setattr("ora.tools.scrape.scrape_page", mock_scrape)

        def fake_extract_and_evaluate(*args, **kwargs):
            return (
                Source(url=kwargs.get("url", "https://example.com"), title=""),
                SourceExtraction(summary="test"),
            )

        monkeypatch.setattr("ora.tools.extract.extract_and_evaluate", fake_extract_and_evaluate)

        def fake_generate_gap_queries_dynamic(*args, **kwargs):
            gap_dynamic_calls.append(True)
            return ["fresh gap query from regeneration"]

        monkeypatch.setattr(
            "ora.agents.researcher.generate_gap_queries_dynamic",
            fake_generate_gap_queries_dynamic,
        )

        researcher_node(state)

        assert len(gap_dynamic_calls) > 0, (
            "generate_gap_queries_dynamic should have been called via dedup regeneration"
        )
