"""Tests for researcher agent."""
from ora.agents.researcher import generate_search_queries, _normalize_url_for_dedupe
from ora.state import ResearchState, Source, SourceExtraction


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
        assert _normalize_url_for_dedupe(
            "http://example.com/article"
        ) == _normalize_url_for_dedupe("https://example.com/article")

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

        result = researcher_node(state)

        assert len(template_calls) == 0

    def test_researcher_does_not_overwrite_search_queries(self, monkeypatch):
        """Researcher return dict must not include search_queries key."""
        from ora.agents.researcher import researcher_node
        import types

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
