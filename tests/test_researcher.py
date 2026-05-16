"""Tests for researcher agent."""
from ora.agents.researcher import generate_search_queries, _normalize_url_for_dedupe


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
