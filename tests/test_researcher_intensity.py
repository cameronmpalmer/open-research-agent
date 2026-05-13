"""Tests for intensity level 4-5 behavior."""
from ora.agents.researcher import (
    generate_search_queries,
    generate_gap_queries,
    LEVEL_PARAMS,
)


class TestLevelParams:
    def test_level_1_params(self):
        p = LEVEL_PARAMS[1]
        assert p["min_sources"] == 3
        assert p["max_rounds"] == 1
        assert p["urls_per_query"] == 3
        assert p["scrapes_per_query"] == 2
        assert p["max_content_chars"] == 8000

    def test_level_4_params(self):
        p = LEVEL_PARAMS[4]
        assert p["min_sources"] == 50
        assert p["max_rounds"] == 3
        assert p["urls_per_query"] == 8
        assert p["scrapes_per_query"] == 4
        assert p["max_content_chars"] == 12000

    def test_level_5_params(self):
        p = LEVEL_PARAMS[5]
        assert p["min_sources"] == 100
        assert p["max_rounds"] == 4
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
