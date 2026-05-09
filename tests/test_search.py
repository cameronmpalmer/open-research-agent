"""Tests for Firecrawl search tool."""
from ora.tools.search import web_search


class TestSearchTool:
    def test_tool_has_name(self):
        assert web_search.name == "web_search"

    def test_tool_has_description(self):
        assert "search" in web_search.description.lower()
