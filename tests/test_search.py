"""Tests for Firecrawl search tool."""
import pytest
from unittest.mock import patch
from ora.tools.search import create_search_tool


class TestSearchTool:
    def test_tool_creation(self):
        tool = create_search_tool(api_key="test-key")
        assert tool.name == "web_search"

    def test_search_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        with pytest.raises(ValueError, match="api_key"):
            create_search_tool(api_key="")
