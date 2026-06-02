"""Tests for supervisor agent."""
import pytest
from ora.agents.supervisor import _extract_search_queries, _search_queries_fence_found
from ora.state import ResearchState


class TestExtractSearchQueries:
    def test_extracts_valid_json(self):
        text = """# Research Plan

Some content here.

```search_queries
["Mem0 agent memory 2026", "Letta vs Cognee comparison"]
```

More text."""
        result = _extract_search_queries(text)
        assert result == ["Mem0 agent memory 2026", "Letta vs Cognee comparison"]

    def test_returns_empty_on_no_fence(self):
        text = "# Research Plan\n\nNo search queries here."
        assert _extract_search_queries(text) == []

    def test_returns_empty_on_bad_json(self):
        text = """```search_queries
{invalid json}
```"""
        assert _extract_search_queries(text) == []

    def test_returns_empty_when_not_list(self):
        text = """```search_queries
{"key": "value"}
```"""
        assert _extract_search_queries(text) == []

    def test_handles_single_quoted_strings(self):
        text = """```search_queries
['single quoted query', 'another one']
```"""
        result = _extract_search_queries(text)
        assert result == ["single quoted query", "another one"]

    def test_handles_blank_line_in_fence(self):
        text = """```search_queries

["query with blank line before"]

```"""
        result = _extract_search_queries(text)
        assert result == ["query with blank line before"]

    def test_returns_empty_on_empty_json_list(self):
        text = """```search_queries
[]
```"""
        assert _extract_search_queries(text) == []

    def test_returns_empty_on_nonstring_items(self):
        text = """```search_queries
[{"not": "a string"}]
```"""
        assert _extract_search_queries(text) == []

    def test_handles_space_between_backticks_and_tag(self):
        """LLMs sometimes insert a space: ``` search_queries (standard markdown)."""
        text = """``` search_queries
["query with space", "another"]
```"""
        result = _extract_search_queries(text)
        assert result == ["query with space", "another"]


class TestSearchQueriesFenceFound:
    def test_fence_found_valid(self):
        text = """```search_queries
["valid"]
```"""
        assert _search_queries_fence_found(text) is True

    def test_fence_found_bad_content(self):
        text = """```search_queries
{invalid}
```"""
        assert _search_queries_fence_found(text) is True

    def test_fence_found_blank_content(self):
        text = """```search_queries

```"""
        assert _search_queries_fence_found(text) is True

    def test_no_fence(self):
        text = "Just a plan with no fence."
        assert _search_queries_fence_found(text) is False

    def test_fence_with_space(self):
        text = """``` search_queries
["query"]
```"""
        assert _search_queries_fence_found(text) is True


def test_plan_node_sets_search_queries(monkeypatch):
    """plan_node should extract search_queries from supervisor response."""
    from ora.agents.supervisor import plan_node

    plan_text = '''# Research Plan

## Subtopics
- Topic 1

```search_queries
["keyword query one", "keyword query two"]
```'''

    monkeypatch.setattr(
        "ora.agents.supervisor._invoke_supervisor",
        lambda prompt: plan_text,
    )

    state = ResearchState(query="test", intensity=3, messages=[])
    result = plan_node(state)

    assert result["search_queries"] == ["keyword query one", "keyword query two"]
    assert result["research_plan"] == plan_text


def test_plan_node_empty_search_queries_on_no_fence(monkeypatch):
    """plan_node should set empty list when supervisor omits search_queries."""
    from ora.agents.supervisor import plan_node

    plan_text = "# Research Plan\n\nNo queries here."

    monkeypatch.setattr(
        "ora.agents.supervisor._invoke_supervisor",
        lambda prompt: plan_text,
    )

    state = ResearchState(query="test", intensity=2, messages=[])
    result = plan_node(state)

    assert result["search_queries"] == []
