"""Tests for supervisor agent."""

import json
import re

import pytest

from ora.agents.supervisor import _extract_search_queries, _search_queries_fence_found
from ora.state import ResearchState

# The exact multi-array payload a live supervisor produced: seven independent
# single-element arrays, one per line, instead of one JSON array.
_MULTILINE_SEVEN_QUERIES = [
    "Japan elderly population projection 2020 2050",
    "silver economy market size Japan seniors",
    "Japanese seniors consumption habits trends",
    "active aging spending power Japan retail",
    "geriatric mobility solutions market forecast Japan",
    "senior housing real estate demand Japan",
    "functional food supplement sales Japan elderly",
]


def _seven_line_payload() -> str:
    return "\n".join(f'["{q}"]' for q in _MULTILINE_SEVEN_QUERIES)


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

    def test_extracts_multiline_independent_arrays(self):
        """The real user-facing payload: one JSON array per line."""
        text = f"""```search_queries
{_seven_line_payload()}
```"""
        result = _extract_search_queries(text)
        assert result == _MULTILINE_SEVEN_QUERIES
        assert len(result) == 7

    def test_multiline_arrays_accept_blank_lines_and_indentation(self):
        text = """```search_queries
    ["first query"]

  ["second query", "third query"]

["fourth query"]
```"""
        result = _extract_search_queries(text)
        assert result == ["first query", "second query", "third query", "fourth query"]

    def test_multiline_arrays_preserve_operators_commas_and_non_ascii(self):
        text = (
            "```search_queries\n"
            '["café \\"silver economy\\" size, 2025"]\n'
            '["高齢者 消費 動向", "R&D spend, 2020-2050"]\n'
            "```"
        )
        result = _extract_search_queries(text)
        assert result == [
            'café "silver economy" size, 2025',
            "高齢者 消費 動向",
            "R&D spend, 2020-2050",
        ]

    @pytest.mark.parametrize(
        "payload",
        [
            '["valid query"]\nthis is prose',
            '["valid query"]\n[1, 2]',
            '["valid query"]\n"scalar string"',
            '["valid query"]\n{"key": "value"}',
            '["valid query"]\n[["nested"]]',
            '["valid query"]\n["unclosed',
            '["valid query"]\n...',
            'this is prose\n["valid query"]',
        ],
    )
    def test_multiline_recovery_rejects_whole_block_on_any_invalid_row(self, payload):
        """A single bad row must invalidate the entire block, never a partial list."""
        text = f"```search_queries\n{payload}\n```"
        assert _extract_search_queries(text) == []


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

    plan_text = """# Research Plan

## Subtopics
- Topic 1

```search_queries
["keyword query one", "keyword query two"]
```"""

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


def test_plan_node_extracts_multiline_queries_without_fallback_warning(monkeypatch):
    """plan_node returns the tailored multi-array queries and emits no warning."""
    from ora.agents.supervisor import plan_node

    plan_text = (
        "# Research Plan\n\n## Subtopics\n- Topic 1\n\n"
        "```search_queries\n"
        f"{_seven_line_payload()}\n"
        "```"
    )

    monkeypatch.setattr(
        "ora.agents.supervisor._invoke_supervisor",
        lambda prompt: plan_text,
    )

    events = []
    config = {"configurable": {"progress_callback": events.append}}
    state = ResearchState(query="test", intensity=3, messages=[])
    result = plan_node(state, config)

    assert result["search_queries"] == _MULTILINE_SEVEN_QUERIES
    assert not any(event.get("kind") == "warning" for event in events)
    assert not any("could not be parsed" in event.get("message", "") for event in events)


def test_revise_plan_text_uses_multiline_extractor(monkeypatch):
    """revise_plan_text shares the same extractor, including the recovery path."""
    from ora.agents.supervisor import revise_plan_text

    response = f"# Revised Plan\n\n```search_queries\n{_seven_line_payload()}\n```"

    monkeypatch.setattr(
        "ora.agents.supervisor._invoke_supervisor",
        lambda prompt: response,
    )

    plan, queries = revise_plan_text("query", 3, "old plan", "feedback")

    assert plan == response
    assert queries == _MULTILINE_SEVEN_QUERIES


@pytest.mark.parametrize("count", [1, 3, 16])
def test_plan_and_revise_prompts_require_single_valid_json_array(count):
    """The formatting contract must state one JSON array and ship a valid example."""
    from ora.prompts.supervisor import SUPERVISOR_PLAN_PROMPT, SUPERVISOR_REVISE_PROMPT

    assert "exactly ONE JSON array" in SUPERVISOR_PLAN_PROMPT
    assert "exactly ONE JSON array" in SUPERVISOR_REVISE_PROMPT

    plan_rendered = SUPERVISOR_PLAN_PROMPT.format(query="q", intensity=3, count=count)
    revise_rendered = SUPERVISOR_REVISE_PROMPT.format(
        query="q", intensity=3, plan="p", feedback="f", count=count
    )

    for rendered in (plan_rendered, revise_rendered):
        assert f"your response must contain exactly {count} queries:" in rendered
        example = re.search(r"```search_queries\n(.*?)```", rendered, re.DOTALL)
        assert example is not None
        # Validate strict JSON independently of the tolerant query extractor.
        assert json.loads(example.group(1)) == [
            "keyword query 1",
            "keyword query 2",
            "keyword query 3",
        ]
