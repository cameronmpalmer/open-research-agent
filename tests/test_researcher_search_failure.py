"""A failed search must not be laundered into a finding.

Regression: with ``search.provider: decodo`` and no credentials exported, every
search returns ``"Search error: decodo credentials missing (...)"``. The node
detected the failure and logged it but then fell through as if the search had
simply returned nothing, and the empty-findings guard fabricated a Finding whose
claim was the error text. That made ``findings`` non-empty, so the graph routed
to the writer and produced a blank report from an error message instead of
failing.
"""

import types

import pytest

from ora.agents.researcher import researcher_node
from ora.state import ResearchState

SEARCH_ERROR = "Search error: decodo credentials missing (DECODO_USERNAME/DECODO_PASSWORD)"


def _state(intensity=1, queries=("q1",)):
    state = ResearchState(
        query="test query", intensity=intensity, sources=[], findings=[], messages=[]
    )
    state["search_queries"] = list(queries)
    return state


def _patch_search(monkeypatch, invoke):
    """Install a fake web_search whose .invoke is the given callable."""
    monkeypatch.setattr("ora.tools.search.web_search", types.SimpleNamespace(invoke=invoke))


def test_failed_search_is_not_extracted_as_a_result(monkeypatch):
    """A search error string must not be fed to URL extraction or scraping."""
    _patch_search(monkeypatch, lambda _input: SEARCH_ERROR)

    scraped: list[list[str]] = []

    def fake_scrape_and_collect(urls, *args, **kwargs):
        scraped.append(list(urls))
        return False

    monkeypatch.setattr("ora.agents.researcher._scrape_and_collect", fake_scrape_and_collect)

    with pytest.raises(RuntimeError):
        researcher_node(_state())

    assert all(not urls for urls in scraped), f"error text reached the scraper: {scraped}"


def test_content_check_search_failure_does_not_fabricate_from_error(monkeypatch):
    """Round searches succeed with nothing found, then the final content check
    errors. The error text must not become a finding claim."""

    def fake(_input):
        if _input["query"] == "test query":
            return SEARCH_ERROR
        return "No search results found."

    _patch_search(monkeypatch, fake)

    with pytest.raises(RuntimeError, match="search"):
        researcher_node(_state())
