"""Tests for researcher progress events."""
from ora.agents.researcher import researcher_node
from ora.state import Source


class FakeTool:
    def __init__(self, value):
        self.value = value

    def invoke(self, _args):
        return self.value


def test_researcher_emits_progress_events(monkeypatch):
    events = []

    monkeypatch.setattr(
        "ora.tools.search.web_search",
        FakeTool("1. [Example](https://example.com/article)\n   snippet"),
    )
    monkeypatch.setattr(
        "ora.tools.scrape.scrape_page",
        FakeTool("This is scraped content about Rust vs Go."),
    )
    monkeypatch.setattr(
        "ora.tools.evaluate.evaluate_source",
        lambda url, title, content, source_type: Source(
            url=url, title="Example", source_type=source_type
        ),
    )

    result = researcher_node(
        {"query": "Rust vs Go", "intensity": 1},
        {"configurable": {"progress_callback": events.append}},
    )

    messages = [event["message"] for event in events]
    kinds = [event["kind"] for event in events]

    assert result["findings"]
    assert result["findings"][0].confidence == "Moderate"
    assert "search" in kinds
    assert "scrape" in kinds
    assert "success" in kinds
    assert any("1 search query" in message for message in messages)
    assert any('searching "Rust vs Go"' in message for message in messages)
    assert any("found 1 new URL" in message for message in messages)
    assert any("scraped" in message for message in messages)
    assert any("finished with 1 source" in message for message in messages)


def test_researcher_handles_source_evaluation_failure(monkeypatch):
    events = []

    monkeypatch.setattr(
        "ora.tools.search.web_search",
        FakeTool("1. [Example](https://example.com/article)\n   snippet"),
    )
    monkeypatch.setattr(
        "ora.tools.scrape.scrape_page",
        FakeTool("This is scraped content about Rust vs Go."),
    )

    def fail_evaluate(*_args, **_kwargs):
        raise RuntimeError("bad evaluation")

    monkeypatch.setattr("ora.tools.evaluate.evaluate_source", fail_evaluate)

    result = researcher_node(
        {"query": "Rust vs Go", "intensity": 1},
        {"configurable": {"progress_callback": events.append}},
    )

    messages = [event["message"] for event in events]
    kinds = [event["kind"] for event in events]

    assert result["sources"]
    assert result["findings"]
    assert result["findings"][0].confidence == "Low"
    assert "error" in kinds
    assert any("source evaluation failed" in message for message in messages)


def test_researcher_uses_fallback_finding_when_no_urls_found(monkeypatch):
    events = []

    monkeypatch.setattr(
        "ora.tools.search.web_search",
        FakeTool("No links here, just plain text search results."),
    )

    result = researcher_node(
        {"query": "Rust vs Go", "intensity": 1},
        {"configurable": {"progress_callback": events.append}},
    )

    messages = [event["message"] for event in events]
    kinds = [event["kind"] for event in events]

    assert len(result["findings"]) == 1
    assert result["findings"][0].confidence == "Unknown"
    assert "success" in kinds
    assert any("found 0 new URLs" in message for message in messages)
    assert any("finished with 0 sources and 1 finding" in message for message in messages)


def test_researcher_handles_scrape_error_and_continues(monkeypatch):
    events = []

    monkeypatch.setattr(
        "ora.tools.search.web_search",
        FakeTool("1. [Example](https://example.com/article)\n   snippet"),
    )
    monkeypatch.setattr(
        "ora.tools.scrape.scrape_page",
        FakeTool("Scrape failed for https://example.com/article: {'error': '403'}"),
    )

    result = researcher_node(
        {"query": "Rust vs Go", "intensity": 1},
        {"configurable": {"progress_callback": events.append}},
    )

    messages = [event["message"] for event in events]
    kinds = [event["kind"] for event in events]

    assert result["sources"] == []
    assert len(result["findings"]) == 1
    assert result["findings"][0].confidence == "Unknown"
    assert "error" in kinds
    assert any("scrape failed" in message for message in messages)


def test_researcher_deduplicates_repeated_source_urls(monkeypatch):
    events = []

    monkeypatch.setattr(
        "ora.tools.search.web_search",
        FakeTool("1. [Example](https://example.com/article)\n   snippet"),
    )
    monkeypatch.setattr(
        "ora.tools.scrape.scrape_page",
        FakeTool("This is scraped content about Rust vs Go."),
    )
    monkeypatch.setattr(
        "ora.tools.evaluate.evaluate_source",
        lambda url, title, content, source_type: Source(
            url=url, title="Example", source_type=source_type
        ),
    )

    result = researcher_node(
        {"query": "Rust vs Go", "intensity": 1},
        {"configurable": {"progress_callback": events.append}},
    )

    urls = [source.url for source in result["sources"]]
    assert urls == ["https://example.com/article"]
    assert result["research_status"] == "interim"

    messages = [event["message"] for event in events]
    # After the first scrape, subsequent searches should see the URL pre-filtered as a duplicate
    assert any("1 dups" in message for message in messages)
    assert any("found 0 new URLs" in message for message in messages)


def test_researcher_uses_search_result_titles_for_sources(monkeypatch):
    monkeypatch.setattr(
        "ora.tools.search.web_search",
        FakeTool("1. [Rust vs Go: Which One to Choose?](https://example.com/article)\n   snippet"),
    )
    monkeypatch.setattr(
        "ora.tools.scrape.scrape_page",
        FakeTool("This is scraped content about Rust vs Go."),
    )
    monkeypatch.setattr(
        "ora.tools.evaluate.evaluate_source",
        lambda url, title, content, source_type: Source(
            url=url, title=title, source_type=source_type
        ),
    )

    result = researcher_node({"query": "Rust vs Go", "intensity": 1})

    assert result["sources"][0].title == "Rust vs Go: Which One to Choose?"


def test_researcher_handles_search_failure(monkeypatch):
    events = []

    monkeypatch.setattr(
        "ora.tools.search.web_search",
        FakeTool("Search failed: {'error': 'timeout'}"),
    )

    result = researcher_node(
        {"query": "Rust vs Go", "intensity": 1},
        {"configurable": {"progress_callback": events.append}},
    )

    messages = [event["message"] for event in events]
    kinds = [event["kind"] for event in events]

    assert result["sources"] == []
    assert len(result["findings"]) == 1
    assert result["findings"][0].confidence == "Unknown"
    assert "error" in kinds
    assert any("search failed" in message for message in messages)


def test_researcher_skips_hostile_domains(monkeypatch):
    events = []

    monkeypatch.setattr(
        "ora.tools.search.web_search",
        FakeTool(
            "1. [Reddit thread](https://www.reddit.com/r/rust/comments/12345)\n   snippet\n\n"
            "2. [Medium post](https://medium.com/@user/rust-vs-go)\n   snippet\n\n"
            "3. [Good blog](https://example.com/article)\n   snippet"
        ),
    )
    monkeypatch.setattr(
        "ora.tools.scrape.scrape_page",
        FakeTool("This is scraped content about Rust vs Go."),
    )
    monkeypatch.setattr(
        "ora.tools.evaluate.evaluate_source",
        lambda url, title, content, source_type: Source(
            url=url, title="Example", source_type=source_type
        ),
    )

    result = researcher_node(
        {"query": "Rust vs Go", "intensity": 1},
        {"configurable": {"progress_callback": events.append}},
    )

    messages = [event["message"] for event in events]
    kinds = [event["kind"] for event in events]

    assert len(result["sources"]) == 1
    assert len(result["findings"]) == 1
    assert result["findings"][0].confidence == "Moderate"
    # Hostile domains are pre-filtered before scraping. The summary shows:
    # "found 1 new URL (filtered: 0 dups, 2 hostile)"
    assert any("2 hostile" in message for message in messages)
    assert any("example.com" in message for message in messages)
    assert not any("reddit.com" in message and "scraped" in message for message in messages)
    assert not any("medium.com" in message and "scraped" in message for message in messages)
