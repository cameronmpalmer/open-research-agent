"""Tests for per-source LLM extraction."""
import json
from ora.tools.extract import extract_and_evaluate
from ora.state import SourceExtraction


class FakeLLM:
    """Returns a valid JSON extraction."""
    def __init__(self, extraction_dict):
        self._dict = extraction_dict

    class _Response:
        def __init__(self, content):
            self.content = content

    def invoke(self, _prompt):
        return self._Response(json.dumps(self._dict))


def test_extract_and_evaluate_returns_structured_data(monkeypatch):
    """extract_and_evaluate should return a Source and SourceExtraction
    with the LLM's extracted content."""
    monkeypatch.setattr(
        "ora.tools.extract.get_llm",
        lambda *a, **kw: FakeLLM({
            "summary": "This page recommends the Brilliant Cut Grinder as the best overall option.",
            "key_claims": ["Brilliant Cut Grinder uses Aerospace 7075 Aluminum"],
            "recommendations": ["Buy the Brilliant Cut Grinder for $88"],
            "named_entities": ["Brilliant Cut Grinder", "Grinders For Life"],
            "data_points": ["$88", "7075 Aluminum"],
            "comparisons": ["Brilliant Cut vs Santa Cruz Shredder"],
            "criticisms": ["Expensive compared to budget options"],
            "source_reliability": "High",
            "reliability_rationale": "Established reviewer with methodology."
        }),
    )

    source, extraction = extract_and_evaluate(
        url="https://example.com/article",
        title="Best Grinders 2026",
        content="The Brilliant Cut Grinder is the best weed grinder...",
        query="best weed grinder",
    )

    assert source.overall_reliability == "High"
    assert source.url == "https://example.com/article"
    assert source.title == "Best Grinders 2026"

    assert extraction.summary != ""
    assert "Brilliant Cut Grinder" in extraction.summary
    assert len(extraction.key_claims) >= 1
    assert len(extraction.recommendations) >= 1
    assert len(extraction.named_entities) >= 2
    assert len(extraction.data_points) >= 1
    assert len(extraction.comparisons) >= 1
    assert extraction.source_reliability == "High"


def test_extract_and_evaluate_falls_back_on_json_error(monkeypatch):
    """When the LLM returns invalid JSON, fall back to heuristic evaluation
    and use the raw text as a summary."""

    class BrokenLLM:
        class _Response:
            content = "not valid json at all, just prose"
        def invoke(self, _prompt):
            return self._Response()

    monkeypatch.setattr("ora.tools.extract.get_llm", lambda *a, **kw: BrokenLLM())

    source, extraction = extract_and_evaluate(
        url="https://example.com/broken",
        title="Broken Page",
        content="Some content that was scraped.",
    )

    # Should not crash. Should fall back to heuristic.
    assert source.url == "https://example.com/broken"
    assert extraction.summary != ""
    # Heuristic evaluation should have populated CRAAP dimensions.
    assert 1 <= source.currency <= 5


def test_extract_and_evaluate_falls_back_on_llm_error(monkeypatch):
    """When the LLM call raises an exception, fall back to heuristic."""

    def raise_error(*a, **kw):
        raise RuntimeError("LLM API is down")

    monkeypatch.setattr("ora.tools.extract.get_llm", raise_error)

    source, extraction = extract_and_evaluate(
        url="https://example.com/down",
        title="Down Page",
        content="Some content.",
    )

    assert source.url == "https://example.com/down"
    assert source.overall_reliability in ("Low", "Medium")
    # Extraction should have the fallback summary from raw content.
    assert extraction.summary != ""


def test_extract_and_evaluate_maps_reliability_to_craap(monkeypatch):
    """High reliability should produce high CRAAP dimension scores."""
    monkeypatch.setattr(
        "ora.tools.extract.get_llm",
        lambda *a, **kw: FakeLLM({
            "summary": "Test.",
            "key_claims": [],
            "recommendations": [],
            "named_entities": [],
            "data_points": [],
            "comparisons": [],
            "criticisms": [],
            "source_reliability": "High",
            "reliability_rationale": "Trusted source."
        }),
    )

    source, _ = extract_and_evaluate(
        url="https://example.com/trusted",
        title="Trusted",
        content="Good content here.",
    )

    assert source.overall_reliability == "High"
    # High reliability should map to high individual dimensions.
    assert source.authority >= 4
    assert source.accuracy >= 4


def test_extract_and_evaluate_handles_markdown_fences(monkeypatch):
    """JSON wrapped in ```json fences should still parse correctly."""

    class FencedLLM:
        class _Response:
            content = '```json\n{"summary": "test", "key_claims": ["claim 1"], "recommendations": [], "named_entities": [], "data_points": [], "comparisons": [], "criticisms": [], "source_reliability": "Medium", "reliability_rationale": "ok"}\n```'
        def invoke(self, _prompt):
            return self._Response()

    monkeypatch.setattr("ora.tools.extract.get_llm", lambda *a, **kw: FencedLLM())

    _, extraction = extract_and_evaluate(
        url="https://example.com",
        title="Test",
        content="Content.",
    )

    assert extraction.summary == "test"
    assert extraction.key_claims == ["claim 1"]
