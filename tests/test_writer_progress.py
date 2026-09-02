"""Tests for writer progress events."""

import pytest

from ora.agents import writer as writer_module
from ora.agents.writer import writer_node
from ora.state import Finding, SourceExtraction


class FakeResponse:
    content = "# Research: Rust vs Go\n\nDraft report body."


class FakeLLM:
    def invoke(self, _prompt):
        return FakeResponse()


class RecordingLLM:
    def __init__(self, error=None):
        self.error = error
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return FakeResponse()


def test_writer_emits_progress_events(monkeypatch):
    events = []
    llm = RecordingLLM()

    monkeypatch.setattr(writer_module, "get_llm", lambda model_name, temperature=0.3: llm)
    monkeypatch.setattr(writer_module, "get_researcher_model", lambda settings: "fake-model")

    result = writer_node(
        {
            "query": "Rust vs Go",
            "intensity": 1,
            "findings": [
                Finding(claim="Rust has memory safety", supporting_sources=["https://example.com"])
            ],
        },
        {"configurable": {"progress_callback": events.append}},
    )

    messages = [event["message"] for event in events]
    kinds = [event["kind"] for event in events]

    assert result["draft_report"].startswith("# Research")
    assert kinds == ["write", "success"]
    assert any("synthesizing report from 1 finding" in message for message in messages)
    assert any("draft generated" in message for message in messages)


def test_writer_handles_llm_failure_with_progress_event(monkeypatch):
    events = []
    llm = RecordingLLM(error=RuntimeError("boom"))

    monkeypatch.setattr(writer_module, "get_llm", lambda model_name, temperature=0.3: llm)
    monkeypatch.setattr(writer_module, "get_researcher_model", lambda settings: "fake-model")

    with pytest.raises(RuntimeError, match="boom"):
        writer_node(
            {
                "query": "Rust vs Go",
                "intensity": 1,
                "findings": [
                    Finding(
                        claim="Rust has memory safety", supporting_sources=["https://example.com"]
                    )
                ],
            },
            {"configurable": {"progress_callback": events.append}},
        )

    messages = [event["message"] for event in events]
    kinds = [event["kind"] for event in events]

    assert "write" in kinds
    assert "error" in kinds
    assert any("LLM call failed" in message for message in messages)


def test_writer_handles_empty_findings_prompt(monkeypatch):
    events = []
    llm = RecordingLLM()

    monkeypatch.setattr(writer_module, "get_llm", lambda model_name, temperature=0.3: llm)
    monkeypatch.setattr(writer_module, "get_researcher_model", lambda settings: "fake-model")

    result = writer_node(
        {
            "query": "Rust vs Go",
            "intensity": 1,
            "findings": [],
        },
        {"configurable": {"progress_callback": events.append}},
    )

    assert result["draft_report"].startswith("# Research")
    assert any("No findings available." in prompt for prompt in llm.prompts)
    assert [event["kind"] for event in events] == ["write", "success"]


def test_format_findings_includes_dict_extraction_data():
    """Dict-serialized findings (from LangGraph re-invocation) should
    surface extraction fields, not silently drop them."""
    from ora.agents.writer import _format_findings_for_prompt

    finding = {
        "claim": "This page recommends the Brilliant Cut Grinder.",
        "confidence": "Moderate",
        "supporting_sources": ["https://example.com"],
        "extraction": {
            "summary": "Best grinder recommendation",
            "key_claims": [
                "BCG uses 7075 Aluminum",
                "BCG is threadless",
            ],
            "recommendations": ["Buy the BCG for $88"],
            "data_points": ["$88", "3 grind plates"],
            "named_entities": [
                "Brilliant Cut Grinder",
                "Grinders For Life",
            ],
            "comparisons": ["BCG vs Santa Cruz Shredder"],
            "criticisms": ["Expensive"],
        },
    }

    formatted = _format_findings_for_prompt([finding])

    # Dict-extracted fields should appear in the output.
    assert "BCG uses 7075 Aluminum" in formatted
    assert "Buy the BCG for $88" in formatted
    assert "$88" in formatted
    assert "Brilliant Cut Grinder" in formatted
    assert "BCG vs Santa Cruz Shredder" in formatted


def test_format_findings_includes_pydantic_extraction_data():
    """Pydantic model findings with SourceExtraction should surface
    extraction fields, not silently drop them."""
    from ora.agents.writer import _format_findings_for_prompt

    extraction = SourceExtraction(
        summary="Best grinder recommendation",
        key_claims=["BCG uses 7075 Aluminum", "BCG is threadless"],
        recommendations=["Buy the BCG for $88"],
        data_points=["$88", "3 grind plates"],
        named_entities=["Brilliant Cut Grinder", "Grinders For Life"],
        comparisons=["BCG vs Santa Cruz Shredder"],
        criticisms=["Expensive"],
        source_reliability="High",
        reliability_rationale="Established reviewer.",
    )

    finding = Finding(
        claim="This page recommends the Brilliant Cut Grinder.",
        confidence="Moderate",
        supporting_sources=["https://example.com"],
        extraction=extraction,
    )

    formatted = _format_findings_for_prompt([finding])

    assert "BCG uses 7075 Aluminum" in formatted
    assert "Buy the BCG for $88" in formatted
    assert "$88" in formatted
    assert "Brilliant Cut Grinder" in formatted
    assert "BCG vs Santa Cruz Shredder" in formatted


def test_writer_uses_researcher_run_override(monkeypatch):
    """A config-layer researcher override (CLI --model) must reach the writer."""
    from ora.config import clear_model_overrides, set_model_override

    captured = {}

    def fake_get_llm(model_name, temperature=0.3):
        captured["model_name"] = model_name
        return FakeLLM()

    monkeypatch.setattr(writer_module, "get_llm", fake_get_llm)
    set_model_override("researcher", "openrouter:qwen/qwen3.7-flash")
    try:
        writer_node({"query": "Rust vs Go", "intensity": 1, "findings": []})
    finally:
        clear_model_overrides()

    assert captured["model_name"] == "openrouter:qwen/qwen3.7-flash"
