"""Tests for writer progress events."""
import pytest
from ora.agents import writer as writer_module
from ora.agents.writer import writer_node
from ora.state import Finding


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
            "findings": [Finding(claim="Rust has memory safety", supporting_sources=["https://example.com"])],
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
                "findings": [Finding(claim="Rust has memory safety", supporting_sources=["https://example.com"])],
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
