"""Tests for adversarial reviewer agent."""

import json

from ora.agents.reviewer import parse_reviewer_output


class TestParseReviewerOutput:
    def test_parses_pass_verdict(self):
        output = json.dumps(
            {
                "verdict": "PASS",
                "blocking": [],
                "required": ["needs more sources"],
                "suggested": ["add examples"],
                "contradicting_evidence_found": [],
                "confidence_recalibrations": {},
            }
        )
        verdict = parse_reviewer_output(output)
        assert verdict.verdict == "PASS"
        assert len(verdict.blocking) == 0
        assert len(verdict.required) == 1

    def test_parses_revise_verdict(self):
        output = json.dumps(
            {
                "verdict": "REVISE",
                "blocking": ["broken URL: example.com"],
                "required": [],
                "suggested": [],
                "contradicting_evidence_found": ["source X contradicts claim Y"],
                "confidence_recalibrations": {"claim about AI": "Low"},
            }
        )
        verdict = parse_reviewer_output(output)
        assert verdict.verdict == "REVISE"
        assert len(verdict.blocking) == 1
        assert verdict.contradicting_evidence_found == ["source X contradicts claim Y"]

    def test_handles_malformed_json(self):
        verdict = parse_reviewer_output("not valid json {")
        assert verdict.verdict == "REVISE"
        assert "parsing failed" in verdict.blocking[0].lower()

    def test_parses_json_in_markdown_fence(self):
        output = '```json\n{"verdict": "PASS", "blocking": [], "required": [], "suggested": [], "contradicting_evidence_found": [], "confidence_recalibrations": {}}\n```'
        verdict = parse_reviewer_output(output)
        assert verdict.verdict == "PASS"


class _PassLLM:
    class _Response:
        content = (
            '{"verdict": "PASS", "blocking": [], "required": [], "suggested": [],'
            ' "contradicting_evidence_found": [], "confidence_recalibrations": {}}'
        )

    def invoke(self, _prompt):
        return self._Response()


class TestReviewerNodeModelOverride:
    def test_reviewer_uses_reviewer_run_override(self, monkeypatch):
        """A config-layer reviewer override (CLI --reviewer-model) must reach the reviewer."""
        from ora.agents import reviewer as reviewer_module
        from ora.agents.reviewer import reviewer_node
        from ora.config import clear_model_overrides, set_model_override

        captured = {}

        def fake_get_llm(model_name, temperature=0.2):
            captured["model_name"] = model_name
            return _PassLLM()

        monkeypatch.setattr(reviewer_module, "get_llm", fake_get_llm)
        set_model_override("reviewer", "openrouter:deepseek/deepseek-v4-pro")
        try:
            reviewer_node({"query": "Rust vs Go", "draft_report": "# Research\nbody"})
        finally:
            clear_model_overrides()

        assert captured["model_name"] == "openrouter:deepseek/deepseek-v4-pro"
