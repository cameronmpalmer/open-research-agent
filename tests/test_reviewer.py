"""Tests for adversarial reviewer agent."""
import json
from ora.state import ReviewVerdict
from ora.agents.reviewer import parse_reviewer_output


class TestParseReviewerOutput:
    def test_parses_pass_verdict(self):
        output = json.dumps({
            "verdict": "PASS",
            "blocking": [],
            "required": ["needs more sources"],
            "suggested": ["add examples"],
            "contradicting_evidence_found": [],
            "confidence_recalibrations": {},
        })
        verdict = parse_reviewer_output(output)
        assert verdict.verdict == "PASS"
        assert len(verdict.blocking) == 0
        assert len(verdict.required) == 1

    def test_parses_revise_verdict(self):
        output = json.dumps({
            "verdict": "REVISE",
            "blocking": ["broken URL: example.com"],
            "required": [],
            "suggested": [],
            "contradicting_evidence_found": ["source X contradicts claim Y"],
            "confidence_recalibrations": {"claim about AI": "Low"},
        })
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
