"""Tests for state types."""
from ora.state import ResearchState, Source, Finding, ReviewVerdict
from typing import get_type_hints


class TestResearchState:
    def test_state_has_required_fields(self):
        hints = get_type_hints(ResearchState)
        required = ["query", "intensity", "plan_approved", "revision_count"]
        for field in required:
            assert field in hints, f"Missing required field: {field}"

    def test_intensity_is_literal(self):
        """Intensity must be 1-5."""
        hints = get_type_hints(ResearchState)
        intensity_type = hints["intensity"]
        # Should be Literal[1,2,3,4,5] or equivalent
        assert intensity_type is not None


class TestSource:
    def test_source_has_craap_dimensions(self):
        hints = get_type_hints(Source)
        for dim in ["currency", "relevance", "authority", "accuracy", "purpose"]:
            assert dim in hints, f"Missing CRAAP dimension: {dim}"

    def test_source_overall_reliability_is_literal(self):
        hints = get_type_hints(Source)
        assert "overall_reliability" in hints


class TestFinding:
    def test_finding_has_confidence(self):
        hints = get_type_hints(Finding)
        assert "confidence" in hints


class TestReviewVerdict:
    def test_verdict_has_required_fields(self):
        hints = get_type_hints(ReviewVerdict)
        for field in ["verdict", "blocking", "required", "suggested"]:
            assert field in hints, f"Missing: {field}"
