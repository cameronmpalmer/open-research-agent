"""Tests for source evaluation."""
import pytest
from ora.tools.evaluate import evaluate_source, compute_craap_score, rate_reliability


class TestEvaluateSource:
    def test_returns_source_with_ratings(self):
        from ora.state import Source
        result = evaluate_source(
            url="https://example.com/article",
            title="Test Article",
            content="Some factual content about AI research.",
            source_type="blog",
        )
        assert isinstance(result, Source)
        assert result.url == "https://example.com/article"
        assert 1 <= result.currency <= 5
        assert result.overall_reliability in ("High", "Medium", "Low")

    def test_academic_paper_gets_higher_authority(self):
        result = evaluate_source(
            url="https://arxiv.org/abs/1234.5678",
            title="A Novel Approach",
            content="Peer-reviewed research...",
            source_type="academic_paper",
        )
        assert result.authority >= 3

    def test_unknown_source_gets_low_reliability(self):
        result = evaluate_source(
            url="https://random-blog.example.com",
            title="??",
            content="very short",
            source_type="unknown",
        )
        assert result.overall_reliability == "Low"


class TestComputeCraapScore:
    def test_perfect_source_scores_25(self):
        source_data = {
            "currency": 5, "relevance": 5, "authority": 5,
            "accuracy": 5, "purpose": 5
        }
        assert compute_craap_score(source_data) == 25

    def test_poor_source_scores_5(self):
        source_data = {
            "currency": 1, "relevance": 1, "authority": 1,
            "accuracy": 1, "purpose": 1
        }
        assert compute_craap_score(source_data) == 5


class TestRateReliability:
    def test_high_score_is_high_reliability(self):
        assert rate_reliability(22) == "High"

    def test_medium_score_is_medium(self):
        assert rate_reliability(14) == "Medium"

    def test_low_score_is_low(self):
        assert rate_reliability(8) == "Low"
