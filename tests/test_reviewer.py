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

    def test_parses_unresolvable_gaps(self):
        output = json.dumps(
            {
                "verdict": "REVISE",
                "blocking": [],
                "required": [],
                "suggested": [],
                "contradicting_evidence_found": [],
                "confidence_recalibrations": {},
                "unresolvable_gaps": ["x"],
            }
        )
        verdict = parse_reviewer_output(output)
        assert verdict.unresolvable_gaps == ["x"]

    def test_unresolvable_gaps_defaults_to_empty(self):
        verdict = parse_reviewer_output('{"verdict": "PASS"}')
        assert verdict.unresolvable_gaps == []


class TestReviewItemsFromVerdict:
    def test_blocking_and_required_become_open_items(self):
        from ora.agents.reviewer import review_items_from_verdict
        from ora.state import ReviewVerdict

        verdict = ReviewVerdict(
            verdict="REVISE",
            blocking=["b1"],
            required=["r1"],
            suggested=["s1"],
        )
        assert review_items_from_verdict(verdict) == [
            {"category": "blocking", "text": "b1", "status": "open"},
            {"category": "required", "text": "r1", "status": "open"},
        ]

    def test_empty_verdict_yields_no_items(self):
        from ora.agents.reviewer import review_items_from_verdict
        from ora.state import ReviewVerdict

        assert review_items_from_verdict(ReviewVerdict()) == []


class _PassLLM:
    class _Response:
        content = (
            '{"verdict": "PASS", "blocking": [], "required": [], "suggested": [],'
            ' "contradicting_evidence_found": [], "confidence_recalibrations": {}}'
        )

    def invoke(self, _prompt):
        return self._Response()


class _ReviseLLM:
    class _Response:
        content = (
            '{"verdict": "REVISE", "blocking": ["broken URL: example.com"],'
            ' "required": ["add more sources"], "suggested": ["add examples"],'
            ' "contradicting_evidence_found": [], "confidence_recalibrations": {},'
            ' "unresolvable_gaps": []}'
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


class TestReviewerNodeReviewItems:
    def test_revise_verdict_populates_review_items_without_resetting_deltas(self, monkeypatch):
        """A REVISE verdict must surface open review_items and reset per-round deltas."""
        from ora.agents import reviewer as reviewer_module
        from ora.agents.reviewer import reviewer_node

        monkeypatch.setattr(
            reviewer_module,
            "get_llm",
            lambda model_name, temperature=0.2: _ReviseLLM(),
        )

        result = reviewer_node(
            {
                "query": "Rust vs Go",
                "draft_report": "# Research\nbody",
                "revision_count": 0,
            }
        )

        assert result["review_verdict"].verdict == "REVISE"
        assert result["review_items"] == [
            {"category": "blocking", "text": "broken URL: example.com", "status": "open"},
            {"category": "required", "text": "add more sources", "status": "open"},
        ]
        # Deltas are deliberately NOT reset here (routing reads them right
        # after this node); the researcher owns last_round_new_*.
        assert "last_round_new_sources" not in result


class _AuditRecordingLLM:
    """Records every prompt and returns canned content on each invoke."""

    def __init__(self, contents):
        self.prompts = []
        self._contents = list(contents)

    def invoke(self, prompt):
        self.prompts.append(prompt)
        content = self._contents.pop(0)
        return type("_Response", (), {"content": content})()


PASS_WITH_GAPS_JSON = (
    '{"verdict": "PASS", "blocking": [], "required": [], "suggested": [],'
    ' "contradicting_evidence_found": [], "confidence_recalibrations": {},'
    ' "unresolvable_gaps": ["2026 outlook unavailable in sources"]}'
)

REVISE_ACTIONABLE_JSON = (
    '{"verdict": "REVISE", "blocking": ["Add pricing details: disposition says resolved'
    ' but the report still omits pricing"], "required": [], "suggested": [],'
    ' "contradicting_evidence_found": [], "confidence_recalibrations": {},'
    ' "unresolvable_gaps": []}'
)


class TestReviewerNodeRevisionAudit:
    """Re-audits receive the previous items with statuses, the writer's change
    notes, and the new-source count; the verdict drives which items stay open."""

    def _audit_state(self, **overrides) -> dict:
        state = {
            "query": "Rust vs Go",
            "draft_report": "# Research\nrevised body",
            "review_items": [
                {"category": "blocking", "text": "Add pricing details", "status": "open"},
                {
                    "category": "required",
                    "text": "2026 outlook unavailable in sources",
                    "status": "evidence_exhausted",
                },
            ],
            "writer_change_notes": (
                "## Changes made\n"
                "- [blocking] Add pricing details: resolved with new source.\n"
                "- [required] 2026 outlook unavailable in sources: documented as a gap.\n"
            ),
            "last_round_new_sources": 3,
            "last_round_new_findings": 2,
            "revision_count": 1,
        }
        state.update(overrides)
        return state

    def test_prompt_carries_items_notes_and_new_source_count(self, monkeypatch):
        """The re-audit prompt must include every previous item with category
        and status, the writer's change notes, and the new-source count."""
        from ora.agents import reviewer as reviewer_module
        from ora.agents.reviewer import reviewer_node

        llm = _AuditRecordingLLM([PASS_WITH_GAPS_JSON])
        monkeypatch.setattr(
            reviewer_module,
            "get_llm",
            lambda model_name, temperature=0.2: llm,
        )

        result = reviewer_node(self._audit_state())

        prompt = llm.prompts[-1]
        assert "- [blocking] (open) Add pricing details" in prompt
        assert "- [required] (evidence_exhausted) 2026 outlook unavailable in sources" in prompt
        assert "Add pricing details: resolved with new source." in prompt
        assert "2026 outlook unavailable in sources: documented as a gap." in prompt
        assert "NEW_SOURCES_SINCE_LAST_AUDIT: 3" in prompt

        # PASS with unresolvable_gaps parses; the exhausted gap is accepted
        # (blocking/required empty) so no new open items are produced.
        assert result["review_verdict"].verdict == "PASS"
        assert result["review_verdict"].unresolvable_gaps == ["2026 outlook unavailable in sources"]
        assert result["review_items"] == []
        assert result["revision_count"] == 2

    def test_revise_for_actionable_unaddressed_item(self, monkeypatch):
        """When the writer's disposition claims an item resolved but the report
        does not reflect it, the reviewer REVISEses and the item stays open."""
        from ora.agents import reviewer as reviewer_module
        from ora.agents.reviewer import reviewer_node

        llm = _AuditRecordingLLM([REVISE_ACTIONABLE_JSON])
        monkeypatch.setattr(
            reviewer_module,
            "get_llm",
            lambda model_name, temperature=0.2: llm,
        )

        result = reviewer_node(self._audit_state())

        assert result["review_verdict"].verdict == "REVISE"
        assert result["review_verdict"].unresolvable_gaps == []
        # The re-raised actionable item becomes the next pass's open item.
        assert result["review_items"] == [
            {
                "category": "blocking",
                "text": (
                    "Add pricing details: disposition says resolved but the report"
                    " still omits pricing"
                ),
                "status": "open",
            }
        ]
        assert result["revision_count"] == 2

    def test_first_audit_defaults_context_placeholders(self, monkeypatch):
        """A first audit (no review_items/notes/deltas) renders the context
        placeholders with their default values."""
        from ora.agents import reviewer as reviewer_module
        from ora.agents.reviewer import reviewer_node

        llm = _AuditRecordingLLM([PASS_WITH_GAPS_JSON])
        monkeypatch.setattr(
            reviewer_module,
            "get_llm",
            lambda model_name, temperature=0.2: llm,
        )

        reviewer_node({"query": "Rust vs Go", "draft_report": "# Research\nbody"})

        prompt = llm.prompts[-1]
        assert "(first audit)" in prompt
        assert "(no revision notes)" in prompt
        assert "NEW_SOURCES_SINCE_LAST_AUDIT: 0" in prompt
