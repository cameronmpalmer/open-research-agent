"""Tests for adversarial reviewer agent."""

import json

import pytest

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


class TestFoldRepeatedExhausted:
    """The hard repeat-raise guard folds a blocking/required item whose
    normalized text was previously evidence_exhausted ONLY when the writer
    acknowledged it in the change notes or this is the final audit."""

    def _verdict(self, **overrides):
        from ora.state import ReviewVerdict

        fields = {
            "verdict": "REVISE",
            "blocking": [],
            "required": [],
            "suggested": [],
            "unresolvable_gaps": [],
        }
        fields.update(overrides)
        return ReviewVerdict(**fields)

    def _exhausted(self, text="X"):
        return [{"category": "blocking", "text": text, "status": "evidence_exhausted"}]

    def _notes_for(self, text="X", marker="documented as a gap"):
        return f"## Changes made\n- [blocking] {text}: {marker}.\n"

    def test_exhausted_and_documented_folds_out_of_blocking(self):
        from ora.agents.reviewer import _fold_repeated_exhausted

        verdict = _fold_repeated_exhausted(
            self._verdict(blocking=["X"]),
            self._exhausted(),
            self._notes_for(),
            is_final_audit=False,
        )

        assert verdict.blocking == []
        assert verdict.unresolvable_gaps == ["X"]

    @pytest.mark.parametrize("marker", ["documented as a gap", "partially addressed", "resolved"])
    def test_exhausted_acknowledged_with_any_disposition_marker_folds(self, marker):
        from ora.agents.reviewer import _fold_repeated_exhausted

        verdict = _fold_repeated_exhausted(
            self._verdict(blocking=["X"]),
            self._exhausted(),
            self._notes_for(marker=marker),
            is_final_audit=False,
        )

        assert verdict.blocking == []
        assert verdict.unresolvable_gaps == ["X"]

    def test_exhausted_undocumented_not_final_stays_blocking(self):
        """The exhausted-but-undocumented documentation-gap REVISE must be
        able to reach the writer: without acknowledgment and before the final
        audit the re-raise stays in blocking."""
        from ora.agents.reviewer import _fold_repeated_exhausted

        verdict = _fold_repeated_exhausted(
            self._verdict(blocking=["X"]),
            self._exhausted(),
            writer_change_notes="",
            is_final_audit=False,
        )

        assert verdict.blocking == ["X"]
        assert verdict.unresolvable_gaps == []

    def test_acknowledgment_of_a_different_item_does_not_fold(self):
        """Notes that respond to a different item are not acknowledgment of X."""
        from ora.agents.reviewer import _fold_repeated_exhausted

        verdict = _fold_repeated_exhausted(
            self._verdict(blocking=["X"]),
            self._exhausted(),
            writer_change_notes=self._notes_for(text="Y"),
            is_final_audit=False,
        )

        assert verdict.blocking == ["X"]
        assert verdict.unresolvable_gaps == []

    def test_marker_less_mention_does_not_acknowledge(self):
        """A marker-less mention of the item text is not acknowledgment: a
        near-duplicate in the notes must not fold a non-final, undocumented
        item out of blocking/required into gaps."""
        from ora.agents.reviewer import _fold_repeated_exhausted

        verdict = _fold_repeated_exhausted(
            self._verdict(blocking=["Add pricing"]),
            self._exhausted("Add pricing"),
            writer_change_notes="## Changes made\n- Add pricing details: no action taken.\n",
            is_final_audit=False,
        )

        assert verdict.blocking == ["Add pricing"]
        assert verdict.unresolvable_gaps == []

    @pytest.mark.parametrize(
        "notes",
        [
            "## Changes made\n- [blocking] X: unresolved - no public data found.\n",
            "## Changes made\n- [required] X: not resolved - no public data found.\n",
            "## Changes made\n- [blocking] X: never resolved - no public data found.\n",
        ],
    )
    def test_negated_disposition_is_not_acknowledgment(self, notes):
        """A negated disposition ("unresolved", "not resolved", "never
        resolved") must not satisfy the "resolved" marker. The previous
        substring test let "unresolved" match "resolved" and folded an item the
        writer had explicitly NOT acknowledged, ending the loop before the
        limitation was documented."""
        from ora.agents.reviewer import _fold_repeated_exhausted

        verdict = _fold_repeated_exhausted(
            self._verdict(blocking=["X"]),
            self._exhausted(),
            writer_change_notes=notes,
            is_final_audit=False,
        )

        assert verdict.blocking == ["X"]
        assert verdict.unresolvable_gaps == []

    def test_superset_restatement_is_not_acknowledgment(self):
        """A disposition line that restates a longer text (a superset) must not
        acknowledge the shorter item: the restatement has to be the exact item
        text, so "Add pricing details" does not acknowledge "Add pricing"."""
        from ora.agents.reviewer import _fold_repeated_exhausted

        verdict = _fold_repeated_exhausted(
            self._verdict(blocking=["Add pricing"]),
            self._exhausted("Add pricing"),
            writer_change_notes="## Changes made\n- [blocking] Add pricing details: resolved.\n",
            is_final_audit=False,
        )

        assert verdict.blocking == ["Add pricing"]
        assert verdict.unresolvable_gaps == []

    def test_acknowledgment_requires_the_disposition_after_a_colon(self):
        """The marker must sit in the disposition that follows the restated
        item, not merely anywhere on a line that happens to contain the text."""
        from ora.agents.reviewer import _notes_acknowledge_item

        assert _notes_acknowledge_item("- [blocking] X: resolved.\n", "X") is True
        assert _notes_acknowledge_item("- [blocking] X: unresolved.\n", "X") is False
        assert _notes_acknowledge_item("- [blocking] X resolved elsewhere.\n", "X") is False
        assert _notes_acknowledge_item("- [blocking] X details: resolved.\n", "X") is False

    def test_item_text_containing_a_colon_is_acknowledged(self):
        """An item whose own text contains a colon must still be acknowledged:
        the delimiter is the colon that follows the whole restated item, not the
        first colon on the line."""
        from ora.agents.reviewer import _fold_repeated_exhausted

        verdict = _fold_repeated_exhausted(
            self._verdict(blocking=["Add pricing: US and EU"]),
            self._exhausted("Add pricing: US and EU"),
            writer_change_notes=(
                "## Changes made\n- [blocking] Add pricing: US and EU: resolved - added sources.\n"
            ),
            is_final_audit=False,
        )

        assert verdict.blocking == []
        assert verdict.unresolvable_gaps == ["Add pricing: US and EU"]

    def test_numbered_disposition_marker_is_acknowledged(self):
        """Numbered list markers ("1." / "1)") are valid disposition lines."""
        from ora.agents.reviewer import _notes_acknowledge_item

        assert _notes_acknowledge_item("1. [blocking] X: resolved.\n", "X") is True
        assert _notes_acknowledge_item("1) [blocking] X: resolved.\n", "X") is True

    def test_status_token_in_restatement_is_acknowledged(self):
        """The writer is shown review items as "- [category] (status) text", so
        restating that line verbatim (including the (status) token) must still
        count as acknowledgment, not silently stay blocking."""
        from ora.agents.reviewer import _fold_repeated_exhausted

        verdict = _fold_repeated_exhausted(
            self._verdict(blocking=["Add pricing"]),
            self._exhausted("Add pricing"),
            writer_change_notes="## Changes made\n- [blocking] (open) Add pricing: resolved.\n",
            is_final_audit=False,
        )
        assert verdict.blocking == []
        assert verdict.unresolvable_gaps == ["Add pricing"]

        documented = _fold_repeated_exhausted(
            self._verdict(required=["2026 outlook unavailable in sources"]),
            [
                {
                    "category": "required",
                    "text": "2026 outlook unavailable in sources",
                    "status": "evidence_exhausted",
                }
            ],
            writer_change_notes=(
                "## Changes made\n- [required] (evidence_exhausted) 2026 outlook"
                " unavailable in sources: documented as a gap.\n"
            ),
            is_final_audit=False,
        )
        assert documented.required == []
        assert documented.unresolvable_gaps == ["2026 outlook unavailable in sources"]

    @pytest.mark.parametrize(
        "disposition",
        ["cannot be resolved", "not yet resolved", "not  resolved", "could not resolve"],
    )
    def test_extended_negation_is_not_acknowledgment(self, disposition):
        """Broader negations must not satisfy the "resolved" marker: the
        disposition has to begin with a declared disposition token."""
        from ora.agents.reviewer import _notes_acknowledge_item

        assert _notes_acknowledge_item(f"- [blocking] X: {disposition}.\n", "X") is False

    def test_exhausted_undocumented_final_audit_folds(self):
        from ora.agents.reviewer import _fold_repeated_exhausted

        verdict = _fold_repeated_exhausted(
            self._verdict(blocking=["X"]),
            self._exhausted(),
            writer_change_notes="",
            is_final_audit=True,
        )

        assert verdict.blocking == []
        assert verdict.unresolvable_gaps == ["X"]

    def test_exhausted_documented_final_audit_folds(self):
        from ora.agents.reviewer import _fold_repeated_exhausted

        verdict = _fold_repeated_exhausted(
            self._verdict(blocking=["X"]),
            self._exhausted(),
            self._notes_for(),
            is_final_audit=True,
        )

        assert verdict.blocking == []
        assert verdict.unresolvable_gaps == ["X"]

    def test_open_previous_item_is_not_folded(self):
        from ora.agents.reviewer import _fold_repeated_exhausted

        previous = [{"category": "blocking", "text": "X", "status": "open"}]
        verdict = _fold_repeated_exhausted(
            self._verdict(blocking=["X"]), previous, self._notes_for(), is_final_audit=True
        )

        assert verdict.blocking == ["X"]
        assert verdict.unresolvable_gaps == []

    def test_normalization_matches_case_and_whitespace(self):
        from ora.agents.reviewer import _fold_repeated_exhausted

        previous = [
            {"category": "required", "text": "  Add Pricing Data  ", "status": "evidence_exhausted"}
        ]
        verdict = _fold_repeated_exhausted(
            self._verdict(blocking=["add pricing data"]),
            previous,
            writer_change_notes="## Changes made\n- [required] Add Pricing Data: documented as a gap.\n",
            is_final_audit=False,
        )

        assert verdict.blocking == []
        assert verdict.unresolvable_gaps == ["add pricing data"]

    def test_unexhausted_new_items_are_kept_alongside_folded_ones(self):
        from ora.agents.reviewer import _fold_repeated_exhausted

        verdict = _fold_repeated_exhausted(
            self._verdict(blocking=["X"], required=["NEW fabricated statistic"]),
            self._exhausted(),
            self._notes_for(),
            is_final_audit=False,
        )

        assert verdict.blocking == []
        assert verdict.required == ["NEW fabricated statistic"]
        assert verdict.unresolvable_gaps == ["X"]

    def test_no_previous_items_returns_verdict_unchanged(self):
        from ora.agents.reviewer import _fold_repeated_exhausted

        verdict = _fold_repeated_exhausted(
            self._verdict(blocking=["X"]), [], self._notes_for(), is_final_audit=True
        )
        assert verdict.blocking == ["X"]
        assert verdict.unresolvable_gaps == []

    def test_existing_model_gaps_are_preserved_and_fold_appends(self):
        from ora.agents.reviewer import _fold_repeated_exhausted

        verdict = _fold_repeated_exhausted(
            self._verdict(blocking=["X"], unresolvable_gaps=["existing gap"]),
            self._exhausted(),
            self._notes_for(),
            is_final_audit=False,
        )

        assert verdict.blocking == []
        assert verdict.unresolvable_gaps == ["existing gap", "X"]

    def test_text_present_in_both_blocking_and_required_folds_once(self):
        from ora.agents.reviewer import _fold_repeated_exhausted

        verdict = _fold_repeated_exhausted(
            self._verdict(blocking=["X"], required=["X"]),
            self._exhausted(),
            self._notes_for(),
            is_final_audit=False,
        )

        assert verdict.blocking == []
        assert verdict.required == []
        assert verdict.unresolvable_gaps == ["X"]

    def test_fold_does_not_duplicate_model_supplied_gap(self):
        """Folding never appends a text the model already accepted into gaps."""
        from ora.agents.reviewer import _fold_repeated_exhausted

        verdict = _fold_repeated_exhausted(
            self._verdict(blocking=["X"], required=["X"], unresolvable_gaps=["X"]),
            self._exhausted(),
            self._notes_for(),
            is_final_audit=False,
        )

        assert verdict.blocking == []
        assert verdict.required == []
        assert verdict.unresolvable_gaps == ["X"]


class TestCarryExhaustedItems:
    """Previously-exhausted items the reviewer neither re-raised nor accepted
    into gaps carry forward as evidence_exhausted records so history survives
    into the next audit. Acknowledgment by the writer alone does not drop the
    record: only acceptance into this verdict's unresolvable_gaps does."""

    def _exhausted(self, text="X"):
        return [{"category": "required", "text": text, "status": "evidence_exhausted"}]

    def _notes_for(self, text="X"):
        return f"## Changes made\n- [required] {text}: documented as a gap.\n"

    def test_unacknowledged_not_rerisen_item_is_carried(self):
        from ora.agents.reviewer import _carry_exhausted_items
        from ora.state import ReviewVerdict

        carried = _carry_exhausted_items(
            ReviewVerdict(verdict="PASS"),
            self._exhausted(),
            writer_change_notes="",
        )

        assert carried == [{"category": "required", "text": "X", "status": "evidence_exhausted"}]

    def test_item_accepted_into_gaps_is_not_carried(self):
        from ora.agents.reviewer import _carry_exhausted_items
        from ora.state import ReviewVerdict

        carried = _carry_exhausted_items(
            ReviewVerdict(verdict="PASS", unresolvable_gaps=["X"]),
            self._exhausted(),
            writer_change_notes="",
        )

        assert carried == []

    def test_rerisen_item_is_not_carried(self):
        """A re-raise is handled by the fold decision (folded or kept open),
        so it is not also carried as an exhausted record."""
        from ora.agents.reviewer import _carry_exhausted_items
        from ora.state import ReviewVerdict

        carried = _carry_exhausted_items(
            ReviewVerdict(verdict="REVISE", blocking=["X"]),
            self._exhausted(),
            writer_change_notes="",
        )

        assert carried == []

    def test_acknowledged_item_still_carried_when_not_accepted(self):
        """Acknowledgment alone must not drop the record: if the reviewer did
        not also accept the item into unresolvable_gaps, it is still carried
        as evidence_exhausted so it cannot vanish from structured state."""
        from ora.agents.reviewer import _carry_exhausted_items
        from ora.state import ReviewVerdict

        carried = _carry_exhausted_items(
            ReviewVerdict(verdict="PASS"),
            self._exhausted(),
            writer_change_notes=self._notes_for(),
        )

        assert carried == [{"category": "required", "text": "X", "status": "evidence_exhausted"}]

    def test_open_previous_items_are_never_carried(self):
        from ora.agents.reviewer import _carry_exhausted_items
        from ora.state import ReviewVerdict

        previous = [
            {"category": "blocking", "text": "X", "status": "open"},
            {"category": "required", "text": "Y", "status": "evidence_exhausted"},
        ]
        carried = _carry_exhausted_items(
            ReviewVerdict(verdict="PASS"),
            previous,
            writer_change_notes="",
        )

        assert carried == [{"category": "required", "text": "Y", "status": "evidence_exhausted"}]


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
        """A REVISE verdict must surface open review_items; per-round deltas are owned by the researcher."""
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
            },
            # The writer acknowledged the 2026 item but the reviewer neither
            # re-raised nor accepted it: the record is carried, not dropped.
            {
                "category": "required",
                "text": "2026 outlook unavailable in sources",
                "status": "evidence_exhausted",
            },
        ]
        assert result["revision_count"] == 2

    def test_repeat_raise_of_exhausted_item_is_folded_by_node(self, monkeypatch):
        """If the reviewer model re-raises an evidence_exhausted item, the
        hard guard must fold it into unresolvable_gaps: it never becomes an
        open review_item, and genuinely new/open items are unaffected."""
        from ora.agents import reviewer as reviewer_module
        from ora.agents.reviewer import reviewer_node

        repeat_exhausted_json = (
            '{"verdict": "REVISE", "blocking": ["2026 outlook unavailable in sources"],'
            ' "required": ["Add pricing details"], "suggested": [],'
            ' "contradicting_evidence_found": [], "confidence_recalibrations": {},'
            ' "unresolvable_gaps": []}'
        )
        llm = _AuditRecordingLLM([repeat_exhausted_json])
        monkeypatch.setattr(
            reviewer_module,
            "get_llm",
            lambda model_name, temperature=0.2: llm,
        )

        # _audit_state carries "2026 outlook unavailable in sources" with
        # status evidence_exhausted and "Add pricing details" as open.
        result = reviewer_node(self._audit_state(revision_count=1))

        verdict = result["review_verdict"]
        assert verdict.verdict == "REVISE"
        # The exhausted item is gone from blocking and folded into gaps.
        assert verdict.blocking == []
        assert verdict.unresolvable_gaps == ["2026 outlook unavailable in sources"]
        # The genuinely open prior item stays actionable as an open item.
        assert result["review_items"] == [
            {"category": "required", "text": "Add pricing details", "status": "open"}
        ]
        assert result["revision_count"] == 2

    def test_undocumented_exhausted_rerisen_stays_blocking_before_final(self, monkeypatch):
        """I1: when the writer has NOT documented an exhausted item and the
        audit is not final, a re-raise must stay in blocking so the
        exhausted-but-undocumented documentation-gap REVISE reaches the
        writer (the fold must not silently accept it)."""
        from ora.agents import reviewer as reviewer_module
        from ora.agents.reviewer import reviewer_node

        rerisen_json = (
            '{"verdict": "REVISE", "blocking": ["2026 outlook unavailable in sources"],'
            ' "required": [], "suggested": [], "contradicting_evidence_found": [],'
            ' "confidence_recalibrations": {}, "unresolvable_gaps": []}'
        )
        llm = _AuditRecordingLLM([rerisen_json])
        monkeypatch.setattr(
            reviewer_module,
            "get_llm",
            lambda model_name, temperature=0.2: llm,
        )

        state = {
            "query": "Rust vs Go",
            "draft_report": "# Research\nrevised body",
            "review_items": [
                {
                    "category": "required",
                    "text": "2026 outlook unavailable in sources",
                    "status": "evidence_exhausted",
                }
            ],
            # The writer did NOT document the limitation this round.
            "writer_change_notes": "(no revision notes)",
            "revision_count": 1,  # audit 2 of 3: not final
        }
        result = reviewer_node(state)

        verdict = result["review_verdict"]
        assert verdict.verdict == "REVISE"
        assert verdict.blocking == ["2026 outlook unavailable in sources"]
        assert verdict.unresolvable_gaps == []
        # The documentation-gap item becomes the next pass's open item.
        assert result["review_items"] == [
            {
                "category": "blocking",
                "text": "2026 outlook unavailable in sources",
                "status": "open",
            }
        ]
        assert result["revision_count"] == 2

    def test_undocumented_exhausted_rerisen_folds_on_final_audit(self, monkeypatch):
        """I1: on the FINAL audit a re-raised exhausted item folds into gaps
        even when the writer never documented it: no researcher pass remains,
        and the gap text must be surfaced in the end state."""
        from ora.agents import reviewer as reviewer_module
        from ora.agents.reviewer import reviewer_node

        rerisen_json = (
            '{"verdict": "REVISE", "blocking": ["2026 outlook unavailable in sources"],'
            ' "required": [], "suggested": [], "contradicting_evidence_found": [],'
            ' "confidence_recalibrations": {}, "unresolvable_gaps": []}'
        )
        llm = _AuditRecordingLLM([rerisen_json])
        monkeypatch.setattr(
            reviewer_module,
            "get_llm",
            lambda model_name, temperature=0.2: llm,
        )

        state = {
            "query": "Rust vs Go",
            "draft_report": "# Research\nrevised body",
            "review_items": [
                {
                    "category": "required",
                    "text": "2026 outlook unavailable in sources",
                    "status": "evidence_exhausted",
                }
            ],
            "writer_change_notes": "(no revision notes)",
            "revision_count": 2,  # audit 3 of 3: FINAL
        }
        result = reviewer_node(state)

        verdict = result["review_verdict"]
        assert verdict.verdict == "REVISE"
        assert verdict.blocking == []
        assert verdict.unresolvable_gaps == ["2026 outlook unavailable in sources"]
        assert result["review_items"] == []
        assert result["revision_count"] == 3

    def test_exhausted_item_carried_then_dropped_after_documentation(self, monkeypatch):
        """I2: an exhausted item the writer has NOT documented persists in
        review_items as evidence_exhausted after a PASS-with-gaps audit; once
        the writer documents it and the reviewer accepts it into gaps, the
        record is gone."""
        from ora.agents import reviewer as reviewer_module
        from ora.agents.reviewer import reviewer_node

        gap_item = "2026 outlook unavailable in sources"
        pass_with_other_gap = (
            '{"verdict": "PASS", "blocking": [], "required": [], "suggested": [],'
            ' "contradicting_evidence_found": [], "confidence_recalibrations": {},'
            ' "unresolvable_gaps": ["unrelated accepted gap"]}'
        )
        pass_accepting_item = (
            '{"verdict": "PASS", "blocking": [], "required": [], "suggested": [],'
            ' "contradicting_evidence_found": [], "confidence_recalibrations": {},'
            f' "unresolvable_gaps": ["{gap_item}"]}}'
        )
        llm = _AuditRecordingLLM([pass_with_other_gap, pass_accepting_item])
        monkeypatch.setattr(
            reviewer_module,
            "get_llm",
            lambda model_name, temperature=0.2: llm,
        )

        def _state(notes):
            return {
                "query": "Rust vs Go",
                "draft_report": "# Research\nrevised body",
                "review_items": [
                    {"category": "required", "text": gap_item, "status": "evidence_exhausted"}
                ],
                "writer_change_notes": notes,
                "revision_count": 1,
            }

        # Phase 1: writer never documented the exhausted item; reviewer PASSes
        # with a different accepted gap. The item must persist as
        # evidence_exhausted (history survives, writer still sees it).
        phase1 = reviewer_node(_state("(no revision notes)"))
        assert phase1["review_verdict"].verdict == "PASS"
        assert phase1["review_items"] == [
            {"category": "required", "text": gap_item, "status": "evidence_exhausted"}
        ]

        # Phase 2: the writer documented the item and the reviewer accepts it
        # into unresolvable_gaps. The exhausted record is now gone.
        phase2 = reviewer_node(
            _state(f"## Changes made\n- [required] {gap_item}: documented as a gap.\n")
        )
        assert phase2["review_verdict"].verdict == "PASS"
        assert phase2["review_verdict"].unresolvable_gaps == [gap_item]
        assert phase2["review_items"] == []

    def test_acknowledged_item_not_rerisen_or_accepted_survives_audit(self, monkeypatch):
        """Acknowledged + not re-raised + not accepted: the exhausted item
        must still be present in structured state as an evidence_exhausted
        record rather than silently disappearing from both review_items and
        unresolvable_gaps."""
        from ora.agents import reviewer as reviewer_module
        from ora.agents.reviewer import reviewer_node

        gap_item = "2026 outlook unavailable in sources"
        pass_with_other_gap = (
            '{"verdict": "PASS", "blocking": [], "required": [], "suggested": [],'
            ' "contradicting_evidence_found": [], "confidence_recalibrations": {},'
            ' "unresolvable_gaps": ["unrelated accepted gap"]}'
        )
        llm = _AuditRecordingLLM([pass_with_other_gap])
        monkeypatch.setattr(
            reviewer_module,
            "get_llm",
            lambda model_name, temperature=0.2: llm,
        )

        result = reviewer_node(
            {
                "query": "Rust vs Go",
                "draft_report": "# Research\nrevised body",
                "review_items": [
                    {"category": "required", "text": gap_item, "status": "evidence_exhausted"}
                ],
                # The writer acknowledged the item in its notes...
                "writer_change_notes": (
                    f"## Changes made\n- [required] {gap_item}: documented as a gap.\n"
                ),
                # ...but the reviewer neither re-raised it nor accepted it.
                "revision_count": 1,
            }
        )

        assert result["review_verdict"].verdict == "PASS"
        assert result["review_verdict"].unresolvable_gaps == ["unrelated accepted gap"]
        assert result["review_items"] == [
            {"category": "required", "text": gap_item, "status": "evidence_exhausted"}
        ]

    def test_first_audit_defaults_context_placeholders(self, monkeypatch):
        """A first audit (no review_items/notes/deltas) renders the context
        placeholders with their default values and audit_number 1."""
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
        # Audit position context and the final-audit rule text are present.
        assert "AUDIT_NUMBER: 1" in prompt
        assert "MAX_AUDITS: 3" in prompt
        assert "FINAL audit" in prompt
        assert "material flaws" in prompt
        assert "Inverse rule" in prompt

    def test_final_audit_renders_audit_number_at_max(self, monkeypatch):
        """A re-audit at revision_count == MAX_REVISIONS - 1 is the FINAL
        audit: the prompt must render AUDIT_NUMBER equal to MAX_AUDITS so the
        reviewer knows no researcher pass will follow a REVISE."""
        from ora.agents import reviewer as reviewer_module
        from ora.agents.reviewer import reviewer_node

        llm = _AuditRecordingLLM([PASS_WITH_GAPS_JSON])
        monkeypatch.setattr(
            reviewer_module,
            "get_llm",
            lambda model_name, temperature=0.2: llm,
        )

        reviewer_node(self._audit_state(revision_count=2))

        prompt = llm.prompts[-1]
        assert "AUDIT_NUMBER: 3" in prompt
        assert "MAX_AUDITS: 3" in prompt

    def test_non_final_audit_renders_audit_number_below_max(self, monkeypatch):
        """A second audit (revision_count 1) renders AUDIT_NUMBER 2 of 3."""
        from ora.agents import reviewer as reviewer_module
        from ora.agents.reviewer import reviewer_node

        llm = _AuditRecordingLLM([PASS_WITH_GAPS_JSON])
        monkeypatch.setattr(
            reviewer_module,
            "get_llm",
            lambda model_name, temperature=0.2: llm,
        )

        reviewer_node(self._audit_state(revision_count=1))

        prompt = llm.prompts[-1]
        assert "AUDIT_NUMBER: 2" in prompt
        assert "MAX_AUDITS: 3" in prompt
        assert "AUDIT_NUMBER: 3" not in prompt

    def test_later_audit_prompt_carries_escalated_audit_policy(self, monkeypatch):
        """A later audit's prompt must carry the full Audit Policy: Audit 1 is
        open-ended, later audits restrict REVISE to unaddressed prior items or
        NEW factual errors, and the final audit folds everything else to
        unresolvable_gaps + PASS."""
        from ora.agents import reviewer as reviewer_module
        from ora.agents.reviewer import reviewer_node

        llm = _AuditRecordingLLM([PASS_WITH_GAPS_JSON])
        monkeypatch.setattr(
            reviewer_module,
            "get_llm",
            lambda model_name, temperature=0.2: llm,
        )

        reviewer_node(self._audit_state(revision_count=1))

        prompt = llm.prompts[-1]
        assert "## Audit Policy" in prompt
        assert "**Audit 1**" in prompt
        assert "**Later audits** (AUDIT_NUMBER >= 2)" in prompt
        assert "must NOT trigger REVISE on later audits" in prompt
        assert "**Final audit** (AUDIT_NUMBER == MAX_AUDITS)" in prompt
        assert "return PASS" in prompt
        # Scoped new-issue rule: no blanket permission to raise new issues.
        assert "New issues are scoped by the Audit Policy above" in prompt

    def test_budget_one_prompt_notes_first_and_final_audit(self, monkeypatch):
        """M2: with a budget of 1 the first audit is also the final audit; the
        prompt must resolve the Audit-1 / Final-audit contradiction."""
        from ora.agents import reviewer as reviewer_module
        from ora.agents.reviewer import reviewer_node

        llm = _AuditRecordingLLM([PASS_WITH_GAPS_JSON])
        monkeypatch.setattr(
            reviewer_module,
            "get_llm",
            lambda model_name, temperature=0.2: llm,
        )

        reviewer_node(
            {"query": "Rust vs Go", "draft_report": "# Research\nbody", "max_revisions": 1}
        )

        prompt = llm.prompts[-1]
        assert "AUDIT_NUMBER: 1" in prompt
        assert "MAX_AUDITS: 1" in prompt
        assert "both the first and the final audit" in prompt
