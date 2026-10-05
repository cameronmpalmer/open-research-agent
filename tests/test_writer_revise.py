"""Tests for writer surgical revision mode and change notes.

Covers the revision branch of writer_node: the prompt receives the previous
draft body plus the review items (with statuses) and only the new findings;
the assembled draft keeps fresh programmatic sections; writer_change_notes is
extracted from the LLM's "Changes made" block (and is "" on first pass);
programmatic sections echoed back by the LLM are not duplicated.
"""

from ora.agents import writer as writer_module
from ora.agents.writer import _extract_change_notes, _strip_previous_change_notes, writer_node
from ora.prompts import REVISION_PROMPT, WRITER_PROMPT
from ora.state import Finding, Source

PREV_BODY = """## Executive Summary
Original summary text.

## Key Findings
### Costs
**Finding:** Original claim.
"""

PREV_SOURCE = Source(
    url="https://example.com/old", title="Old source", overall_reliability="Medium"
)

NEW_SOURCE = Source(url="https://example.com/new", title="New source", overall_reliability="High")

OLD_DRAFT = (
    "# Research: test query\n"
    "**Intensity:** Level 3 | **Sources:** 1 | **Date:** 2026-09-03\n\n"
    + PREV_BODY
    + "\n"
    + "## Source Table\n"
    "| # | Title | URL | Type | Reliability |\n"
    "|---|-------|-----|------|-------------|\n"
    "| 1 | Old source | https://example.com/old | unknown | Medium |\n" + "\n" + "## Bibliography\n"
    "1. Old source. [https://example.com/old](https://example.com/old)\n"
)

REVIEW_ITEMS = [
    {"category": "blocking", "text": "Add pricing details", "status": "open"},
    {
        "category": "required",
        "text": "2026 outlook unavailable in sources",
        "status": "evidence_exhausted",
    },
]


class _RecordingLLM:
    def __init__(self, content=None, error=None):
        self.prompts = []
        self._content = content
        self.error = error

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return type("_Response", (), {"content": self._content})()


def _patch_writer(monkeypatch, llm):
    monkeypatch.setattr(writer_module, "get_llm", lambda model_name, temperature=0.3: llm)
    monkeypatch.setattr(writer_module, "get_researcher_model", lambda settings: "fake-model")


def _revision_state(**overrides) -> dict:
    state = {
        "query": "test query",
        "intensity": 3,
        "sources": [PREV_SOURCE, NEW_SOURCE],
        "findings": [
            Finding(claim="old finding", supporting_sources=[PREV_SOURCE.url]),
            Finding(claim="new finding", supporting_sources=[NEW_SOURCE.url]),
        ],
        "draft_report": OLD_DRAFT,
        "review_items": [dict(i) for i in REVIEW_ITEMS],
        "last_round_new_sources": 1,
        "last_round_new_findings": 1,
    }
    state.update(overrides)
    return state


def test_revision_mode_prompt_includes_previous_draft_items_and_new_evidence(monkeypatch):
    """The revision prompt must carry the previous draft (body plus its
    source table/bibliography context, so items about specific citations can
    be addressed), every item with its status, and only the new findings
    from the pass just audited."""
    llm = _RecordingLLM(
        content=(
            "## Executive Summary\nRevised summary.\n\n"
            "## Changes made\n"
            "- [blocking] Add pricing details: resolved with new source.\n"
            "- [required] 2026 outlook: documented as a gap.\n"
        )
    )
    _patch_writer(monkeypatch, llm)

    result = writer_node(_revision_state())

    prompt = llm.prompts[-1]
    assert "REVIEW ITEMS TO ADDRESS" in prompt
    # The previous LLM body is present...
    assert "Original summary text." in prompt
    assert "Original claim." in prompt
    # ...together with its programmatic context (source table + bibliography).
    assert "## Source Table" in prompt
    assert "Old source | https://example.com/old" in prompt
    assert "## Bibliography" in prompt
    # Every item appears with category + status.
    assert "[blocking] (open) Add pricing details" in prompt
    assert "[required] (evidence_exhausted) 2026 outlook unavailable in sources" in prompt
    # Only the new finding (tail) is passed as new evidence.
    assert "new finding" in prompt
    assert "old finding" not in prompt
    assert "NEW EVIDENCE SINCE THE PREVIOUS DRAFT" in prompt

    # Change notes extracted and draft assembled with fresh sections.
    assert result["writer_change_notes"] != ""
    assert "Add pricing details: resolved" in result["writer_change_notes"]
    assert result["draft_report"].startswith("# Research: test query")
    assert "Revised summary." in result["draft_report"]


def test_revision_mode_draft_keeps_fresh_programmatic_sections(monkeypatch):
    """Even when the LLM echoes the previous draft's programmatic sections,
    the assembled draft must contain exactly one source table/bibliography
    rebuilt from the CURRENT sources (old source dropped, new source listed)."""
    echoed = (
        PREV_BODY + "\n" + "## Source Table\n"
        "| # | Title | URL | Type | Reliability |\n"
        "|---|-------|-----|------|-------------|\n"
        "| 1 | Old source | https://example.com/old | unknown | Medium |\n"
        + "\n"
        + "## Bibliography\n"
        "1. Old source. [https://example.com/old](https://example.com/old)\n"
    )
    llm = _RecordingLLM(content=echoed)
    _patch_writer(monkeypatch, llm)

    result = writer_node(_revision_state())

    draft = result["draft_report"]
    # Fresh header reflects the current source count (2).
    assert "**Sources:** 2" in draft
    # New source appears; old source only as part of the LLM body (the
    # body above mentions it) - the programmatic sections list the new one.
    assert draft.count("## Source Table") == 1
    assert draft.count("## Bibliography") == 1
    source_table_idx = draft.index("## Source Table")
    bibliography_idx = draft.index("## Bibliography")
    assert "New source | https://example.com/new" in draft[source_table_idx:bibliography_idx]
    assert "New source. [https://example.com/new]" in draft[bibliography_idx:]


def test_purely_programmatic_response_does_not_duplicate_sections(monkeypatch):
    """A revision response whose pre-marker body strips to "" must not fall
    back to the raw response: assembly keeps exactly one freshly built Source
    Table and Bibliography instead of appending to the echoed ones."""
    echoed = (
        "## Source Table\n"
        "| # | Title | URL | Type | Reliability |\n"
        "|---|-------|-----|------|-------------|\n"
        "| 1 | Old source | https://example.com/old | unknown | Medium |\n"
        "\n"
        "## Bibliography\n"
        "1. Old source. [https://example.com/old](https://example.com/old)\n"
    )
    llm = _RecordingLLM(content=echoed)
    _patch_writer(monkeypatch, llm)

    result = writer_node(_revision_state())

    draft = result["draft_report"]
    assert draft.count("## Source Table") == 1
    assert draft.count("## Bibliography") == 1
    # The surviving sections are the fresh ones built from the current sources.
    assert "New source | https://example.com/new" in draft
    assert "New source. [https://example.com/new]" in draft


def test_revision_mode_change_notes_survive_full_report_echo(monkeypatch):
    """When the LLM echoes the whole previous report (body + change notes +
    programmatic tail), writer_change_notes is still extracted and the
    assembled draft contains no duplicated programmatic sections."""
    echoed = (
        PREV_BODY + "\n" + "## Changes made\n"
        "- [blocking] Add pricing details: resolved with new source.\n" + "\n" + "## Source Table\n"
        "| # | Title | URL | Type | Reliability |\n"
        "|---|-------|-----|------|-------------|\n"
        "| 1 | Old source | https://example.com/old | unknown | Medium |\n"
        + "\n"
        + "## Bibliography\n"
        "1. Old source. [https://example.com/old](https://example.com/old)\n"
    )
    llm = _RecordingLLM(content=echoed)
    _patch_writer(monkeypatch, llm)

    result = writer_node(_revision_state())

    assert "Add pricing details: resolved" in result["writer_change_notes"]
    draft = result["draft_report"]
    assert draft.count("## Source Table") == 1
    assert draft.count("## Bibliography") == 1


def test_literal_order_full_report_echo_with_trailing_change_notes(monkeypatch):
    """Literal-order interpretation: the LLM echoes the full previous report
    (header + body + source table + bibliography) and appends the 'Changes
    made' section at the very END, after the programmatic tail.

    writer_change_notes must still be extracted from the raw response, and
    the assembled draft must not duplicate programmatic sections - only the
    fresh source table/bibliography (current sources) remain."""
    echoed = (
        "# Research: test query\n"
        "**Intensity:** Level 3 | **Sources:** 1 | **Date:** 2026-09-03\n\n"
        + PREV_BODY
        + "\n"
        + "## Source Table\n"
        "| # | Title | URL | Type | Reliability |\n"
        "|---|-------|-----|------|-------------|\n"
        "| 1 | Old source | https://example.com/old | unknown | Medium |\n"
        "\n"
        "## Bibliography\n"
        "1. Old source. [https://example.com/old](https://example.com/old)\n"
        "\n"
        "## Changes made\n"
        "- [blocking] Add pricing details: resolved with new source.\n"
        "- [required] 2026 outlook unavailable in sources: documented as a gap.\n"
    )
    llm = _RecordingLLM(content=echoed)
    _patch_writer(monkeypatch, llm)

    result = writer_node(_revision_state())

    # Dispositions are extracted from the raw response (marker to end), even
    # though the marker sits after the echoed programmatic tail.
    assert "Add pricing details: resolved with new source." in result["writer_change_notes"]
    assert (
        "2026 outlook unavailable in sources: documented as a gap." in result["writer_change_notes"]
    )

    draft = result["draft_report"]
    # No duplicated programmatic sections: exactly one of each, rebuilt fresh
    # from the CURRENT sources (both old and new appear with the fresh
    # header's source count of 2).
    assert draft.count("# Research: test query") == 1
    assert draft.count("## Source Table") == 1
    assert draft.count("## Bibliography") == 1
    assert "**Sources:** 2" in draft
    source_table_idx = draft.index("## Source Table")
    bibliography_idx = draft.index("## Bibliography")
    assert "New source | https://example.com/new" in draft[source_table_idx:bibliography_idx]
    assert "New source. [https://example.com/new]" in draft[bibliography_idx:]


def test_revision_mode_change_notes_empty_when_marker_absent(monkeypatch):
    """If the LLM omits the 'Changes made' section, writer_change_notes is ''
    and the draft still assembles normally."""
    llm = _RecordingLLM(content="## Executive Summary\nRevised without notes.\n")
    _patch_writer(monkeypatch, llm)

    result = writer_node(_revision_state())

    assert result["writer_change_notes"] == ""
    assert result["draft_report"].startswith("# Research: test query")
    assert "Revised without notes." in result["draft_report"]


def test_first_pass_unchanged_and_change_notes_empty(monkeypatch):
    """No review_items -> WRITER_PROMPT path (unchanged) and
    writer_change_notes == ''."""
    llm = _RecordingLLM(content="# Research: test query\n\nFirst-pass body.")
    _patch_writer(monkeypatch, llm)

    result = writer_node(
        {
            "query": "test query",
            "intensity": 3,
            "sources": [NEW_SOURCE],
            "findings": [Finding(claim="a finding", supporting_sources=[NEW_SOURCE.url])],
        }
    )

    prompt = llm.prompts[-1]
    assert "REVIEW ITEMS TO ADDRESS" not in prompt
    assert prompt.startswith(WRITER_PROMPT.split("Research findings")[0][:40])
    assert "a finding" in prompt
    assert result["writer_change_notes"] == ""
    assert result["draft_report"].startswith("# Research: test query")


def test_review_items_without_previous_draft_falls_back_to_first_pass(monkeypatch):
    """Review items but no draft_report (defensive) must not enter revision
    mode: WRITER_PROMPT is used and writer_change_notes stays ''."""
    llm = _RecordingLLM(content="First-pass body.")
    _patch_writer(monkeypatch, llm)

    result = writer_node(
        {
            "query": "test query",
            "intensity": 3,
            "findings": [Finding(claim="a finding")],
            "review_items": [dict(i) for i in REVIEW_ITEMS],
        }
    )

    prompt = llm.prompts[-1]
    assert "REVIEW ITEMS TO ADDRESS" not in prompt
    assert "ORIGINAL QUERY" not in prompt
    assert result["writer_change_notes"] == ""


def test_emit_message_marks_revision(monkeypatch):
    events = []
    llm = _RecordingLLM(content="## Executive Summary\nRev.\n\n## Changes made\n- done")
    _patch_writer(monkeypatch, llm)

    writer_node(
        _revision_state(),
        {"configurable": {"progress_callback": events.append}},
    )

    messages = [event["message"] for event in events]
    assert any("draft generated" in m and "(revision)" in m for m in messages)


def test_second_revision_strips_prior_change_notes_from_context(monkeypatch):
    """On a second revision, the previous draft's 'Changes made' block must
    not be fed back as context (each revision writes a fresh block for the
    current items), while the source table/bibliography context stays."""

    prior = (
        "# Research: test query\n"
        "**Intensity:** Level 3 | **Sources:** 1 | **Date:** 2026-09-03\n\n"
        "## Executive Summary\nEarlier text.\n\n"
        "## Changes made\n"
        "- [blocking] Add pricing details: resolved.\n"
        "\n"
        "## Source Table\n"
        "| # | Title | URL | Type | Reliability |\n"
        "|---|-------|-----|------|-------------|\n"
        "| 1 | Old source | https://example.com/old | unknown | Medium |\n"
        "\n"
        "## Bibliography\n"
        "1. Old source. [https://example.com/old](https://example.com/old)\n"
    )
    context = _strip_previous_change_notes(prior)

    # Change-notes block dropped, but the header/body and programmatic
    # context stay (the marker heading AND its disposition lines are gone).
    assert "## Changes made" not in context
    assert "Add pricing details: resolved." not in context
    assert "# Research: test query" in context
    assert "Earlier text." in context
    assert "## Source Table" in context
    assert "https://example.com/old" in context

    # End-to-end: revision prompt receives the stripped context (the prompt
    # template itself mentions "Changes made" as the required closing
    # section, so assert on the previous draft's disposition text instead).
    llm = _RecordingLLM(content="## Executive Summary\nNewer text.\n\n## Changes made\n- fixed")
    _patch_writer(monkeypatch, llm)

    result = writer_node(_revision_state(draft_report=prior))

    prompt = llm.prompts[-1]
    assert "Add pricing details: resolved." not in prompt
    assert "Earlier text." in prompt
    assert "## Source Table" in prompt
    assert result["writer_change_notes"].startswith("## Changes made") or result[
        "writer_change_notes"
    ].startswith("Changes made")


def test_assembled_revision_keeps_change_notes_in_body(monkeypatch):
    """The LLM's 'Changes made' block is retained in the revised draft body
    (alongside the fresh programmatic sections) so the re-audit reviewer can
    see the writer's per-item dispositions."""
    llm = _RecordingLLM(
        content=(
            "## Executive Summary\nRevised summary.\n\n"
            "## Changes made\n"
            "- [blocking] Add pricing details: resolved with new source.\n"
        )
    )
    _patch_writer(monkeypatch, llm)

    result = writer_node(_revision_state())

    draft = result["draft_report"]
    assert "Changes made" in draft
    assert "Add pricing details: resolved with new source." in draft
    assert draft.index("Changes made") < draft.index("## Source Table")


class TestExtractChangeNotes:
    def test_returns_section_from_marker_to_end(self):
        body = "## Key Findings\nText.\n\n## Changes made\n- item one: fixed\n- item two: gap\n"
        notes = _extract_change_notes(body)
        assert notes.startswith("## Changes made")
        assert "item one: fixed" in notes
        assert "item two: gap" in notes

    def test_stops_at_echoed_programmatic_section(self):
        body = (
            "## Key Findings\nText.\n\n"
            "## Changes made\n- item one: fixed\n\n"
            "## Source Table\n| 1 | Old | http://x | unknown | Medium |\n"
        )
        notes = _extract_change_notes(body)
        assert "item one: fixed" in notes
        assert "## Source Table" not in notes
        assert "http://x" not in notes

    def test_returns_empty_when_marker_absent(self):
        assert _extract_change_notes("## Executive Summary\nNo notes.") == ""
        assert _extract_change_notes("") == ""

    def test_heading_without_hash_marker_also_matches(self):
        body = "Body prose.\n\nChanges made\n- item one: resolved"
        notes = _extract_change_notes(body)
        assert notes.startswith("Changes made")
        assert "item one: resolved" in notes

    def test_prose_mentioning_changes_made_is_not_a_marker(self):
        body = "The reviewer noted changes made to the draft were insufficient.\nStill body."
        assert _extract_change_notes(body) == ""


def test_revision_prompt_is_exported():
    """The REVISION_PROMPT constant is exported and formatable with the
    writer's expected placeholders."""
    assert "REVISION_PROMPT" in __import__("ora.prompts", fromlist=["REVISION_PROMPT"]).__all__
    rendered = REVISION_PROMPT.format(
        query="q",
        review_items="- [blocking] (open) item",
        new_evidence="(none)",
        previous_draft="prev body",
    )
    assert "ORIGINAL QUERY: q" in rendered
    assert "- [blocking] (open) item" in rendered
    assert "Changes made" in rendered
