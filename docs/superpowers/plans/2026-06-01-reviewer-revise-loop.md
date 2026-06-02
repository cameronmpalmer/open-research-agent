# Fix Reviewer REVISE Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Force the researcher to run at least one round of gap queries on reviewer REVISE, even when `min_sources` is already met.

**Architecture:** Two-line change to the while loop condition in `researcher_node`, plus one-line change to the dedup regeneration check. No new state fields, no new LLM calls, no other files touched.

**Tech Stack:** Python 3.12, LangGraph, Pytest

---

### Task 1: Fix researcher REVISE loop and add tests

**Files:**
- Modify: `ora/agents/researcher.py:462` (while loop condition)
- Modify: `ora/agents/researcher.py:490` (dedup regeneration condition)
- Test: `tests/test_researcher.py` (add new test class)

- [ ] **Step 1: Read the current researcher_node code**

Read lines 430-470 of `ora/agents/researcher.py` to understand the context. `reviewer_feedback` is computed at line 458. The while loop starts at line 462. The dedup regeneration check is at line 490.

- [ ] **Step 2: Write failing tests**

Add to `tests/test_researcher.py`, at the end of the file, a new test class:

```python
class TestResearcherReviseLoop:
    def test_researcher_runs_gap_queries_on_revise(self, monkeypatch):
        """When min_sources is met but reviewer has issued REVISE,
        the researcher should still run at least one round."""
        from ora.agents.researcher import researcher_node, _format_reviewer_feedback
        from ora.state import ReviewVerdict
        from ora.state import ResearchState, Source, SourceExtraction
        import types

        # Build state with min_sources already met and a REVISE verdict.
        existing_sources = [Source(url=f"https://example.com/{i}", title=f"Source {i}") for i in range(15)]
        verdict = ReviewVerdict(
            verdict="REVISE",
            blocking=["Need source on topic X"],
            required=["Add coverage of Y"],
        )

        state = ResearchState(
            query="test query",
            intensity=3,
            sources=existing_sources,
            findings=[],
            messages=[],
            review_verdict=verdict,
            revision_count=1,
            executed_queries=["already searched this"],
        )
        # Note: intensity 3 has min_sources=15. With 15 existing sources,
        # min_sources is met. The while loop must still run.

        # Patch web_search to return a result that gives us a new URL.
        def fake_web_search_invoke(input_dict):
            return "[New Source](https://example.com/new)\n  snippet about topic X"

        mock_web_search = types.SimpleNamespace(invoke=fake_web_search_invoke)
        monkeypatch.setattr("ora.tools.search.web_search", mock_web_search)

        def fake_scrape_page_invoke(input_dict):
            return "Scraped content about " + input_dict["url"]

        mock_scrape = types.SimpleNamespace(invoke=fake_scrape_page_invoke)
        monkeypatch.setattr("ora.tools.scrape.scrape_page", mock_scrape)

        def fake_extract_and_evaluate(*args, **kwargs):
            return (
                Source(url=kwargs.get("url", "https://example.com"), title=""),
                SourceExtraction(summary="test"),
            )

        monkeypatch.setattr("ora.tools.extract.extract_and_evaluate", fake_extract_and_evaluate)
        monkeypatch.setattr("ora.tools.evaluate.evaluate_source", lambda u, t, c, cc: Source(url=u, title=t))

        result = researcher_node(state)

        # Should have found the new source.
        new_sources = result.get("sources", [])
        assert len(new_sources) > 0, f"Expected at least one new source, got {len(new_sources)}"

    def test_revise_without_verdict_runs_normally(self, monkeypatch):
        """Without a REVISE verdict, normal behavior: no bypass of min_sources."""
        from ora.agents.researcher import researcher_node
        from ora.state import ResearchState, Source
        import types

        existing_sources = [Source(url=f"https://example.com/{i}", title=f"Source {i}") for i in range(15)]

        state = ResearchState(
            query="test query",
            intensity=3,
            sources=existing_sources,
            findings=[],
            messages=[],
            executed_queries=[],
        )
        # No review_verdict — normal behavior. min_sources=15 is met.

        result = researcher_node(state)

        # Should return zero new sources (no REVISE → loop skipped).
        new_sources = result.get("sources", [])
        assert len(new_sources) == 0, "Expected zero new sources when no REVISE and min_sources met"

    def test_gap_regeneration_fires_round_one_on_revise(self, monkeypatch):
        """When round 1 queries are all deduped and reviewer feedback exists,
        the gap query regeneration should fire even on round 1."""
        from ora.agents.researcher import researcher_node
        from ora.state import ResearchState, Source, SourceExtraction, ReviewVerdict

        verdict = ReviewVerdict(
            verdict="REVISE",
            required=["Find evidence about Z"],
        )

        # State with some existing sources and an executed query that matches
        # the supervisor's planned query (which will be used in round 1).
        state = ResearchState(
            query="test query",
            intensity=3,
            sources=[Source(url="https://example.com/1", title="Source 1")],
            findings=[],
            messages=[],
            review_verdict=verdict,
            revision_count=1,
            executed_queries=["already searched plan query"],
        )

        # The researcher normally uses search_queries from state for round 1.
        # If those are in executed_queries, they get deduped.
        # We test that the regeneration fires by verifying web_search gets called
        # with a new query (not the deduped "already searched plan query").

        search_calls = []

        def fake_web_search_invoke(input_dict):
            search_calls.append(input_dict["query"])
            return "[New Source](https://example.com/new)\n  snippet"

        mock_web_search = types.SimpleNamespace(invoke=fake_web_search_invoke)
        monkeypatch.setattr("ora.tools.search.web_search", mock_web_search)

        def fake_scrape_page_invoke(input_dict):
            return "Content"

        mock_scrape = types.SimpleNamespace(invoke=fake_scrape_page_invoke)
        monkeypatch.setattr("ora.tools.scrape.scrape_page", mock_scrape)

        def fake_extract_and_evaluate(*args, **kwargs):
            return (
                Source(url=kwargs.get("url", "https://example.com"), title=""),
                SourceExtraction(summary="test"),
            )

        monkeypatch.setattr("ora.tools.extract.extract_and_evaluate", fake_extract_and_evaluate)
        monkeypatch.setattr("ora.tools.evaluate.evaluate_source", lambda u, t, c, cc: Source(url=u, title=t))

        result = researcher_node(state)

        # Verify web_search was called. If regeneration didn't fire, no search
        # calls would happen because round 1 queries are all deduped.
        assert len(search_calls) > 0, "web_search should have been called via gap query regeneration"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_researcher.py::TestResearcherReviseLoop -v`
Expected: `test_researcher_runs_gap_queries_on_revise` FAILS (zero new sources — loop skipped). `test_revise_without_verdict_runs_normally` should PASS (current behavior is correct). `test_gap_regeneration_fires_round_one_on_revise` FAILS.

- [ ] **Step 4: Implement the fix**

In `ora/agents/researcher.py`, line 458 already has `reviewer_feedback = _format_reviewer_feedback(state)`. Add immediately after (before the while loop):

```python
    on_revise = bool(reviewer_feedback)
```

Change line 462 from:

```python
    while len(sources) < min_sources and round_num < max_rounds:
        round_num += 1
```

to:

```python
    while (len(sources) < min_sources or on_revise) and round_num < max_rounds:
        round_num += 1
        if on_revise:
            on_revise = False  # one-shot: run one round, then check min_sources normally
```

Change line 490 from:

```python
        if not fresh_queries and round_num > 1:
```

to:

```python
        if not fresh_queries and (round_num > 1 or bool(reviewer_feedback)):
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_researcher.py::TestResearcherReviseLoop -v`
Expected: 3 passed

- [ ] **Step 6: Run full test suite for regressions**

Run: `python3 -m pytest tests/ -v`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add ora/agents/researcher.py tests/test_researcher.py
git commit -m "fix(researcher): force gap query round on reviewer REVISE even when min_sources met"
```
