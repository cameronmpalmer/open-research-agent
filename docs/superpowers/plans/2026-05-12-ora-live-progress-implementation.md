# ORA Live Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add live, line-by-line progress output by default to `ora research`, with a quiet mode for spinner-only output.

**Architecture:** Add a tiny progress-event helper that agents can call through LangGraph `RunnableConfig`, keeping agent code decoupled from terminal rendering. The CLI will pass a progress callback by default and will keep the existing spinner-only behavior when `--quiet` is set.

**Tech Stack:** Python, Click, Rich, LangGraph `RunnableConfig`, pytest, monkeypatch fixtures.

---

## File Structure

- Create: `ora/progress.py`
  - Owns the progress event convention and `emit_progress(config, message, kind="info")` helper.
  - Has no Click or Rich dependency.
- Create: `tests/test_progress.py`
  - Tests callback/no-callback behavior and callback exception safety.
- Modify: `ora/agents/researcher.py`
  - Emits progress events around query generation, search, URL extraction, scrape attempts, scrape failures/successes, source evaluation, and final counts.
- Modify: `ora/agents/writer.py`
  - Emits progress events before and after report synthesis.
- Create: `tests/test_researcher_progress.py`
  - Tests researcher progress events with mocked search/scrape/evaluation.
- Create: `tests/test_writer_progress.py`
  - Tests writer progress events with mocked LLM.
- Modify: `ora/cli.py`
  - Adds live progress rendering by default and passes graph config to `research_graph.invoke()` unless `--quiet` is active.
- Modify: `tests/test_cli.py`
  - Adds lightweight tests for progress icon formatting without invoking real LLMs or Firecrawl.

---

### Task 1: Add the progress helper

**Files:**
- Create: `ora/progress.py`
- Create: `tests/test_progress.py`

- [ ] **Step 1: Write failing tests for progress event emission**

Create `tests/test_progress.py` with:

```python
"""Tests for progress event helper."""
from ora.progress import emit_progress


def test_emit_progress_no_config_is_noop():
    emit_progress(None, "hello", kind="info")


def test_emit_progress_without_callback_is_noop():
    emit_progress({"configurable": {}}, "hello", kind="info")


def test_emit_progress_calls_callback_with_event_dict():
    events = []
    config = {"configurable": {"progress_callback": events.append}}

    emit_progress(config, "Researcher: searching", kind="search")

    assert events == [{"message": "Researcher: searching", "kind": "search"}]


def test_emit_progress_callback_exception_does_not_break_research():
    def callback(_event):
        raise RuntimeError("display failed")

    config = {"configurable": {"progress_callback": callback}}

    emit_progress(config, "Researcher: searching", kind="search")
```

- [ ] **Step 2: Run the failing progress tests**

Run:

```bash
python3 -m pytest tests/test_progress.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ora.progress'`.

- [ ] **Step 3: Implement `ora/progress.py`**

Create `ora/progress.py` with:

```python
"""Progress event helpers for ORA agent nodes.

Agents call these helpers through LangGraph RunnableConfig. The CLI decides
whether and how to render events.
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict


ProgressKind = Literal["info", "search", "scrape", "success", "error", "write"]


class ProgressEvent(TypedDict):
    """A live progress event emitted by an ORA agent."""

    message: str
    kind: ProgressKind


def emit_progress(config: dict[str, Any] | None, message: str, kind: ProgressKind = "info") -> None:
    """Emit a progress event if the caller supplied a callback.

    Progress rendering is best-effort. A broken display callback must not break
    research execution.
    """
    configurable = (config or {}).get("configurable", {})
    callback = configurable.get("progress_callback")
    if not callback:
        return

    try:
        callback({"message": message, "kind": kind})
    except Exception:
        return
```

- [ ] **Step 4: Run progress tests to verify they pass**

Run:

```bash
python3 -m pytest tests/test_progress.py -q
```

Expected: PASS, 4 tests.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add ora/progress.py tests/test_progress.py
git commit -m "feat: add progress event helper"
```

---

### Task 2: Emit researcher progress events

**Files:**
- Modify: `ora/agents/researcher.py`
- Create: `tests/test_researcher_progress.py`

- [ ] **Step 1: Write failing researcher progress test**

Create `tests/test_researcher_progress.py` with:

```python
"""Tests for researcher progress events."""
from ora.agents import researcher as researcher_module
from ora.agents.researcher import researcher_node
from ora.state import Source


class FakeTool:
    def __init__(self, value):
        self.value = value

    def invoke(self, _args):
        return self.value


def test_researcher_emits_progress_events(monkeypatch):
    events = []

    monkeypatch.setattr(
        "ora.tools.search.web_search",
        FakeTool("1. [Example](https://example.com/article)\n   snippet"),
    )
    monkeypatch.setattr(
        "ora.tools.scrape.scrape_page",
        FakeTool("This is scraped content about Rust vs Go."),
    )
    monkeypatch.setattr(
        "ora.tools.evaluate.evaluate_source",
        lambda url, content, source_type: Source(url=url, title="Example", source_type=source_type),
    )

    result = researcher_node(
        {"query": "Rust vs Go", "intensity": 1},
        {"configurable": {"progress_callback": events.append}},
    )

    messages = [event["message"] for event in events]
    kinds = [event["kind"] for event in events]

    assert result["findings"]
    assert "search" in kinds
    assert "scrape" in kinds
    assert "success" in kinds
    assert any("generated 1 search query" in message for message in messages)
    assert any('searching "Rust vs Go"' in message for message in messages)
    assert any("found 1 URL" in message for message in messages)
    assert any("scraped" in message for message in messages)
    assert any("finished with 1 source" in message for message in messages)
```

- [ ] **Step 2: Run the failing researcher progress test**

Run:

```bash
python3 -m pytest tests/test_researcher_progress.py -q
```

Expected: FAIL because `researcher_node()` does not emit progress events yet.

- [ ] **Step 3: Import and call `emit_progress()` in researcher**

Modify `ora/agents/researcher.py`.

Add this import near the existing imports:

```python
from ora.progress import emit_progress
```

In `researcher_node()`, after `queries = generate_search_queries(...)[:query_count]`, add:

```python
    query_label = "query" if len(queries) == 1 else "queries"
    emit_progress(
        config,
        f"Researcher: generated {len(queries)} search {query_label}",
        kind="search",
    )
```

Inside the `for q in queries:` loop, before `web_search.invoke(...)`, add:

```python
        emit_progress(config, f'Researcher: searching "{q}"', kind="search")
```

After URL extraction, add:

```python
        url_label = "URL" if len(urls) == 1 else "URLs"
        emit_progress(config, f"Researcher: found {len(urls)} {url_label}", kind="success")
```

Before each scrape call, add:

```python
            display_url = url.replace("https://", "").replace("http://", "")[:80]
            emit_progress(config, f"Researcher: scraping {display_url}", kind="scrape")
```

After computing `is_error`, replace the existing success path with progress events:

```python
            if is_error:
                emit_progress(config, f"Researcher: scrape failed for {display_url}", kind="error")
                continue

            emit_progress(config, f"Researcher: scraped {len(c)} chars from {display_url}", kind="success")
            source = evaluate_source(url=url, content=c, source_type="unknown")
            emit_progress(config, "Researcher: evaluated source reliability", kind="info")
            sources.append(source)
            findings.append(Finding(claim=c[:500], supporting_sources=[url]))
            scraped_count += 1
            if scraped_count >= 2:
                break
```

Before the return statement, add:

```python
    source_label = "source" if len(sources) == 1 else "sources"
    finding_label = "finding" if len(findings) == 1 else "findings"
    emit_progress(
        config,
        f"Researcher: finished with {len(sources)} {source_label} and {len(findings)} {finding_label}",
        kind="success",
    )
```

- [ ] **Step 4: Run researcher progress test**

Run:

```bash
python3 -m pytest tests/test_researcher_progress.py -q
```

Expected: PASS.

- [ ] **Step 5: Run existing researcher tests**

Run:

```bash
python3 -m pytest tests/test_researcher.py tests/test_researcher_progress.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add ora/agents/researcher.py tests/test_researcher_progress.py
git commit -m "feat: emit researcher progress events"
```

---

### Task 3: Emit writer progress events

**Files:**
- Modify: `ora/agents/writer.py`
- Create: `tests/test_writer_progress.py`

- [ ] **Step 1: Write failing writer progress test**

Create `tests/test_writer_progress.py` with:

```python
"""Tests for writer progress events."""
from ora.agents import writer as writer_module
from ora.agents.writer import writer_node
from ora.state import Finding


class FakeResponse:
    content = "# Research: Rust vs Go\n\nDraft report body."


class FakeLLM:
    def invoke(self, _prompt):
        return FakeResponse()


def test_writer_emits_progress_events(monkeypatch):
    events = []

    monkeypatch.setattr(writer_module, "get_llm", lambda model_name, temperature=0.3: FakeLLM())
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
```

- [ ] **Step 2: Run the failing writer progress test**

Run:

```bash
python3 -m pytest tests/test_writer_progress.py -q
```

Expected: FAIL because `writer_node()` does not emit progress events yet.

- [ ] **Step 3: Add progress events to writer**

Modify `ora/agents/writer.py`.

Add this import near existing imports:

```python
from ora.progress import emit_progress
```

In `writer_node()`, after `findings_raw = state.get("findings", [])`, add:

```python
    finding_label = "finding" if len(findings_raw) == 1 else "findings"
    emit_progress(
        config,
        f"Writer: synthesizing report from {len(findings_raw)} {finding_label}",
        kind="write",
    )
```

After `draft_report = ...`, add:

```python
    emit_progress(config, f"Writer: draft generated, {len(draft_report)} chars", kind="success")
```

- [ ] **Step 4: Run writer progress test**

Run:

```bash
python3 -m pytest tests/test_writer_progress.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add ora/agents/writer.py tests/test_writer_progress.py
git commit -m "feat: emit writer progress events"
```

---

### Task 4: Render live progress in the CLI

**Files:**
- Modify: `ora/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI tests for progress formatting**

Append these tests to `tests/test_cli.py`:

```python
def test_format_progress_event_search():
    from ora.cli import _format_progress_event

    assert _format_progress_event({"kind": "search", "message": "Researcher: searching"}) == "🔎 Researcher: searching"


def test_format_progress_event_unknown_kind_uses_bullet():
    from ora.cli import _format_progress_event

    assert _format_progress_event({"kind": "unexpected", "message": "Something happened"}) == "• Something happened"
```

- [ ] **Step 2: Run failing CLI formatting tests**

Run:

```bash
python3 -m pytest tests/test_cli.py::test_format_progress_event_search tests/test_cli.py::test_format_progress_event_unknown_kind_uses_bullet -q
```

Expected: FAIL because `_format_progress_event` does not exist.

- [ ] **Step 3: Add CLI progress formatting helpers**

Modify `ora/cli.py` near `_print_markdown()` and add:

```python
def _format_progress_event(event: dict) -> str:
    """Format a progress event for line-by-line verbose output."""
    icons = {
        "search": "🔎",
        "scrape": "🌐",
        "success": "✓",
        "error": "✗",
        "write": "✍️",
        "info": "•",
    }
    icon = icons.get(event.get("kind"), "•")
    return f"{icon} {event.get('message', '')}"


def _print_progress_event(event: dict) -> None:
    """Print a progress event immediately."""
    click.echo(_format_progress_event(event))
```

- [ ] **Step 4: Wire quiet mode into research graph invocation**

In `ora/cli.py`, replace:

```python
    final_state = _spin(lambda: research_graph.invoke(plan_result), message="Researching...")
```

with:

```python
    if quiet:
        final_state = _spin(lambda: research_graph.invoke(plan_result), message="Researching...")
    else:
        graph_config = {"configurable": {"progress_callback": _print_progress_event}}
        final_state = research_graph.invoke(plan_result, graph_config)
```

This preserves the spinner in quiet mode and avoids spinner/log output conflicts in live-progress mode.

- [ ] **Step 5: Run CLI tests**

Run:

```bash
python3 -m pytest tests/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

Run:

```bash
git add ora/cli.py tests/test_cli.py
git commit -m "feat: render verbose research progress"
```

---

### Task 5: Full verification and manual smoke test

**Files:**
- No planned file changes unless tests reveal a defect.

- [ ] **Step 1: Run the full test suite**

Run:

```bash
python3 -m pytest tests/ -q
```

Expected: all tests pass.

- [ ] **Step 2: Smoke test quiet CLI still works**

Run:

```bash
printf 'y\n' | ora research "Rust vs Go" --intensity 1 --quiet --output /tmp/ora-quiet.md
```

Expected:

- Output shows plan, approval, `Researching...`, and final save message.
- `/tmp/ora-quiet.md` exists and contains a markdown report.

- [ ] **Step 3: Smoke test default CLI shows live progress**

Run:

```bash
printf 'y\n' | ora research "Rust vs Go" --intensity 1 --output /tmp/ora-live.md
```

Expected output includes lines like:

```text
🔎 Researcher: generated 1 search query
🔎 Researcher: searching "Rust vs Go"
✓ Researcher: found
🌐 Researcher: scraping
✍️ Writer: synthesizing report
✓ Writer: draft generated
```

Expected file result:

- `/tmp/ora-live.md` exists and contains a markdown report.

- [ ] **Step 4: Check git status**

Run:

```bash
git status --short
```

Expected: clean working tree, unless Step 2 or Step 3 created untracked files outside the repo only.

- [ ] **Step 5: Final commit if smoke-test fixes were needed**

Only if Step 2 or Step 3 required additional code changes, run:

```bash
git add ora tests
git commit -m "fix: polish live progress output"
```

Expected: commit succeeds and full tests still pass.
