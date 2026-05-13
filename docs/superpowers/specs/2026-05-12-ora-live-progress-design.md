# ORA Live Progress Design

## Overview

The current `ora research` command shows a single `Researching...` spinner while the researcher searches, scrapes, evaluates sources, and the writer drafts the report. This hides useful activity and makes slow Firecrawl calls or blocked scrapes hard to diagnose.

This design adds live, line-by-line progress events by default. The goal is better visibility without building a full terminal dashboard, with a quiet mode for users who want spinner-only output.

## Goals

- Show what ORA is doing during research by default.
- Keep the quiet output clean and spinner-based.
- Make progress reporting reusable by researcher, writer, and future reviewer nodes.
- Avoid coupling agent code directly to Click, Rich, or terminal output.
- Preserve the current two-phase CLI flow: plan approval, then research.

## Non-Goals

- No Rich live table or dashboard in this iteration.
- No persistent trace file.
- No JSON event output mode.
- No reviewer progress yet, since the reviewer is currently disabled in the default research graph.

## Proposed User Experience

Default mode remains compact:

```text
ORA Research: Rust vs Go
  Intensity: 1 | Researcher: deepseek-v4-flash | Reviewer: deepseek-v4-pro

Generating research plan...
...
Approve this plan and begin research? [y/N]: y

Researching...
  Sources: 2 | Findings: 2 | Draft: 2134 chars
```

Verbose mode prints progress as events happen:

```text
ORA Research: Rust vs Go
  Intensity: 1 | Researcher: deepseek-v4-flash | Reviewer: deepseek-v4-pro

Generating research plan...
...
Approve this plan and begin research? [y/N]: y

🔎 Researcher: generated 1 search query
🔎 Researcher: searching "Rust vs Go"
✓ Researcher: found 5 URLs
🌐 Researcher: scraping reddit.com/r/rust/...
✗ Researcher: scrape failed for reddit.com/r/rust/...
🌐 Researcher: scraping bitfieldconsulting.com/posts/rust-vs-go
✓ Researcher: scraped 8040 chars from bitfieldconsulting.com
🧾 Researcher: evaluated source reliability
✍️ Writer: synthesizing report from 1 finding
✓ Writer: draft generated, 2134 chars

  Sources: 1 | Findings: 1 | Draft: 2134 chars
```

## Architecture

Add a small progress helper module:

```python
def emit_progress(config, message: str, kind: str = "info") -> None:
    callback = (config or {}).get("configurable", {}).get("progress_callback")
    if callback:
        callback({"message": message, "kind": kind})
```

Agents call `emit_progress()` with the `RunnableConfig` they already receive. This keeps agent nodes independent from the CLI and lets future renderers consume the same event stream.

The CLI passes a callback by default and only skips it in quiet mode:

```python
graph_config = {
    "configurable": {
        "progress_callback": print_progress_event,
    }
}
research_graph.invoke(plan_result, graph_config)
```

In quiet mode, no callback is passed and `emit_progress()` becomes a no-op.

## Components

### `ora/progress.py`

Responsibilities:

- Define `ProgressEvent` as a typed dictionary or simple dictionary convention.
- Provide `emit_progress(config, message, kind="info")`.
- Swallow callback exceptions only if needed to avoid breaking research because display failed. If swallowed, callback errors should not hide agent errors.

### `ora/cli.py`

Responsibilities:

- When `--quiet` is false, pass a progress callback and avoid using a spinner that conflicts with line output.
- When `--quiet` is true, keep the existing spinner behavior.
- Render event kinds consistently:
  - `search` → `🔎`
  - `scrape` → `🌐`
  - `success` → `✓`
  - `error` → `✗`
  - `write` → `✍️`
  - `info` → `•`

### `ora/agents/researcher.py`

Emit progress for:

- Search query generation.
- Each search query.
- URL count per search.
- Each scrape attempt.
- Scrape success/failure.
- Source evaluation.
- Final source/finding count.

Keep the existing returned `messages` log for post-run debugging and tests.

### `ora/agents/writer.py`

Emit progress for:

- Report synthesis start.
- Draft generation with character count.

## Data Flow

1. User runs `ora research "..."`.
2. CLI generates and displays the plan as it does today.
3. After approval, CLI builds the research graph and passes `progress_callback` in the graph config.
4. Researcher and writer nodes call `emit_progress()` as they work.
5. CLI prints each event immediately.
6. Final summary and report render as they do today.

## Error Handling

- Scrape failures should emit a visible `error` event but should not abort research if other URLs are available.
- Search failures should emit an `error` event and continue to the fallback finding behavior already present.
- Progress callback errors should not cause the research graph to fail. Display should be best-effort.
- Agent or graph exceptions should still surface normally.

## Testing

- Add unit tests for `emit_progress()`:
  - no callback does nothing
  - callback receives event dict
  - missing config is safe
- Add researcher tests using a fake callback if existing test fixtures can mock search/scrape cheaply.
- Add writer test using a fake callback and mocked LLM response if existing tests already mock writer behavior.
- Keep CLI output tests minimal to avoid brittle Rich/Click assertions.

## Future Extensions

The event structure can support a future Rich dashboard without changing researcher or writer code. A dashboard renderer would consume the same `progress_callback` events and render them differently.
