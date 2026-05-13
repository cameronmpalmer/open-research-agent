# Intensity Levels 4 and 5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement research intensity levels 4 (Deep) and 5 (Exhaustive) with per-level source scaling, multi-round gap targeting, and reviewer gate.

**Architecture:** Scale existing researcher node with per-level parameter tables for queries, scraping depth, content length, and minimum sources. Add a multi-round loop that generates gap-targeted queries when source counts fall short. Re-enable the reviewer gate for levels 4-5. Label reports interim when thresholds aren't met.

**Tech Stack:** Python, LangGraph, pytest, existing ora modules.

---

## File Structure

- Modify: `ora/agents/researcher.py` — per-level parameters, gap queries, multi-round loop, interim labeling
- Modify: `ora/graph.py` — conditionally enable reviewer for L4-L5
- Modify: `ora/cli.py` — remove L4-L5 fallback, accept levels 4-5 in help text and --intensity
- Modify: `ora/prompts/supervisor.py` — update plan prompt intensity descriptions for L4-L5
- Modify: `tests/test_researcher.py` — add L4-L5 query generation tests
- Create: `tests/test_researcher_intensity.py` — per-level parameter tests, gap logic, interim labeling
- Modify: `tests/test_cli.py` — verify L4-L5 not rejected, update help assertions
- Create: `tests/test_graph_intensity.py` — verify reviewer enabled for L4-L5

---

### Task 1: Per-level parameter table and query scaling

**Files:**
- Modify: `ora/agents/researcher.py`
- Modify: `tests/test_researcher.py`
- Create: `tests/test_researcher_intensity.py`

- [ ] **Step 1: Write failing test for L4-L5 query counts**

Create `tests/test_researcher_intensity.py` with:

```python
"""Tests for intensity level 4-5 behavior."""
from ora.agents.researcher import (
    generate_search_queries,
    LEVEL_PARAMS,
    generate_gap_queries,
)


class TestLevelParams:
    def test_level_1_params(self):
        p = LEVEL_PARAMS[1]
        assert p["min_sources"] == 3
        assert p["max_rounds"] == 1
        assert p["urls_per_query"] == 3
        assert p["scrapes_per_query"] == 2
        assert p["max_content_chars"] == 8000

    def test_level_4_params(self):
        p = LEVEL_PARAMS[4]
        assert p["min_sources"] == 50
        assert p["max_rounds"] == 3
        assert p["urls_per_query"] == 8
        assert p["scrapes_per_query"] == 4
        assert p["max_content_chars"] == 12000

    def test_level_5_params(self):
        p = LEVEL_PARAMS[5]
        assert p["min_sources"] == 100
        assert p["max_rounds"] == 4
        assert p["urls_per_query"] == 10
        assert p["scrapes_per_query"] == 5
        assert p["max_content_chars"] == 16000

    def test_level_4_queries_exist(self):
        queries = generate_search_queries("test", intensity=4)
        assert len(queries) >= 8
        assert "detailed analysis" in " ".join(queries).lower()

    def test_level_5_queries_exist(self):
        queries = generate_search_queries("test", intensity=5)
        assert len(queries) >= 12
```

- [ ] **Step 2: Verify tests fail**

Run: `python3 -m pytest tests/test_researcher_intensity.py -q`
Expected: FAIL (LEVEL_PARAMS / generate_gap_queries not defined)

- [ ] **Step 3: Add LEVEL_PARAMS table and update generate_search_queries**

In `ora/agents/researcher.py`, add after the SKIP_DOMAINS declaration:

```python
LEVEL_PARAMS = {
    1: {"min_sources": 3, "max_rounds": 1, "urls_per_query": 3, "scrapes_per_query": 2, "max_content_chars": 8000},
    2: {"min_sources": 8, "max_rounds": 1, "urls_per_query": 3, "scrapes_per_query": 2, "max_content_chars": 8000},
    3: {"min_sources": 15, "max_rounds": 2, "urls_per_query": 5, "scrapes_per_query": 3, "max_content_chars": 10000},
    4: {"min_sources": 50, "max_rounds": 3, "urls_per_query": 8, "scrapes_per_query": 4, "max_content_chars": 12000},
    5: {"min_sources": 100, "max_rounds": 4, "urls_per_query": 10, "scrapes_per_query": 5, "max_content_chars": 16000},
}
```

Update `generate_search_queries` to return 8+ queries for L4 and 12+ for L5:

```python
def generate_search_queries(query: str, intensity: int) -> list[str]:
    """Generate search queries based on intensity level."""
    angles = {
        1: ["{query}"],
        2: ["{query}", "{query} latest", "opposing view on {query}"],
        3: [
            "{query}",
            "{query} latest research 2025 2026",
            'critique of {query}',
            '"{query}" expert analysis',
            '{query} vs alternatives',
            '{query} best practices',
            'problems with {query}',
        ],
        4: [
            "{query}",
            "{query} latest research 2025 2026",
            'critique of {query}',
            '"{query}" expert analysis',
            '{query} vs alternatives',
            '{query} best practices',
            'problems with {query}',
            "{query} detailed analysis",
            "recent developments in {query}",
            "opposing view on {query}",
            "industry perspective on {query}",
            "academic perspective on {query}",
        ],
        5: [
            "{query}",
            "{query} latest research 2025 2026",
            'critique of {query}',
            '"{query}" expert analysis',
            '{query} vs alternatives',
            '{query} best practices',
            'problems with {query}',
            "{query} detailed analysis",
            "recent developments in {query}",
            "opposing view on {query}",
            "industry perspective on {query}",
            "academic perspective on {query}",
            "{query} statistics and data",
            "{query} case studies",
            "limitations of {query}",
            "controversies about {query}",
        ],
    }
    templates = angles.get(intensity, angles[2])
    return [t.format(query=query) for t in templates]
```

- [ ] **Step 4: Verify tests pass**

Run: `python3 -m pytest tests/test_researcher_intensity.py tests/test_researcher.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ora/agents/researcher.py tests/test_researcher.py tests/test_researcher_intensity.py
git commit -m "feat: add per-level parameter table and L4-L5 query sets"
```

---

### Task 2: Gap-targeted queries and multi-round loop

**Files:**
- Modify: `ora/agents/researcher.py`
- Modify: `tests/test_researcher_intensity.py`

- [ ] **Step 1: Write failing test for gap query generation**

Append to `tests/test_researcher_intensity.py`:

```python
class TestGapQueries:
    def test_generates_at_least_min_gap_queries(self):
        queries = generate_gap_queries("AI safety", intensity=4)
        assert len(queries) >= 5

    def test_gap_queries_include_varied_angles(self):
        queries = generate_gap_queries("AI safety", intensity=4)
        joined = " ".join(queries).lower()
        assert any(phrase in joined for phrase in ["detailed", "expert", "recent"])

    def test_gap_queries_level_5_produces_more(self):
        q4 = generate_gap_queries("AI safety", intensity=4)
        q5 = generate_gap_queries("AI safety", intensity=5)
        assert len(q5) >= len(q4)
```

- [ ] **Step 2: Verify gap tests fail**

Run: `python3 -m pytest tests/test_researcher_intensity.py::TestGapQueries -q`
Expected: FAIL (generate_gap_queries not defined)

- [ ] **Step 3: Add generate_gap_queries**

In `ora/agents/researcher.py`, add after `generate_search_queries`:

```python
def generate_gap_queries(query: str, intensity: int) -> list[str]:
    """Generate gap-targeted queries when source counts fall short."""
    bases = {
        2: [
            "{query} detailed analysis",
            "key aspects of {query}",
        ],
        3: [
            "{query} detailed analysis",
            "key aspects of {query}",
            "expert review of {query}",
        ],
        4: [
            "{query} detailed analysis",
            "key aspects of {query}",
            "expert review of {query}",
            "recent developments in {query}",
            "opposing view on {query}",
        ],
        5: [
            "{query} detailed analysis",
            "key aspects of {query}",
            "expert review of {query}",
            "recent developments in {query}",
            "opposing view on {query}",
            "{query} statistics and data",
            "{query} case studies",
        ],
    }
    templates = bases.get(intensity, bases[2])
    return [t.format(query=query) for t in templates]
```

- [ ] **Step 4: Verify gap tests pass**

Run: `python3 -m pytest tests/test_researcher_intensity.py -q`
Expected: PASS

- [ ] **Step 5: Add multi-round loop and interim labeling to researcher_node**

In `ora/agents/researcher.py`, refactor the search logic into a multi-round loop. After the existing imports, modify `researcher_node` to:

- Read level params from `LEVEL_PARAMS`
- Replace the single `for q in queries` loop with an outer round loop
- After each round, check `len(sources) >= params["min_sources"]`
- If shortfall and rounds remain, call `generate_gap_queries` for the next round
- After max rounds, if still short, set `research_status = "interim"` in the return dict

No code shown here — implement to match the existing researcher style with the round loop.

- [ ] **Step 6: Run all researcher and intensity tests**

Run: `python3 -m pytest tests/test_researcher.py tests/test_researcher_intensity.py tests/test_researcher_progress.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add ora/agents/researcher.py tests/test_researcher_intensity.py
git commit -m "feat: add gap-targeted queries and multi-round research loop"
```

---

### Task 3: Remove L4-L5 CLI fallback and update help text

**Files:**
- Modify: `ora/cli.py`
- Modify: `ora/prompts/supervisor.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing test that L4 is accepted**

Append to `tests/test_cli.py`:

```python
def test_research_accepts_intensity_4(self, monkeypatch):
    from ora import cli as cli_module

    intensity_seen = []

    class FakePlanGraph:
        def invoke(self, state, config=None):
            intensity_seen.append(state.get("intensity"))
            return {"research_plan": "# Plan", "plan_approved": False, "messages": ["# Plan"]}

    class FakeResearchGraph:
        def invoke(self, state, config=None):
            return {"draft_report": "# Research\nbody", "sources": [], "findings": []}

    monkeypatch.setattr(cli_module, "load_config", lambda: _fake_settings())
    monkeypatch.setattr(cli_module, "_spin", lambda func, message="Working...": func())
    monkeypatch.setattr(cli_module, "_print_markdown", lambda text: None)
    monkeypatch.setattr(cli_module.click, "prompt", lambda *a, **kw: "A")
    monkeypatch.setattr("ora.graph.build_plan_graph", lambda: FakePlanGraph())
    monkeypatch.setattr("ora.graph.build_research_graph", lambda: FakeResearchGraph())

    runner = CliRunner()
    result = runner.invoke(main, ["research", "test", "-i", "4"])

    assert result.exit_code == 0
    assert intensity_seen and intensity_seen[0] == 4
```

- [ ] **Step 2: Verify test fails**

Run: `python3 -m pytest tests/test_cli.py::TestCLI::test_research_accepts_intensity_4 -q`
Expected: FAIL (intensity_seen[0] == 3, not 4)

- [ ] **Step 3: Remove intensity fallback and update help text**

In `ora/cli.py`:

Remove the `if intensity > 3` fallback block (lines 140-142) from both `research` and `plan`.

Update the `--intensity` help text:

```python
@click.option("--intensity", "-i", type=click.IntRange(1, 5), default=2,
              help="Research intensity (1=Quick, 2=Standard, 3=Thorough, 4=Deep, 5=Exhaustive)")
```

Do this for both `research` and `plan` commands.

Update `ora/prompts/supervisor.py`, change both prompt template intensity lines from:
```
Intensity level: {intensity} (1=Quick, 2=Standard, 3=Thorough)
```
to:
```
Intensity level: {intensity} (1=Quick, 2=Standard, 3=Thorough, 4=Deep, 5=Exhaustive)
```

- [ ] **Step 4: Verify CLI tests pass**

Run: `python3 -m pytest tests/test_cli.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ora/cli.py ora/prompts/supervisor.py tests/test_cli.py
git commit -m "feat: enable intensity levels 4 and 5 in CLI"
```

---

### Task 4: Enable reviewer gate for L4-L5

**Files:**
- Modify: `ora/graph.py`
- Modify: `ora/cli.py`

- [ ] **Step 1: Add conditional reviewer node to build_research_graph**

Modify `ora/graph.py` `build_research_graph` to accept an `intensity` parameter. When intensity >= 4, include the reviewer node in the graph. When < 4, skip it (current behavior).

```python
def build_research_graph(intensity: int = 2) -> StateGraph:
```

- [ ] **Step 2: Update CLI to pass intensity to build_research_graph**

In `ora/cli.py`, pass `intensity=intensity` to `build_research_graph(intensity=intensity)`.

- [ ] **Step 3: Run all tests**

Run: `python3 -m pytest tests/ -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add ora/graph.py ora/cli.py
git commit -m "feat: enable reviewer gate for intensity levels 4 and 5"
```

---

### Task 5: Full integration smoke test

- [ ] **Step 1: Run full suite**

```bash
python3 -m pytest tests/ -q
```

- [ ] **Step 2: Smoke L4 with fake harness**

```bash
printf 'A\n' | PYTHONPATH="/tmp/opencode/ora_smoke_patch:..." python3 -m ora.cli research "Rust vs Go" --intensity 4 --no-save
```

Expected: plan and research proceed without fallback warning; progress shows multiple rounds.

- [ ] **Step 3: Commit if fixes needed**

```bash
git add -A && git commit -m "fix: polish intensity 4-5 integration"
```
