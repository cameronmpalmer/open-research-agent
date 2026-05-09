# Open Research Audit (ORA) -- v0 Design Spec

**Date:** 2026-05-09
**Status:** Draft, pending review

## 1. Overview

ORA is an open-source CLI research tool that produces calibrated, source-traced research reports with a built-in adversarial reviewer. It closes the LangChain/LangGraph knowledge gap and demonstrates production-quality multi-agent architecture.

**Core thesis:** LLM-generated research has a systematic problem -- plausible but unsupported claims. The solution is not better prompting or bigger models. It is structural: a separate agent with fresh context that is explicitly tasked with attacking the findings before the user ever sees them.

## 2. Architecture

### 2.1 High-Level Pipeline

```
User Query → Supervisor → Researcher → Writer → Adversarial Reviewer → [REVISE] → Final Report
                              ↑              ↑              │
                              └──────────────┴──────────────┘
                                    (revision loop, max 3 cycles)
```

### 2.2 Agents

| Agent | Role | Tools | Model |
|-------|------|-------|-------|
| **Supervisor** | Routes between agents based on state. Decides when done. | None (LLM routing via LangGraph conditional edges) | Fast/cheap (gpt-4.1-mini) |
| **Researcher** | Generates search queries, scrapes sources, evaluates quality, records findings with citations | `web_search` (Firecrawl), `scrape_page`, `evaluate_source` | Capable reasoning (gpt-4.1 / claude-sonnet) |
| **Writer** | Synthesizes research notes into a structured report with calibrated confidence levels | LLM generation only | Capable writing (gpt-4.1 / claude-sonnet) |
| **Adversarial Reviewer** | Attacks the report: verifies URLs exist, checks citations support claims, searches for contradicting evidence, calibrates confidence | `web_search` (Firecrawl), `check_url`, `scrape_page` | **Different model family from Researcher** (if Researcher uses OpenAI, Reviewer uses Anthropic, or vice versa) |

### 2.3 State (LangGraph StateGraph)

```python
class ResearchState(TypedDict):
    # Input
    query: str
    intensity: Literal[1, 2, 3, 4, 5]
    
    # Plan
    research_plan: Optional[ResearchPlan]
    plan_approved: bool
    
    # Research
    search_queries: list[str]
    sources: list[Source]
    findings: list[Finding]
    
    # Report
    draft_report: Optional[str]
    revision_count: int
    
    # Review
    review_verdict: Optional[Literal["PASS", "REVISE"]]
    review_blocking: list[str]
    review_required: list[str]
    review_suggested: list[str]
    
    # Output
    final_report: Optional[str]
```

### 2.4 Graph Structure

```
START
  │
  ▼
[Supervisor: Plan] ──→ generates research plan, search angles
  │
  ▼
[HITL: Plan Review] ──→ user can correct plan, then approve
  │
  ▼
[Researcher] ──→ parallel searches, source evaluation, finding extraction
  │
  ▼
[Writer] ──→ synthesizes findings into structured report
  │
  ▼
[Adversarial Reviewer] ──→ critiques report, returns PASS or REVISE
  │
  ├── PASS ──→ [Format Final Report] ──→ END
  │
  └── REVISE ──→ [check revision_count < 3]
                    │
                    ├── yes ──→ [Researcher] (targeted gap-fill)
                    └── no ──→ [Format Final Report] ──→ END
```

## 3. Component Details

### 3.1 Researcher Agent

**Responsibilities:**
1. Generate search queries from multiple angles (direct, opposing, specific, recent)
2. Execute searches via Firecrawl
3. Scrape top results for full content
4. Evaluate each source using CRAAP dimensions (Currency, Relevance, Authority, Accuracy, Purpose)
5. Record findings with explicit citations (URL, quote, publication date)
6. Flag contradictions between sources
7. Note evidence gaps

**Intensity scaling:**
| Level | Name | Queries | Sources Target | Parallel Searches |
|-------|------|---------|---------------|-------------------|
| 1 | Quick | 1-2 | 3-5 | Sequential |
| 2 | Standard | 3-4 | 8-15 | Sequential |
| 3 | Thorough | 5-7 | 15-30 | Parallel (3-5 concurrent) |
| 4 | Deep | 10-15 | 50-80 | Parallel (5-8 concurrent) |
| 5 | Exhaustive | 15-25 | 100-200 | Parallel + recursive crawl |

### 3.2 Adversarial Reviewer Agent

**This is ORA's primary differentiator.** The reviewer receives ONLY the completed draft report and the original query. It does NOT see the Researcher's reasoning, search queries, or intermediate findings. It uses a different model family from the Researcher.

**Review checklist (encoded as system prompt):**

1. **URL verification (BLOCKING if found):** Spot-check 3-5 cited URLs. Do they resolve? Does the page title/content correspond to what's cited?

2. **Opposing evidence search (REQUIRED if missing):** Search for evidence that contradicts key claims. Was this done? If not, do it now. Report any contradicting evidence found.

3. **Citation accuracy (BLOCKING if found):** For 2-3 key claims, follow the citation and verify the source actually supports the claim. Flag any misattributions.

4. **Confidence calibration (REQUIRED):** Are claims with weak or single-source evidence labeled appropriately (Low/Moderate)? Are claims with strong multi-source support labeled High?

5. **Evidence gaps (REQUIRED):** What did the report NOT find that the query asks for? Are gaps acknowledged?

6. **Contradiction documentation (REQUIRED):** If sources disagree, is this documented with both positions and their evidence?

7. **Completeness (SUGGESTED):** Does the report address all aspects of the query?

**Output format:**
```json
{
  "verdict": "PASS" | "REVISE",
  "blocking": ["..."],
  "required": ["..."],
  "suggested": ["..."],
  "contradicting_evidence_found": ["..."],
  "confidence_recalibrations": {"claim": "new_level"}
}
```

### 3.3 Source Evaluation (per source)

Every source the Researcher scrapes is rated:

```python
class SourceEvaluation:
    url: str
    title: str
    publication_date: Optional[str]
    source_type: Literal["academic_paper", "official_doc", "news", "blog", "forum", "social_media", "unknown"]
    
    # CRAAP dimensions (1-5 scale)
    currency: int
    relevance: int
    authority: int
    accuracy: int
    purpose: int
    
    overall_reliability: Literal["High", "Medium", "Low"]
    notes: str
```

### 3.4 Confidence Calibration (per finding)

Each claim in the final report carries:

| Level | Criteria | IPCC Language Equivalent |
|-------|----------|--------------------------|
| **High** | 2+ strong independent sources, no contradictions, direct evidence | Very likely (90-100%) |
| **Moderate** | Good source(s) support but single-source dependency or minor gaps | Likely (66-100%) |
| **Low** | Limited or conflicting evidence, significant uncertainty | About as likely as not (33-66%) |
| **Unknown** | No reliable evidence found; this is identified as a gap | Insufficient evidence |

### 3.5 Corroboration Tracking

The Researcher tracks per-claim corroboration:
- **Single-sourced claims** are flagged
- **Multi-sourced claims** list all corroborating sources
- **Corroboration chains** show how independent sources confirm each other
- Claims with zero corroborating sources and no original research to cite are flagged as **unverified** and must be downgraded to Low or Unknown

## 4. CLI Interface

### 4.1 Primary Command

```bash
ora research "What is the current state of fusion energy research?" \
  --intensity 3 \
  --output report.md \
  --model openai:gpt-4.1 \
  --reviewer-model anthropic:claude-sonnet-4-20250514
```

### 4.2 Subcommands

```bash
ora research <query>     # Run full research pipeline
ora plan <query>         # Generate research plan only (for review)
ora review <report.md>   # Run adversarial review on an existing report
ora config               # Show/set configuration
ora bench                # Run against Deep Research Bench
```

### 4.3 Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--intensity`, `-i` | 2 | Research intensity (1-5) |
| `--output`, `-o` | stdout | Output file (markdown) |
| `--model`, `-m` | openai:gpt-4.1-mini | LLM for researcher/writer |
| `--reviewer-model`, `-r` | (different family) | LLM for adversarial reviewer |
| `--search`, `-s` | firecrawl | Search backend |
| `--max-revisions` | 3 | Max writer-reviewer revision cycles |
| `--no-review` | false | Skip adversarial review |
| `--verbose`, `-v` | false | Show agent actions in real-time |

## 5. Output Format

### 5.1 Report Structure (Markdown)

```markdown
# Research: [Query]

**Intensity:** Level 3 | **Sources:** 22 | **Date:** 2026-05-09
**Reviewer:** PASS (0 blocking, 2 required resolved, 3 suggested)

## Executive Summary
[2-4 sentence overview]

## Key Findings

### [Subtopic]
**Finding:** [Claim]
**Confidence:** High | **Sources:** [Source 1](url), [Source 2](url)
**Corroborated by:** 2 independent sources
**Contradicting evidence:** [If any, with sources]

## Evidence Gaps
- [What we could not find]

## Source Table
| # | Title | URL | Type | Reliability | Accessed |
|---|-------|-----|------|-------------|----------|
| 1 | ... | ... | ... | High | 2026-05-09 |

## Reviewer Gate
- **Verdict:** PASS
- **Blocking issues resolved:** 0
- **Required issues resolved:** 2 (added contradicting evidence search, recalibrated confidence on claim 4)
- **Suggested issues noted:** 3

## Bibliography
1. [Author]. "[Title]." [Source]. [Date]. [URL]
```

## 6. Configuration

```yaml
# ~/.ora/config.yaml
models:
  default: openai:gpt-4.1-mini
  researcher: openai:gpt-4.1
  reviewer: anthropic:claude-sonnet-4-20250514

search:
  provider: firecrawl
  firecrawl_api_key: ${FIRECRAWL_API_KEY}

output:
  default_format: markdown
  always_include_sources: true
  
limits:
  max_revisions: 3
  default_intensity: 2
```

## 7. Project Structure

```
open-research-audit/
├── ora/
│   ├── __init__.py
│   ├── cli.py              # Click CLI entry point
│   ├── config.py           # Configuration loading (pydantic)
│   ├── state.py            # ResearchState TypedDict + reducer
│   ├── graph.py            # LangGraph StateGraph definition
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── supervisor.py   # Supervisor agent (routing logic)
│   │   ├── researcher.py   # Researcher agent (search + evaluate)
│   │   ├── writer.py       # Writer agent (synthesize report)
│   │   └── reviewer.py     # Adversarial reviewer agent
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── search.py       # Firecrawl search tool
│   │   ├── scrape.py       # Firecrawl scrape tool
│   │   └── evaluate.py     # Source evaluation (CRAAP)
│   └── prompts/
│       ├── supervisor.py   # System prompts
│       ├── researcher.py
│       ├── writer.py
│       └── reviewer.py
├── tests/
│   ├── test_graph.py
│   ├── test_researcher.py
│   ├── test_reviewer.py
│   └── test_integration.py
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-05-09-ora-v0-design.md
├── pyproject.toml
├── README.md
└── LICENSE
```

## 8. Dependencies

```toml
[project]
name = "ora"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "langchain>=0.3.0",
    "langgraph>=0.3.0",
    "langchain-openai",
    "langchain-anthropic",
    "langchain-firecrawl",
    "pydantic>=2.0",
    "click>=8.0",
    "pyyaml>=6.0",
    "rich>=13.0",          # Terminal formatting
    "python-dotenv",
]
```

## 9. v0 Scope Boundaries

### In scope
- Levels 1-3 (Quick, Standard, Thorough) with parallel searches at L3
- Supervisor → Researcher → Writer → Adversarial Reviewer pipeline
- Firecrawl search + scrape as tools
- Source evaluation (CRAAP dimensions)
- Confidence calibration (High/Moderate/Low/Unknown)
- Adversarial reviewer with cross-model support
- CLI with `ora research`, `ora plan`, `ora bench`
- Markdown report output with source table
- Human-in-the-loop plan review via LangGraph `interrupt()`
- Max 3 revision cycles (Writer ↔ Reviewer)
- Deep Research Bench submission format

### Out of scope (v0)
- Levels 4-5 (Deep, Exhaustive) -- too much infrastructure for v0
- MCP server -- orthogonal to core LangChain/LangGraph demo
- Web UI -- CLI first
- Streaming output during research
- Corroboration chain visualization (tracking exists, visualization doesn't)
- Multi-model ensemble for Researcher (single model per run)
- Caching of scraped pages
- Async subprocess dispatch (uses LangGraph's built-in parallelism)
- Docker / deployment config

## 10. Testing Strategy

### 10.1 Unit Tests
- Source evaluation scoring
- State transitions (graph routing logic)
- Prompt template rendering
- Report formatting
- Configuration loading

### 10.2 Integration Tests
- Full pipeline with mocked LLM responses (deterministic output testing)
- Firecrawl tool invocation (with real API, rate-limited)
- Adversarial reviewer catching known issues in intentionally flawed reports

### 10.3 Benchmark
- Deep Research Bench: 100 PhD-level tasks, RACE + FACT metrics
- Target: comparable or better RACE score than open_deep_research baseline (~0.43)
- Primary validation: FACT metric (citation accuracy) should improve with adversarial reviewer enabled vs disabled (A/B test)

## 11. Open Questions

1. **Parallel search implementation:** Should the Researcher use LangChain's `RunnableParallel` for L3 concurrent searches, or LangGraph's `Send` API for parallel node dispatch?
2. **Cost tracking:** Should v0 track and report per-run API costs?
3. **Reviewer model enforcement:** Should the tool refuse to run if researcher and reviewer use the same model family, or just warn?
4. **Intermediate state persistence:** Should research state survive across CLI invocations (for resuming interrupted research)?
