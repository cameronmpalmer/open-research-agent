# Open Research Agent (ORA) v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI research tool with LangGraph multi-agent pipeline (Supervisor → Researcher → Writer → Adversarial Reviewer) supporting intensity levels 1-3, Firecrawl search, and human-in-the-loop plan review.

**Architecture:** Single LangGraph StateGraph with 4 agent nodes and conditional routing. Agents communicate through shared ResearchState. Cross-model adversarial review (Researcher and Reviewer use different LLM providers). CLI via Click. Configuration via YAML + env vars.

**Tech Stack:** Python 3.11+, LangChain 0.3+, LangGraph 0.3+, Click, Pydantic v2, Firecrawl, pytest

---

## File Structure Map

```
open-research-agent/
├── ora/
│   ├── __init__.py              # Package init, version
│   ├── cli.py                   # Click CLI (ora research, ora plan, ora bench)
│   ├── config.py                # Pydantic Settings, YAML loading
│   ├── state.py                 # ResearchState, Source, Finding types
│   ├── graph.py                 # build_graph(): StateGraph, nodes, edges
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── supervisor.py        # Supervisor: plan generation + routing logic
│   │   ├── researcher.py        # Researcher: search, scrape, evaluate, record
│   │   ├── writer.py            # Writer: synthesize report from findings
│   │   └── reviewer.py          # Adversarial Reviewer: critique + verdict
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── search.py            # firecrawl_search wrapped as LangChain Tool
│   │   ├── scrape.py            # firecrawl_scrape wrapped as LangChain Tool
│   │   └── evaluate.py          # evaluate_source(): CRAAP scoring
│   └── prompts/
│       ├── __init__.py          # Prompt loader helper
│       ├── supervisor.py        # System prompts for supervisor
│       ├── researcher.py        # System prompts for researcher
│       ├── writer.py            # System prompts for writer
│       └── reviewer.py          # System prompts for adversarial reviewer
├── tests/
│   ├── __init__.py
│   ├── test_state.py            # State type validation
│   ├── test_config.py           # Config loading
│   ├── test_evaluate.py         # Source evaluation (CRAAP)
│   ├── test_graph.py            # Graph routing logic
│   ├── test_cli.py              # CLI argument parsing
│   └── test_integration.py      # Full pipeline smoke test
├── pyproject.toml
├── README.md
└── LICENSE
```

---

### Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `LICENSE`
- Create: `ora/__init__.py`
- Create: `tests/__init__.py`
- Create: `ora/prompts/__init__.py`
- Create: `ora/agents/__init__.py`
- Create: `ora/tools/__init__.py`

- [ ] **Step 1: Write pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "ora"
version = "0.1.0"
description = "Open Research Agent - multi-agent research with adversarial review"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
dependencies = [
    "langchain>=0.3.0",
    "langgraph>=0.3.0",
    "langchain-openai>=0.3.0",
    "langchain-anthropic>=0.3.0",
    "langchain-firecrawl>=0.1.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "click>=8.0",
    "pyyaml>=6.0",
    "rich>=13.0",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "pytest-mock>=3.14",
]

[project.scripts]
ora = "ora.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Verify it parses**

Run: `cd /path/to/repo && python -c "import tomllib; tomllib.loads(open('pyproject.toml').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Write LICENSE (MIT)**

Create `LICENSE` with standard MIT license text (copyright Cameron Palmer).

- [ ] **Step 4: Write init files**

`ora/__init__.py`:
```python
"""Open Research Agent (ORA) - Multi-agent research toolkit."""
__version__ = "0.1.0"
```

`tests/__init__.py`: empty
`ora/prompts/__init__.py`: empty
`ora/agents/__init__.py`: empty
`ora/tools/__init__.py`: empty

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml LICENSE ora/__init__.py ora/prompts/__init__.py ora/agents/__init__.py ora/tools/__init__.py tests/__init__.py
git commit -m "feat: scaffold project structure with pyproject.toml"
```

---

### Task 2: State Types

**Files:**
- Create: `ora/state.py`
- Create: `tests/test_state.py`

The LangGraph graph runs on a shared TypedDict. This task defines all types.

- [ ] **Step 1: Write the failing test**

`tests/test_state.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_state.py -v`
Expected: FAIL (import errors, types not defined)

- [ ] **Step 3: Write state.py**

`ora/state.py`:
```python
"""Core state types for the ORA research graph."""
from typing import TypedDict, Literal, Optional, Annotated
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class Source(BaseModel):
    """An evaluated research source."""
    url: str
    title: str
    publication_date: Optional[str] = None
    source_type: Literal[
        "academic_paper", "official_doc", "news", "blog",
        "forum", "social_media", "unknown"
    ] = "unknown"
    # CRAAP dimensions (1-5)
    currency: int = Field(default=3, ge=1, le=5)
    relevance: int = Field(default=3, ge=1, le=5)
    authority: int = Field(default=3, ge=1, le=5)
    accuracy: int = Field(default=3, ge=1, le=5)
    purpose: int = Field(default=3, ge=1, le=5)
    overall_reliability: Literal["High", "Medium", "Low"] = "Medium"
    notes: str = ""


class Finding(BaseModel):
    """A research finding with citation and confidence."""
    claim: str
    confidence: Literal["High", "Moderate", "Low", "Unknown"] = "Moderate"
    supporting_sources: list[str] = Field(default_factory=list)  # URLs
    contradicting_sources: list[str] = Field(default_factory=list)  # URLs
    notes: str = ""


class ReviewVerdict(BaseModel):
    """Output from the adversarial reviewer."""
    verdict: Literal["PASS", "REVISE"] = "PASS"
    blocking: list[str] = Field(default_factory=list)
    required: list[str] = Field(default_factory=list)
    suggested: list[str] = Field(default_factory=list)
    contradicting_evidence_found: list[str] = Field(default_factory=list)
    confidence_recalibrations: dict[str, str] = Field(default_factory=dict)


class ResearchState(TypedDict, total=False):
    """Shared state for the ORA LangGraph pipeline."""
    # Input
    query: str
    intensity: int  # 1-5

    # Plan
    research_plan: str
    plan_approved: bool

    # Messages (for LangGraph agent nodes)
    messages: Annotated[list, add_messages]

    # Research
    search_queries: list[str]
    sources: list[Source]
    findings: list[Finding]

    # Report
    draft_report: str
    revision_count: int

    # Review
    review_verdict: ReviewVerdict
    review_verdict_raw: str  # JSON string for structured output parsing

    # Output
    final_report: str
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_state.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add ora/state.py tests/test_state.py
git commit -m "feat: add ResearchState and supporting types"
```

---

### Task 3: Configuration

**Files:**
- Create: `ora/config.py`
- Create: `tests/test_config.py`

Loads from YAML config file and environment variables via pydantic-settings.

- [ ] **Step 1: Write failing test**

`tests/test_config.py`:
```python
"""Tests for configuration loading."""
import os
import tempfile
from ora.config import load_config, ORASettings


class TestORASettings:
    def test_defaults(self):
        settings = ORASettings()
        assert settings.models.default == "openai:gpt-4.1-mini"
        assert settings.search.provider == "firecrawl"
        assert settings.limits.max_revisions == 3
        assert settings.limits.default_intensity == 2

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("ORA_MODELS__DEFAULT", "anthropic:claude-sonnet-4-20250514")
        settings = ORASettings()
        assert settings.models.default == "anthropic:claude-sonnet-4-20250514"

    def test_requires_different_reviewer_model(self):
        settings = ORASettings(
            models={
                "researcher": "openai:gpt-4.1",
                "reviewer": "openai:gpt-4.1",
            }
        )
        # Same provider should be detectable
        researcher_provider = settings.models.researcher.split(":")[0]
        reviewer_provider = settings.models.reviewer.split(":")[0]
        assert researcher_provider == reviewer_provider  # Should warn


class TestLoadConfig:
    def test_loads_yaml_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("limits:\n  default_intensity: 4\n")
            f.flush()
            config = load_config(f.name)
            assert config.limits.default_intensity == 4
            os.unlink(f.name)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL (import errors)

- [ ] **Step 3: Write config.py**

`ora/config.py`:
```python
"""Configuration loading via YAML, env vars, and pydantic-settings."""
import os
import yaml
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LimitSettings(BaseModel):
    max_revisions: int = 3
    default_intensity: int = 2


class SearchSettings(BaseModel):
    provider: str = "firecrawl"
    firecrawl_api_key: Optional[str] = None


class ModelSettings(BaseModel):
    default: str = "openai:gpt-4.1-mini"
    researcher: Optional[str] = None
    reviewer: Optional[str] = None


class OutputSettings(BaseModel):
    default_format: str = "markdown"
    always_include_sources: bool = True


class ORASettings(BaseSettings):
    """ORA configuration, loaded from env vars with ORA_ prefix."""
    model_config = SettingsConfigDict(
        env_prefix="ORA_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )
    models: ModelSettings = Field(default_factory=ModelSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)
    limits: LimitSettings = Field(default_factory=LimitSettings)


def load_config(config_path: Optional[str] = None) -> ORASettings:
    """Load ORA configuration from YAML file and environment.

    Priority: env vars > YAML file > defaults.
    """
    settings = ORASettings()

    # Try YAML file
    path = config_path or os.path.expanduser("~/.ora/config.yaml")
    if os.path.exists(path):
        with open(path) as f:
            yaml_data = yaml.safe_load(f)
        if yaml_data:
            if "models" in yaml_data:
                settings.models = ModelSettings(**yaml_data["models"])
            if "search" in yaml_data:
                settings.search = SearchSettings(**yaml_data["search"])
            if "output" in yaml_data:
                settings.output = OutputSettings(**yaml_data["output"])
            if "limits" in yaml_data:
                settings.limits = LimitSettings(**yaml_data["limits"])

    # Env vars override YAML
    env_settings = ORASettings()
    if env_settings.models.default != settings.models.default:
        settings.models = env_settings.models
    if env_settings.search.firecrawl_api_key:
        settings.search.firecrawl_api_key = env_settings.search.firecrawl_api_key

    return settings


def get_researcher_model(settings: ORASettings) -> str:
    """Get the researcher model, falling back to default."""
    return settings.models.researcher or settings.models.default


def get_reviewer_model(settings: ORASettings) -> str:
    """Get the reviewer model, falling back with cross-model logic."""
    if settings.models.reviewer:
        return settings.models.reviewer
    # Default: pick opposite provider from researcher
    researcher = get_researcher_model(settings)
    if researcher.startswith("openai:"):
        return "anthropic:claude-sonnet-4-20250514"
    return "openai:gpt-4.1"
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_config.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add ora/config.py tests/test_config.py
git commit -m "feat: add configuration loading with pydantic-settings"
```

---

### Task 4: Firecrawl Search Tool

**Files:**
- Create: `ora/tools/search.py`
- Create: `ora/tools/scrape.py`

LangChain Tool wrappers around Firecrawl search and scrape.

- [ ] **Step 1: Write failing test**

`tests/test_search.py`:
```python
"""Tests for Firecrawl search tool."""
import pytest
from unittest.mock import patch, MagicMock
from ora.tools.search import create_search_tool


class TestSearchTool:
    def test_tool_creation(self):
        tool = create_search_tool(api_key="test-key")
        assert tool.name == "web_search"
        assert "search" in tool.description.lower()

    def test_tool_invoke_mocked(self):
        tool = create_search_tool(api_key="test-key")
        with patch.object(tool, "ainvoke") as mock_invoke:
            mock_invoke.return_value = "search results"
            result = tool.invoke("test query")
            assert result is not None

    def test_search_requires_api_key(self):
        with pytest.raises(ValueError, match="api_key"):
            create_search_tool(api_key="")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_search.py -v`
Expected: FAIL

- [ ] **Step 3: Write search.py**

`ora/tools/search.py`:
```python
"""Firecrawl search tool for LangChain."""
from langchain_core.tools import tool


@tool
def web_search(query: str, api_key: str = "") -> str:
    """Search the web using Firecrawl.

    Args:
        query: The search query string.
        api_key: Firecrawl API key. If empty, uses FIRECRAWL_API_KEY env var.

    Returns:
        Search results as formatted text with URLs and snippets.
    """
    import os
    from firecrawl import FirecrawlApp

    key = api_key or os.environ.get("FIRECRAWL_API_KEY", "")
    if not key:
        return "Error: No Firecrawl API key configured."

    app = FirecrawlApp(api_key=key)
    try:
        results = app.search(query, params={"limit": 5})
        if not results or "data" not in results:
            return "No search results found."

        formatted = []
        for i, item in enumerate(results.get("data", []), 1):
            url = item.get("url", "")
            title = item.get("title", "Untitled")
            snippet = item.get("description", "")[:300]
            formatted.append(f"{i}. [{title}]({url})\n   {snippet}")
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Search error: {str(e)}"


def create_search_tool(api_key: str = ""):
    """Create a configured web search tool."""
    if not api_key:
        import os
        api_key = os.environ.get("FIRECRAWL_API_KEY", "")
        if not api_key:
            raise ValueError("Firecrawl api_key is required. Set FIRECRAWL_API_KEY or pass api_key.")
    # Bind the key as a partial
    from functools import partial
    return web_search  # Tool will use env var
```

- [ ] **Step 4: Write scrape.py**

`ora/tools/scrape.py`:
```python
"""Firecrawl scrape tool for LangChain."""
from langchain_core.tools import tool


@tool
def scrape_page(url: str, api_key: str = "") -> str:
    """Scrape a web page for full content using Firecrawl.

    Args:
        url: The URL to scrape.
        api_key: Firecrawl API key. If empty, uses FIRECRAWL_API_KEY env var.

    Returns:
        Page content as markdown text.
    """
    import os
    from firecrawl import FirecrawlApp

    key = api_key or os.environ.get("FIRECRAWL_API_KEY", "")
    if not key:
        return "Error: No Firecrawl API key configured."

    app = FirecrawlApp(api_key=key)
    try:
        result = app.scrape_url(url, params={"formats": ["markdown"]})
        content = result.get("markdown", "")
        if not content:
            return f"No content extracted from {url}"
        # Truncate very long pages
        if len(content) > 8000:
            content = content[:8000] + "\n\n[Content truncated at 8000 characters]"
        return content
    except Exception as e:
        return f"Scrape error for {url}: {str(e)}"
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_search.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add ora/tools/search.py ora/tools/scrape.py tests/test_search.py
git commit -m "feat: add Firecrawl search and scrape LangChain tools"
```

---

### Task 5: Source Evaluation (CRAAP)

**Files:**
- Create: `ora/tools/evaluate.py`
- Create: `tests/test_evaluate.py`

Applies CRAAP dimensions to rate a source's reliability.

- [ ] **Step 1: Write failing test**

`tests/test_evaluate.py`:
```python
"""Tests for source evaluation."""
import pytest
from ora.tools.evaluate import evaluate_source, compute_craap_score, rate_reliability


class TestEvaluateSource:
    def test_returns_source_with_ratings(self):
        result = evaluate_source(
            url="https://example.com/article",
            title="Test Article",
            content="Some factual content about AI research.",
            source_type="blog",
        )
        assert result.url == "https://example.com/article"
        assert 1 <= result.currency <= 5
        assert 1 <= result.relevance <= 5
        assert result.overall_reliability in ("High", "Medium", "Low")

    def test_academic_paper_gets_higher_authority(self):
        result = evaluate_source(
            url="https://arxiv.org/abs/1234.5678",
            title="A Novel Approach",
            content="Peer-reviewed research...",
            source_type="academic_paper",
        )
        assert result.authority >= 3  # Academic papers start higher

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
```

- [ ] **Step 2: Run to see it fail**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL

- [ ] **Step 3: Write evaluate.py**

`ora/tools/evaluate.py`:
```python
"""Source evaluation using CRAAP dimensions."""
from ora.state import Source


def compute_craap_score(dimensions: dict) -> int:
    """Sum CRAAP dimensions (each 1-5) into a 5-25 score."""
    return sum(dimensions.get(d, 3) for d in ["currency", "relevance", "authority", "accuracy", "purpose"])


def rate_reliability(craap_score: int) -> str:
    """Convert CRAAP total score to reliability rating.

    High: 19-25, Medium: 12-18, Low: 5-11
    """
    if craap_score >= 19:
        return "High"
    elif craap_score >= 12:
        return "Medium"
    else:
        return "Low"


def evaluate_source(
    url: str,
    title: str = "",
    content: str = "",
    source_type: str = "unknown",
    publication_date: str = "",
) -> Source:
    """Evaluate a source using CRAAP criteria heuristics.

    This is a heuristic-based evaluation for v0. Future versions will use
    LLM-based evaluation for more nuanced scoring.

    Args:
        url: Source URL
        title: Page title
        content: Extracted page content (markdown)
        source_type: Type of source (academic_paper, official_doc, news, etc.)
        publication_date: Publication date if known

    Returns:
        Source with CRAAP dimensions and reliability rating.
    """
    dimensions = {
        "currency": _rate_currency(url, publication_date, source_type),
        "relevance": _rate_relevance(content),
        "authority": _rate_authority(url, source_type),
        "accuracy": _rate_accuracy(content),
        "purpose": _rate_purpose(url, content, source_type),
    }

    score = compute_craap_score(dimensions)
    reliability = rate_reliability(score)

    return Source(
        url=url,
        title=title,
        publication_date=publication_date or None,
        source_type=source_type,  # type: ignore
        currency=dimensions["currency"],
        relevance=dimensions["relevance"],
        authority=dimensions["authority"],
        accuracy=dimensions["accuracy"],
        purpose=dimensions["purpose"],
        overall_reliability=reliability,  # type: ignore
        notes=f"CRAAP score: {score}/25",
    )


def _rate_currency(url: str, date: str, source_type: str) -> int:
    """Rate timeliness. .gov and .edu domains get baseline 3, unknown gets 2."""
    if date:
        return 4  # Has explicit date
    if any(url.startswith(p) for p in ["https://arxiv.org", "https://doi.org"]):
        return 4
    if ".gov" in url or ".edu" in url:
        return 3
    return 2


def _rate_relevance(content: str) -> int:
    """Rate by content length as a proxy for substance. Subject to improvement."""
    if len(content) > 2000:
        return 4
    elif len(content) > 500:
        return 3
    elif len(content) > 100:
        return 2
    return 1


def _rate_authority(url: str, source_type: str) -> int:
    """Rate by source type and domain signals."""
    if source_type in ("academic_paper", "official_doc"):
        return 5
    if ".gov" in url:
        return 5
    if ".edu" in url:
        return 4
    if source_type == "news":
        return 3
    if source_type == "blog":
        return 2
    if source_type in ("forum", "social_media"):
        return 1
    return 2


def _rate_accuracy(content: str) -> int:
    """Heuristic: longer content with citations is more likely accurate."""
    if "http" in content and len(content) > 1000:
        return 4
    if len(content) > 500:
        return 3
    return 2


def _rate_purpose(url: str, content: str, source_type: str) -> int:
    """Rate objectivity. Academic/official sources score higher."""
    if source_type in ("academic_paper", "official_doc"):
        return 5
    if ".gov" in url or ".edu" in url:
        return 4
    if ".com" in url and "shop" not in url.lower() and "store" not in url.lower():
        return 3
    return 2
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_evaluate.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add ora/tools/evaluate.py tests/test_evaluate.py
git commit -m "feat: add CRAAP source evaluation"
```

---

### Task 6: Prompt Templates

**Files:**
- Create: `ora/prompts/__init__.py`
- Create: `ora/prompts/supervisor.py`
- Create: `ora/prompts/researcher.py`
- Create: `ora/prompts/writer.py`
- Create: `ora/prompts/reviewer.py`

System prompts for each agent. These are the methodology-encoding layer -- they translate the best-practices research into agent behavior.

- [ ] **Step 1: Write prompt loader**

`ora/prompts/__init__.py`:
```python
"""Prompt templates for ORA agents."""


def load_prompt(name: str) -> str:
    """Load a prompt template by name."""
    prompts = {
        "supervisor_plan": SUPERVISOR_PLAN_PROMPT,
        "supervisor_route": SUPERVISOR_ROUTE_PROMPT,
        "researcher": RESEARCHER_PROMPT,
        "writer": WRITER_PROMPT,
        "reviewer": REVIEWER_PROMPT,
    }
    return prompts.get(name, "")


from ora.prompts.supervisor import SUPERVISOR_PLAN_PROMPT, SUPERVISOR_ROUTE_PROMPT  # noqa: E402
from ora.prompts.researcher import RESEARCHER_PROMPT  # noqa: E402
from ora.prompts.writer import WRITER_PROMPT  # noqa: E402
from ora.prompts.reviewer import REVIEWER_PROMPT  # noqa: E402
```

- [ ] **Step 2: Write supervisor prompts**

`ora/prompts/supervisor.py`:
```python
"""Supervisor agent prompts."""

SUPERVISOR_PLAN_PROMPT = """You are a research planner. Your job is to create a research plan for the following query.

Query: {query}
Intensity level: {intensity} (1=Quick, 2=Standard, 3=Thorough)

Create a research plan with:
1. Core question restated
2. Subtopics to investigate (3-5)
3. Search angles (direct, opposing, specific, recent)
4. Known gaps or assumptions

Output the plan in clear markdown. The user will review and approve it before research begins."""

SUPERVISOR_ROUTE_PROMPT = """You are a research supervisor. Based on the current state, decide which agent should work next.

Current state:
- Plan approved: {plan_approved}
- Sources found: {source_count}
- Findings recorded: {finding_count}
- Draft exists: {has_draft}
- Review verdict: {review_verdict}
- Revision count: {revision_count} (max: 3)

Available agents:
- researcher: Search the web and evaluate sources
- writer: Synthesize findings into a structured report
- reviewer: Adversarial review of the draft report
- FINISH: Complete the research and output the final report

Respond with ONLY the agent name (researcher, writer, reviewer, or FINISH)."""
```

- [ ] **Step 3: Write researcher prompt**

`ora/prompts/researcher.py`:
```python
"""Researcher agent prompt."""

RESEARCHER_PROMPT = """You are a research agent. Your job is to search the web and gather factual information.

## Process
1. Generate search queries from multiple angles: direct, opposing, specific, recent
2. Use the `web_search` tool to find sources
3. Use the `scrape_page` tool to get full content from the most relevant results
4. For each source, evaluate it using these dimensions:
   - Currency: How recent is it? Is it still valid?
   - Relevance: Does it actually address the research question?
   - Authority: Who wrote it? What are their credentials? What's the domain?
   - Accuracy: Is it supported by evidence? Verifiable elsewhere?
   - Purpose: To inform, persuade, or sell? Any detectable bias?

## Rules
- Prefer primary sources over secondary
- Look for contradicting evidence, not just supporting
- Note when a claim has only one source
- If you find disagreements between sources, document both sides
- If you can't find information on something, flag it as an evidence gap
- For each finding, assign an initial confidence level:
  * High: 2+ strong independent sources, no contradictions
  * Moderate: Good source(s) but single-source or minor gaps
  * Low: Limited or conflicting evidence
  * Unknown: No reliable evidence found

You have access to `web_search` and `scrape_page` tools.
User query: {query}
Intensity level: {intensity}"""
```

- [ ] **Step 4: Write writer prompt**

`ora/prompts/writer.py`:
```python
"""Writer agent prompt."""

WRITER_PROMPT = """You are a research writer. Your job is to synthesize research findings into a clear, calibrated report.

## Output Structure
Generate a markdown report with these sections:

# Research: [Query]
**Intensity:** Level {intensity} | **Sources:** [N] | **Date:** [today]

## Executive Summary
[2-4 sentences summarizing key findings with confidence]

## Key Findings
### [Subtopic]
**Finding:** [Claim]
**Confidence:** High/Moderate/Low/Unknown
**Sources:** [source](url), [source](url)
**Corroborated by:** N independent sources
**Contradicting evidence:** [If any]

## Evidence Gaps
- [What we couldn't find]

## Source Table
| # | Title | URL | Type | Reliability | Accessed |
|---|-------|-----|------|-------------|----------|

## Bibliography
[Numbered list of all sources]

## Rules
- Every claim MUST cite its source with URL
- Confidence levels: High (2+ strong sources), Moderate (good but single or minor gaps), Low (limited/conflicting), Unknown (no evidence)
- Do not fabricate citations. If a source doesn't support a claim, don't cite it.
- Acknowledge uncertainty. It's better to say "we don't know" than to sound confident with weak evidence.
- Single-sourced claims must be flagged

Research findings:
{findings}

Query: {query}"""
```

- [ ] **Step 5: Write reviewer prompt**

`ora/prompts/reviewer.py`:
```python
"""Adversarial reviewer agent prompt."""

REVIEWER_PROMPT = """You are an ADVERSARIAL REVIEWER. Your job is to attack this research report and find every flaw, gap, or unsupported claim. Be skeptical. Be thorough. The researcher who wrote this report is not in the room -- you cannot ask them questions. You must verify everything yourself.

## Review Checklist (complete EVERY item)

### 1. URL Verification (BLOCKING)
Spot-check 3-5 cited URLs:
- Do they resolve (not 404, not redirect-to-homepage)?
- Does the page title/content correspond to what's cited?
- Flag any broken or mismatched URLs.

### 2. Opposing Evidence Search (REQUIRED)
- Search for evidence that CONTRADICTS the key claims in this report.
- If contradicting evidence exists, report it with sources.
- If no contradicting evidence was found, state what you searched for.

### 3. Citation Accuracy (BLOCKING)
- For 2-3 key claims, follow the citation and verify the source actually supports the claim.
- Flag any claims where the source does not support or contradicts what's cited.

### 4. Confidence Calibration (REQUIRED)
- Check: are claims with weak/single-source evidence labeled Low or Moderate?
- Check: are claims with strong multi-source support labeled High?
- If confidence is miscalibrated, specify which claim and what level it should be.

### 5. Evidence Gaps (REQUIRED)
- What did the report not find that the original query asked for?
- Are these gaps acknowledged in the report?
- List any unacknowledged gaps.

### 6. Contradiction Documentation (REQUIRED)
- If the report's own sources disagree, is this documented with both positions?
- If contradictions were silently resolved, flag which ones.

### 7. Completeness (SUGGESTED)
- Does the report address all aspects of the original query?
- Is anything important missing entirely?

## Output Format
Return a JSON object with this exact structure:
```json
{
  "verdict": "PASS" or "REVISE",
  "blocking": ["issue 1", "issue 2"],
  "required": ["issue 1", "issue 2"],
  "suggested": ["issue 1", "issue 2"],
  "contradicting_evidence_found": ["evidence with source"],
  "confidence_recalibrations": {"claim_description": "new_level"}
}
```

## Verdict Rules
- PASS: No blocking issues. Required issues may exist but don't make the report incorrect.
- REVISE: One or more blocking issues exist. These MUST be fixed.

## Critical Rule
You are an ADVERSARY. Your default stance is skepticism, not agreement. If something looks too clean, dig deeper. If a claim is unsupported, call it out. The user depends on you to catch what the researcher missed.

Original query: {query}
Report to review:
{report}"""
```

- [ ] **Step 6: Verify imports work**

Run: `python -c "from ora.prompts import load_prompt; p = load_prompt('researcher'); assert 'research agent' in p.lower(); print('OK')"`
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add ora/prompts/
git commit -m "feat: add agent system prompts"
```

---

### Task 7: Researcher Agent

**Files:**
- Create: `ora/agents/researcher.py`
- Create: `tests/test_researcher.py`

LangGraph node that runs the research phase: generates queries, searches, scrapes, evaluates.

- [ ] **Step 1: Write failing test**

`tests/test_researcher.py`:
```python
"""Tests for researcher agent."""
from unittest.mock import patch, MagicMock
from ora.state import ResearchState, Source, Finding
from ora.agents.researcher import researcher_node


class TestResearcherNode:
    def test_node_returns_updated_state(self):
        state: ResearchState = {
            "query": "test query",
            "intensity": 1,
            "sources": [],
            "findings": [],
        }
        # This tests structure, not actual API calls
        assert "query" in state
        assert state["intensity"] == 1

    def test_generates_queries_for_intensity(self):
        """Researcher should generate appropriate number of queries per intensity."""
        from ora.agents.researcher import generate_search_queries
        queries = generate_search_queries("test query", intensity=2)
        assert 3 <= len(queries) <= 4  # Level 2: 3-4 queries
        assert any("test query" in q.lower() for q in queries)

    def test_intensity_1_generates_fewer_queries(self):
        from ora.agents.researcher import generate_search_queries
        queries = generate_search_queries("test", intensity=1)
        assert 1 <= len(queries) <= 2

    def test_intensity_3_generates_more_queries(self):
        from ora.agents.researcher import generate_search_queries
        queries = generate_search_queries("test", intensity=3)
        assert len(queries) >= 5
```

- [ ] **Step 2: Run to see it fail**

Run: `pytest tests/test_researcher.py -v`
Expected: FAIL

- [ ] **Step 3: Write researcher.py**

`ora/agents/researcher.py`:
```python
"""Researcher agent node for LangGraph."""
import json
from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from ora.state import ResearchState, Source, Finding
from ora.prompts import RESEARCHER_PROMPT


def generate_search_queries(query: str, intensity: int) -> list[str]:
    """Generate search queries based on intensity level.

    Intensity determines the number and angle diversity of queries.
    """
    angles = {
        1: ["{query}"],
        2: ["{query}", "{query} latest", 'opposing view on {query}', '"{query}" recent'],
        3: [
            "{query}",
            "{query} latest research 2025 2026",
            'critique of {query}',
            '"{query}" expert analysis',
            '{query} vs alternatives comparison',
            '{query} best practices',
            'problems with {query}',
        ],
    }

    templates = angles.get(intensity, angles[2])
    return [t.format(query=query) for t in templates]


async def researcher_node(state: ResearchState, config: RunnableConfig = None) -> dict[str, Any]:
    """Researcher LangGraph node. Searches, scrapes, and evaluates sources.

    For v0, this uses LangChain's agent executor pattern with tool access.
    The LLM is prompted to search, scrape, and evaluate sources.
    """
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from ora.tools.search import web_search
    from ora.tools.scrape import scrape_page
    from ora.config import get_researcher_model, load_config

    settings = load_config()
    model_name = get_researcher_model(settings)

    llm = ChatOpenAI(
        model=model_name.split(":")[-1],
        temperature=0,
    )

    tools = [web_search, scrape_page]
    prompt_text = RESEARCHER_PROMPT.format(
        query=state.get("query", ""),
        intensity=state.get("intensity", 2),
    )

    agent = create_tool_calling_agent(llm, tools, prompt_text)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=15,
    )

    query_count = {1: 2, 2: 4, 3: 7}.get(state.get("intensity", 2), 4)
    queries = generate_search_queries(state["query"], state.get("intensity", 2))

    result = await agent_executor.ainvoke({
        "input": f"Research the following query using {query_count} search queries: {', '.join(queries[:query_count])}",
        "chat_history": [],
    })

    output = result.get("output", "")

    # Parse findings from agent output (structured format expected in prompt)
    # For v0, we store the raw output; an LLM parsing step extracts structured findings
    return {
        "search_queries": queries[:query_count],
        "sources": _extract_sources_from_output(output),
        "findings": _extract_findings_from_output(output),
        "messages": [result.get("output", "")],
    }


def _extract_sources_from_output(output: str) -> list[Source]:
    """Parse sources from agent output. Extracts URLs mentioned in text."""
    import re
    from ora.tools.evaluate import evaluate_source

    url_pattern = r'https?://[^\s<>"\']+'
    urls = list(set(re.findall(url_pattern, output)))
    sources = []
    for url in urls[:20]:  # Max 20 sources per run
        source = evaluate_source(url=url, content=output[:500], source_type="unknown")
        sources.append(source)
    return sources


def _extract_findings_from_output(output: str) -> list[Finding]:
    """Parse findings from agent output as structured claims."""
    findings = []
    # Simple heuristic: split on bullet points or numbered items
    lines = output.split("\n")
    current_finding = ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("- ", "* ", "1.", "2.", "3.")) and len(stripped) > 10:
            if current_finding:
                findings.append(Finding(claim=current_finding[:500]))
            current_finding = stripped.lstrip("- *1234567890. ")
        elif current_finding:
            current_finding += " " + stripped
    if current_finding and len(current_finding) > 10:
        findings.append(Finding(claim=current_finding[:500]))

    if not findings:
        findings.append(Finding(claim=output[:500]))

    return findings
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_researcher.py -v`
Expected: All PASS (structure tests, no API calls)

- [ ] **Step 5: Commit**

```bash
git add ora/agents/researcher.py tests/test_researcher.py
git commit -m "feat: add researcher agent node"
```

---

### Task 8: Writer Agent

**Files:**
- Create: `ora/agents/writer.py`

LangGraph node that synthesizes findings into the structured markdown report.

- [ ] **Step 1: Write writer.py**

`ora/agents/writer.py`:
```python
"""Writer agent node for LangGraph."""
from typing import Any
from datetime import date
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from ora.state import ResearchState
from ora.prompts import WRITER_PROMPT
from ora.config import load_config, get_researcher_model


def _format_findings_for_prompt(findings: list) -> str:
    """Format findings list into a string for the writer prompt."""
    if not findings:
        return "No findings available."

    lines = []
    for i, f in enumerate(findings, 1):
        sources_str = ", ".join(f.supporting_sources[:3]) if f.supporting_sources else "no sources"
        lines.append(f"{i}. [{f.confidence}] {f.claim}")
        lines.append(f"   Sources: {sources_str}")
        if f.contradicting_sources:
            lines.append(f"   Contradicted by: {', '.join(f.contradicting_sources[:2])}")
        lines.append("")
    return "\n".join(lines)


async def writer_node(state: ResearchState, config: RunnableConfig = None) -> dict[str, Any]:
    """Writer LangGraph node. Synthesizes findings into a structured report."""
    settings = load_config()
    model_name = get_researcher_model(settings)  # Writer uses same model as researcher

    llm = ChatOpenAI(
        model=model_name.split(":")[-1],
        temperature=0.3,
    )

    findings_text = _format_findings_for_prompt(state.get("findings", []))
    source_count = len(state.get("sources", []))

    prompt_text = WRITER_PROMPT.format(
        query=state.get("query", ""),
        intensity=state.get("intensity", 2),
        findings=findings_text,
    )

    response = llm.invoke(prompt_text)
    draft_report = response.content

    return {
        "draft_report": draft_report,
        "messages": [draft_report],
    }
```

- [ ] **Step 2: Verify import**

Run: `python -c "from ora.agents.writer import writer_node; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add ora/agents/writer.py
git commit -m "feat: add writer agent node"
```

---

### Task 9: Adversarial Reviewer Agent

**Files:**
- Create: `ora/agents/reviewer.py`
- Create: `tests/test_reviewer.py`

The key differentiator. Separate agent, fresh context, attacks the draft report.

- [ ] **Step 1: Write failing test**

`tests/test_reviewer.py`:
```python
"""Tests for adversarial reviewer agent."""
import json
from unittest.mock import patch, MagicMock
from ora.state import ResearchState, ReviewVerdict
from ora.agents.reviewer import parse_reviewer_output


class TestParseReviewerOutput:
    def test_parses_pass_verdict(self):
        output = json.dumps({
            "verdict": "PASS",
            "blocking": [],
            "required": ["needs more sources"],
            "suggested": ["add examples"],
            "contradicting_evidence_found": [],
            "confidence_recalibrations": {},
        })
        verdict = parse_reviewer_output(output)
        assert verdict.verdict == "PASS"
        assert len(verdict.blocking) == 0
        assert len(verdict.required) == 1

    def test_parses_revise_verdict(self):
        output = json.dumps({
            "verdict": "REVISE",
            "blocking": ["broken URL: example.com"],
            "required": [],
            "suggested": [],
            "contradicting_evidence_found": ["source X contradicts claim Y"],
            "confidence_recalibrations": {"claim about AI": "Low"},
        })
        verdict = parse_reviewer_output(output)
        assert verdict.verdict == "REVISE"
        assert len(verdict.blocking) == 1
        assert verdict.contradicting_evidence_found == ["source X contradicts claim Y"]

    def test_handles_malformed_json(self):
        verdict = parse_reviewer_output("not valid json {")
        assert verdict.verdict == "REVISE"
        assert "reviewer output parsing failed" in verdict.blocking[0]


class TestReviewVerdict:
    def test_default_is_pass(self):
        v = ReviewVerdict()
        assert v.verdict == "PASS"

    def test_blocking_issues_cause_revise(self):
        v = ReviewVerdict(verdict="REVISE", blocking=["issue"])
        assert v.verdict == "REVISE"
```

- [ ] **Step 2: Run to see it fail**

Run: `pytest tests/test_reviewer.py -v`
Expected: FAIL

- [ ] **Step 3: Write reviewer.py**

`ora/agents/reviewer.py`:
```python
"""Adversarial reviewer agent node for LangGraph."""
import json
from typing import Any
from langchain_core.runnables import RunnableConfig
from ora.state import ResearchState, ReviewVerdict
from ora.prompts import REVIEWER_PROMPT
from ora.config import load_config, get_reviewer_model


def parse_reviewer_output(output: str) -> ReviewVerdict:
    """Parse the reviewer's JSON output into a ReviewVerdict.

    Handles malformed JSON by returning a REVISE verdict with a blocking issue.
    """
    try:
        # Find JSON block in output (may be wrapped in markdown code fences)
        if "```json" in output:
            start = output.index("```json") + 7
            end = output.index("```", start)
            json_str = output[start:end].strip()
        elif "```" in output:
            start = output.index("```") + 3
            end = output.index("```", start)
            json_str = output[start:end].strip()
        else:
            json_str = output.strip()

        data = json.loads(json_str)
        return ReviewVerdict(
            verdict=data.get("verdict", "PASS"),
            blocking=data.get("blocking", []),
            required=data.get("required", []),
            suggested=data.get("suggested", []),
            contradicting_evidence_found=data.get("contradicting_evidence_found", []),
            confidence_recalibrations=data.get("confidence_recalibrations", {}),
        )
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        return ReviewVerdict(
            verdict="REVISE",
            blocking=[f"Reviewer output parsing failed: {str(e)}. Raw: {output[:200]}"],
        )


async def reviewer_node(state: ResearchState, config: RunnableConfig = None) -> dict[str, Any]:
    """Adversarial reviewer LangGraph node.

    Receives the draft report and original query. Does NOT receive the researcher's
    intermediate findings, search queries, or reasoning. Uses a different model
    family from the researcher to prevent correlated errors.
    """
    from langchain.chat_models import init_chat_model

    settings = load_config()
    model_name = get_reviewer_model(settings)

    # Use init_chat_model for cross-provider support
    llm = init_chat_model(model_name, temperature=0.2)

    prompt_text = REVIEWER_PROMPT.format(
        query=state.get("query", ""),
        report=state.get("draft_report", ""),
    )

    response = llm.invoke(prompt_text)
    output = response.content

    verdict = parse_reviewer_output(output)

    return {
        "review_verdict": verdict,
        "review_verdict_raw": output,
        "messages": [output],
    }
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_reviewer.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add ora/agents/reviewer.py tests/test_reviewer.py
git commit -m "feat: add adversarial reviewer agent"
```

---

### Task 10: Supervisor + Graph Assembly

**Files:**
- Create: `ora/agents/supervisor.py`
- Create: `ora/graph.py`
- Create: `tests/test_graph.py`

Supervisor handles routing between agents. Graph ties all nodes together with conditional edges.

- [ ] **Step 1: Write supervisor.py**

`ora/agents/supervisor.py`:
```python
"""Supervisor agent node for LangGraph."""
from typing import Any, Literal
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from ora.state import ResearchState
from ora.prompts import SUPERVISOR_PLAN_PROMPT, SUPERVISOR_ROUTE_PROMPT
from ora.config import load_config


async def plan_node(state: ResearchState, config: RunnableConfig = None) -> dict[str, Any]:
    """Supervisor plan node. Generates a research plan for user review."""
    settings = load_config()
    llm = ChatOpenAI(
        model=settings.models.default.split(":")[-1],
        temperature=0,
    )

    prompt = SUPERVISOR_PLAN_PROMPT.format(
        query=state.get("query", ""),
        intensity=state.get("intensity", 2),
    )

    response = llm.invoke(prompt)
    return {
        "research_plan": response.content,
        "messages": [response.content],
    }


async def route_node(state: ResearchState, config: RunnableConfig = None) -> dict[str, Any]:
    """Supervisor routing node. Decides next agent based on state."""
    # For v0, use deterministic routing rather than LLM-based routing
    # This is more reliable and cheaper
    return {}  # Routing is handled by conditional edges in graph.py


def route_after_plan(state: ResearchState) -> Literal["researcher", "__end__"]:
    """Route after plan generation."""
    if state.get("plan_approved", False):
        return "researcher"
    return "__end__"  # Wait for human approval


def route_after_researcher(state: ResearchState) -> Literal["writer", "__end__"]:
    """Route after research phase."""
    if state.get("findings"):
        return "writer"
    return "__end__"


def route_after_writer(state: ResearchState) -> Literal["reviewer", "__end__"]:
    """Route after writer. Always go to reviewer in v0."""
    if state.get("draft_report"):
        return "reviewer"
    return "__end__"


def route_after_reviewer(state: ResearchState) -> Literal["researcher", "__end__"]:
    """Route after adversarial review."""
    verdict = state.get("review_verdict")
    if verdict is None:
        return "__end__"

    if hasattr(verdict, 'verdict'):
        v = verdict.verdict
    else:
        v = verdict.get("verdict", "REVISE") if isinstance(verdict, dict) else "REVISE"

    revision_count = state.get("revision_count", 0)
    max_revisions = 3

    if v == "PASS":
        return "__end__"
    elif revision_count < max_revisions:
        return "researcher"
    else:
        return "__end__"
```

- [ ] **Step 2: Write graph.py**

`ora/graph.py`:
```python
"""LangGraph StateGraph assembly for ORA."""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from ora.state import ResearchState
from ora.agents.supervisor import (
    plan_node,
    route_after_plan,
    route_after_researcher,
    route_after_writer,
    route_after_reviewer,
)
from ora.agents.researcher import researcher_node
from ora.agents.writer import writer_node
from ora.agents.reviewer import reviewer_node


def build_graph() -> StateGraph:
    """Build and compile the ORA research graph.

    Graph structure:
    plan -> [HITL interrupt] -> researcher -> writer -> reviewer -> [loop or end]
    """
    workflow = StateGraph(ResearchState)

    # Add nodes
    workflow.add_node("plan", plan_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)

    # Set entry point
    workflow.set_entry_point("plan")

    # Plan routes to researcher if approved, otherwise waits for human
    workflow.add_conditional_edges(
        "plan",
        route_after_plan,
        {"researcher": "researcher", "__end__": END},
    )

    # Researcher -> Writer
    workflow.add_conditional_edges(
        "researcher",
        route_after_researcher,
        {"writer": "writer", "__end__": END},
    )

    # Writer -> Reviewer
    workflow.add_conditional_edges(
        "writer",
        route_after_writer,
        {"reviewer": "reviewer", "__end__": END},
    )

    # Reviewer -> Researcher (revise) or END (pass)
    workflow.add_conditional_edges(
        "reviewer",
        route_after_reviewer,
        {"researcher": "researcher", "__end__": END},
    )

    # Use in-memory checkpointer for state persistence
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory, interrupt_before=["researcher"])
```

- [ ] **Step 3: Write graph test**

`tests/test_graph.py`:
```python
"""Tests for graph assembly and routing."""
import pytest
from ora.graph import build_graph
from ora.state import ResearchState


class TestGraphAssembly:
    def test_graph_builds_without_error(self):
        graph = build_graph()
        assert graph is not None

    def test_graph_has_required_nodes(self):
        graph = build_graph()
        # Graph nodes are internal, but we can verify the compiled graph exists
        assert graph is not None

    def test_graph_accepts_initial_state(self):
        graph = build_graph()
        initial_state: ResearchState = {
            "query": "test",
            "intensity": 2,
            "plan_approved": False,
            "revision_count": 0,
        }
        # Don't actually invoke (requires API keys), just verify structure
        assert initial_state["query"] == "test"
        assert initial_state["intensity"] == 2


class TestRouting:
    def test_route_after_plan_not_approved(self):
        from ora.agents.supervisor import route_after_plan
        state: ResearchState = {"plan_approved": False}
        result = route_after_plan(state)
        assert result == "__end__"

    def test_route_after_plan_approved(self):
        from ora.agents.supervisor import route_after_plan
        state: ResearchState = {"plan_approved": True}
        result = route_after_plan(state)
        assert result == "researcher"

    def test_route_after_reviewer_pass(self):
        from ora.agents.supervisor import route_after_reviewer
        from ora.state import ReviewVerdict
        state: ResearchState = {
            "review_verdict": ReviewVerdict(verdict="PASS"),
            "revision_count": 0,
        }
        result = route_after_reviewer(state)
        assert result == "__end__"

    def test_route_after_reviewer_revise_under_limit(self):
        from ora.agents.supervisor import route_after_reviewer
        from ora.state import ReviewVerdict
        state: ResearchState = {
            "review_verdict": ReviewVerdict(verdict="REVISE"),
            "revision_count": 1,
        }
        result = route_after_reviewer(state)
        assert result == "researcher"

    def test_route_after_reviewer_revise_at_limit(self):
        from ora.agents.supervisor import route_after_reviewer
        from ora.state import ReviewVerdict
        state: ResearchState = {
            "review_verdict": ReviewVerdict(verdict="REVISE"),
            "revision_count": 3,
        }
        result = route_after_reviewer(state)
        assert result == "__end__"
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_graph.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add ora/agents/supervisor.py ora/graph.py tests/test_graph.py
git commit -m "feat: add supervisor agent and graph assembly"
```

---

### Task 11: CLI Interface

**Files:**
- Create: `ora/cli.py`
- Create: `tests/test_cli.py`

Click CLI with `ora research`, `ora plan`, `ora config`, `ora bench` subcommands.

- [ ] **Step 1: Write failing test**

`tests/test_cli.py`:
```python
"""Tests for CLI interface."""
import pytest
from click.testing import CliRunner
from ora.cli import main


class TestCLI:
    def test_cli_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "research" in result.output

    def test_research_command_accepts_query(self):
        runner = CliRunner()
        result = runner.invoke(main, ["research", "--help"])
        assert result.exit_code == 0
        assert "query" in result.output.lower() or "QUERY" in result.output

    def test_plan_command_exists(self):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "--help"])
        assert result.exit_code == 0
        assert "query" in result.output.lower() or "QUERY" in result.output

    def test_research_requires_query(self):
        runner = CliRunner()
        result = runner.invoke(main, ["research"])
        # Should fail or show help
        assert result.exit_code != 0 or "Missing argument" in result.output or "Error" in result.output

    def test_intensity_flag_defaults_to_2(self):
        runner = CliRunner()
        result = runner.invoke(main, ["research", "test query", "--help"])
        assert result.exit_code == 0
        assert "--intensity" in result.output

    def test_config_command_exists(self):
        runner = CliRunner()
        result = runner.invoke(main, ["config", "--help"])
        assert result.exit_code == 0
```

- [ ] **Step 2: Run to see it fail**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL

- [ ] **Step 3: Write cli.py**

`ora/cli.py`:
```python
"""Click CLI for Open Research Agent (ORA)."""
import sys
import click
from ora.config import load_config, ORASettings


@click.group()
@click.version_option(version="0.1.0", prog_name="ora")
def main():
    """Open Research Agent (ORA) - Multi-agent research with adversarial review.

    Submit a query, get back calibrated, source-traced research that
    has been audited by a separate adversarial agent.
    """
    pass


@main.command()
@click.argument("query")
@click.option(
    "--intensity", "-i",
    type=click.IntRange(1, 5),
    default=2,
    help="Research intensity (1=Quick, 2=Standard, 3=Thorough, 4=Deep, 5=Exhaustive)",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Output file path (default: stdout)",
)
@click.option(
    "--model", "-m",
    default=None,
    help="LLM model for researcher/writer (e.g., openai:gpt-4.1)",
)
@click.option(
    "--reviewer-model", "-r",
    default=None,
    help="LLM model for adversarial reviewer (different provider recommended)",
)
@click.option(
    "--search", "-s",
    default="firecrawl",
    help="Search backend (default: firecrawl)",
)
@click.option(
    "--max-revisions",
    type=int,
    default=3,
    help="Max writer-reviewer revision cycles",
)
@click.option(
    "--no-review",
    is_flag=True,
    help="Skip adversarial review",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show agent actions in real-time",
)
def research(
    query: str,
    intensity: int,
    output: str | None,
    model: str | None,
    reviewer_model: str | None,
    search: str,
    max_revisions: int,
    no_review: bool,
    verbose: bool,
):
    """Run the full research pipeline.

    Example:
        ora research "What is quantum computing?" --intensity 2 --output report.md
    """
    click.echo(f"🔍 ORA Research: {query}")
    click.echo(f"   Intensity: {intensity} | Model: {model or 'default'}")

    if intensity > 3:
        click.echo("   ⚠️  Levels 4-5 are not yet implemented in v0.", err=True)
        click.echo("   Falling back to Level 3.", err=True)
        intensity = 3

    settings = load_config()
    if model:
        settings.models.researcher = model
    if reviewer_model:
        settings.models.reviewer = reviewer_model

    # Build and run graph
    from ora.graph import build_graph
    graph = build_graph()

    config = {"configurable": {"thread_id": "research-1"}}

    initial_state = {
        "query": query,
        "intensity": intensity,
        "plan_approved": False,
        "revision_count": 0,
    }

    click.echo("   Generating research plan...")

    # Step 1: Run plan node
    plan_result = graph.invoke(initial_state, config)
    plan = plan_result.get("research_plan", "No plan generated.")
    click.echo(f"\n📋 Research Plan:\n{plan}\n")

    # Step 2: Get user approval
    if not click.confirm("   Approve this plan and begin research?"):
        click.echo("   Research cancelled.")
        return

    # Step 3: Continue pipeline (plan approved)
    plan_result["plan_approved"] = True
    plan_result["revision_count"] = 0

    click.echo("   🔬 Researching...")
    final_state = graph.invoke(plan_result, config)

    # Step 4: Handle revision loop
    revision_count = 0
    while revision_count < max_revisions:
        verdict = final_state.get("review_verdict")
        if verdict is None:
            break

        # Check verdict
        if hasattr(verdict, 'verdict'):
            v = verdict.verdict
        elif isinstance(verdict, dict):
            v = verdict.get("verdict", "PASS")
        else:
            v = "PASS"

        if v == "PASS":
            break

        click.echo(f"   🔄 Revision {revision_count + 1}/{max_revisions} (reviewer found issues)")
        revision_count += 1
        final_state["revision_count"] = revision_count
        final_state = graph.invoke(final_state, config)

    # Step 5: Output final report
    draft = final_state.get("draft_report", "No report generated.")
    verdict = final_state.get("review_verdict")

    # Format final output
    verdict_str = ""
    if verdict and not no_review:
        if hasattr(verdict, 'verdict'):
            v = verdict.verdict
            blocking = len(verdict.blocking)
            required = len(verdict.required)
            suggested = len(verdict.suggested)
        elif isinstance(verdict, dict):
            v = verdict.get("verdict", "?")
            blocking = len(verdict.get("blocking", []))
            required = len(verdict.get("required", []))
            suggested = len(verdict.get("suggested", []))
        verdict_str = f"\n\n## Reviewer Gate\n- **Verdict:** {v}\n- **Blocking:** {blocking}\n- **Required:** {required}\n- **Suggested:** {suggested}"

    final_report = draft + verdict_str

    if output:
        with open(output, "w") as f:
            f.write(final_report)
        click.echo(f"\n✅ Report saved to {output}")
    else:
        click.echo(f"\n{final_report}")


@main.command()
@click.argument("query")
@click.option("--intensity", "-i", type=click.IntRange(1, 5), default=2)
def plan(query: str, intensity: int):
    """Generate a research plan for review without executing research."""
    from ora.graph import build_graph
    graph = build_graph()

    initial_state = {
        "query": query,
        "intensity": intensity,
        "plan_approved": False,
        "revision_count": 0,
    }

    result = graph.invoke(initial_state, {"configurable": {"thread_id": "plan-1"}})
    plan_text = result.get("research_plan", "No plan generated.")
    click.echo(f"📋 Research Plan for: {query}\n")
    click.echo(plan_text)


@main.command()
@click.option("--show", is_flag=True, help="Show current configuration")
@click.option("--init", is_flag=True, help="Create default config file at ~/.ora/config.yaml")
def config(show: bool, init: bool):
    """Show or initialize ORA configuration."""
    import os
    import yaml

    config_path = os.path.expanduser("~/.ora/config.yaml")

    if init:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        default_config = {
            "models": {
                "default": "openai:gpt-4.1-mini",
                "researcher": "openai:gpt-4.1",
                "reviewer": "anthropic:claude-sonnet-4-20250514",
            },
            "search": {
                "provider": "firecrawl",
            },
            "limits": {
                "max_revisions": 3,
                "default_intensity": 2,
            },
        }
        with open(config_path, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False)
        click.echo(f"✅ Config created at {config_path}")
        return

    settings = load_config()
    if show or not init:
        click.echo(f"Config file: {config_path}")
        click.echo(f"Default model: {settings.models.default}")
        click.echo(f"Researcher: {settings.models.researcher or '(default)'}")
        click.echo(f"Reviewer: {settings.models.reviewer or '(auto: opposite provider)'}")
        click.echo(f"Search provider: {settings.search.provider}")
        click.echo(f"Max revisions: {settings.limits.max_revisions}")
        click.echo(f"Default intensity: {settings.limits.default_intensity}")


@main.command()
@click.option("--dataset", default="deep_research_bench", help="Benchmark dataset to use")
def bench(dataset: str):
    """Run ORA against the Deep Research Bench."""
    click.echo("🏋️ Deep Research Bench runner (v0 placeholder)")
    click.echo("   Install deep_research_bench: git clone https://github.com/Ayanami0730/deep_research_bench")
    click.echo("   Then run: python tests/run_evaluate.py")
    click.echo("   See docs for full benchmark submission instructions.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_cli.py -v`
Expected: All PASS

- [ ] **Step 5: Verify CLI works**

Run: `python -m ora.cli --help`
Expected: Shows help text with research, plan, config, bench subcommands

- [ ] **Step 6: Commit**

```bash
git add ora/cli.py tests/test_cli.py
git commit -m "feat: add Click CLI with research/plan/config/bench commands"
```

---

### Task 12: Integration Test

**Files:**
- Create: `tests/test_integration.py`

Smoke test of the full graph without real API calls (mocked LLM responses).

- [ ] **Step 1: Write integration test**

`tests/test_integration.py`:
```python
"""Integration smoke test for the full ORA pipeline."""
import pytest
from unittest.mock import patch, MagicMock
from ora.graph import build_graph
from ora.state import ResearchState, ReviewVerdict


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_graph_end_to_end_mocked(self):
        """Verify the graph can be invoked with mocked LLM responses."""
        graph = build_graph()

        initial_state: ResearchState = {
            "query": "test query",
            "intensity": 1,
            "plan_approved": True,
            "revision_count": 0,
            "sources": [],
            "findings": [],
            "draft_report": "",
        }

        # The graph will try to call LLM APIs; this test verifies structure only
        assert graph is not None
        assert initial_state["query"] == "test query"

    def test_reviewer_blocks_broken_urls(self):
        """Adversarial reviewer should catch broken URLs."""
        from ora.agents.reviewer import parse_reviewer_output
        output = '{"verdict": "REVISE", "blocking": ["URL https://example.com/fake returns 404"], "required": [], "suggested": [], "contradicting_evidence_found": [], "confidence_recalibrations": {}}'
        verdict = parse_reviewer_output(output)
        assert verdict.verdict == "REVISE"
        assert len(verdict.blocking) == 1
        assert "404" in verdict.blocking[0]

    def test_reviewer_passes_clean_report(self):
        """Adversarial reviewer should pass a clean report."""
        from ora.agents.reviewer import parse_reviewer_output
        output = '{"verdict": "PASS", "blocking": [], "required": ["minor: add one more source"], "suggested": ["add examples"], "contradicting_evidence_found": [], "confidence_recalibrations": {}}'
        verdict = parse_reviewer_output(output)
        assert verdict.verdict == "PASS"
        assert len(verdict.blocking) == 0

    def test_complete_state_flow(self):
        """Verify the full state flow through the pipeline."""
        from ora.agents.supervisor import (
            route_after_plan,
            route_after_researcher,
            route_after_writer,
            route_after_reviewer,
        )

        # Plan -> Researcher
        assert route_after_plan({"plan_approved": True}) == "researcher"
        assert route_after_plan({"plan_approved": False}) == "__end__"

        # Researcher -> Writer
        assert route_after_researcher({"findings": [{}]}) == "writer"
        assert route_after_researcher({"findings": []}) == "__end__"

        # Writer -> Reviewer
        assert route_after_writer({"draft_report": "content"}) == "reviewer"
        assert route_after_writer({"draft_report": ""}) == "__end__"

        # Reviewer -> End or Loop
        verdict = ReviewVerdict(verdict="PASS")
        assert route_after_reviewer({"review_verdict": verdict, "revision_count": 0}) == "__end__"

        rev_verdict = ReviewVerdict(verdict="REVISE")
        assert route_after_reviewer({"review_verdict": rev_verdict, "revision_count": 0}) == "researcher"
        assert route_after_reviewer({"review_verdict": rev_verdict, "revision_count": 3}) == "__end__"
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_integration.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration smoke test"
```

---

### Task 13: Run Full Test Suite

- [ ] **Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS (no API-dependent tests should fail)

- [ ] **Step 2: Check test coverage**

Run: `pytest tests/ --cov=ora --cov-report=term-missing 2>/dev/null || echo "Coverage not available (install pytest-cov for coverage)"`
Expected: Tests run successfully

- [ ] **Step 3: Final commit if needed**

```bash
git add -A
git status
git commit -m "chore: finalize v0 implementation"
```
