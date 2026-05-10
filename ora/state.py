"""Core state types for the ORA research graph."""
from typing import TypedDict, Literal, Optional, Annotated, Any
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


def _list_reducer(left: list | None, right: list | None) -> list:
    """Merge two lists, handling None defaults from total=False TypedDict."""
    if left is None and right is None:
        return []
    if left is None:
        return right
    if right is None:
        return left
    return left + right


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
    intensity: Literal[1, 2, 3, 4, 5]

    # Plan
    research_plan: str
    plan_approved: bool

    # Messages (for LangGraph agent nodes)
    messages: Annotated[list, add_messages]

    # Research
    search_queries: list[str]
    sources: Annotated[list[Source], _list_reducer]
    findings: Annotated[list[Finding], _list_reducer]

    # Report
    draft_report: str
    revision_count: int

    # Review
    review_verdict: ReviewVerdict
    review_verdict_raw: str  # JSON string for structured output parsing

    # Output
    final_report: str
