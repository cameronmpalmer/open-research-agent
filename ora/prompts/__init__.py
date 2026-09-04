"""Prompt templates for ORA agents."""

from ora.prompts.extractor import EXTRACTOR_PROMPT
from ora.prompts.researcher import GAP_QUERY_PROMPT, ITEM_GAP_QUERY_PROMPT, RESEARCHER_PROMPT
from ora.prompts.reviewer import REVIEWER_PROMPT
from ora.prompts.supervisor import (
    SUPERVISOR_PLAN_PROMPT,
    SUPERVISOR_REVISE_PROMPT,
    SUPERVISOR_ROUTE_PROMPT,
    _search_query_count,
)
from ora.prompts.writer import REVISION_PROMPT, WRITER_PROMPT

__all__ = [
    "EXTRACTOR_PROMPT",
    "GAP_QUERY_PROMPT",
    "ITEM_GAP_QUERY_PROMPT",
    "RESEARCHER_PROMPT",
    "REVIEWER_PROMPT",
    "REVISION_PROMPT",
    "SUPERVISOR_PLAN_PROMPT",
    "SUPERVISOR_REVISE_PROMPT",
    "SUPERVISOR_ROUTE_PROMPT",
    "WRITER_PROMPT",
    "_search_query_count",
]
