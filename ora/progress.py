"""Progress event helpers for ORA agent nodes.

Agents call these helpers through LangGraph RunnableConfig. The CLI decides
whether and how to render events.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, TypedDict


ProgressKind = Literal["info", "search", "scrape", "success", "error", "write"]

logger = logging.getLogger(__name__)


class ProgressEvent(TypedDict):
    """A live progress event emitted by an ORA agent."""

    message: str
    kind: ProgressKind


def emit_progress(config: dict[str, Any] | None, message: str, kind: ProgressKind = "info") -> None:
    """Emit a progress event if the caller supplied a callback.

    Progress rendering is best-effort. A broken display callback must not break
    research execution.
    """
    configurable = (config or {}).get("configurable") or {}
    callback = configurable.get("progress_callback")
    if not callback or not callable(callback):
        return

    try:
        callback({"message": message, "kind": kind})
    except Exception:
        logger.debug("progress callback failed", exc_info=True)
        return
