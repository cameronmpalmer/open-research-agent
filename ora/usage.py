"""LLM usage and cost collection for ORA runs.

Every agent LLM call goes through ``get_llm``; when a usage collector is
active (via :func:`usage_collection`), ``get_llm`` wraps the returned
model's ``invoke`` and records token usage and provider-reported cost from
each response.

Token counts come from the message ``usage_metadata``. Cached input tokens
are read from ``input_token_details.cache_read`` (OpenRouter maps its
``prompt_tokens_details.cached_tokens`` there), with fallbacks to the raw
``token_usage`` payload for providers that use other field names (e.g.
DeepSeek's ``prompt_cache_hit_tokens``). Cost is taken from the raw usage
payload's ``cost`` field when the provider reports it (OpenRouter does;
DeepSeek does not), otherwise the run is marked as cost-unknown.
"""

import threading
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Optional

_current_collector: ContextVar[Optional["UsageCollector"]] = ContextVar(
    "ora_usage_collector", default=None
)


@dataclass
class UsageCollector:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    calls: int = 0
    cost: float = 0.0
    cost_known: bool = True
    _cache_extra_keys: tuple = (
        "cached_tokens",
        "prompt_cache_hit_tokens",
        "cache_read",
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def record(self, response: Any) -> None:
        """Record usage from an LLM response (AIMessage or similar)."""
        usage_metadata = getattr(response, "usage_metadata", None) or {}
        input_tokens = int(usage_metadata.get("input_tokens") or 0)
        output_tokens = int(usage_metadata.get("output_tokens") or 0)
        if input_tokens == 0 and output_tokens == 0:
            return

        # Raw provider payload: cached-token fallbacks and reported cost both
        # read from it, so compute it once.
        raw = self._raw_usage(response)

        # Cached input tokens: usage_metadata detail, then raw payload keys.
        details = usage_metadata.get("input_token_details") or {}
        cached = int(details.get("cache_read") or 0)
        if cached == 0:
            for key in self._cache_extra_keys:
                value = raw.get(key)
                if value is not None:
                    cached = int(value)
                    break

        # Provider-reported cost (USD), e.g. OpenRouter's usage.cost.
        cost = raw.get("cost")

        # Worker threads record into a shared collector, so the counter
        # updates must be atomic to avoid lost updates.
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.calls += 1
            self.cached_tokens += cached
            if cost is None:
                self.cost_known = False
            else:
                self.cost += float(cost)

    @staticmethod
    def _raw_usage(response: Any) -> dict:
        metadata = getattr(response, "response_metadata", None) or {}
        raw = metadata.get("token_usage")
        return raw if isinstance(raw, dict) else {}

    def summary_lines(self, elapsed: float) -> list[str]:
        """Human-readable summary lines. Empty when no calls were recorded."""
        if self.calls == 0:
            return []
        lines = [
            (
                f"Usage: {self.input_tokens} in | {self.output_tokens} out"
                f" | {self.cached_tokens} cached-in | {self.calls} calls"
            )
        ]
        if self.cost_known:
            lines.append(f"Cost: {format_cost(self.cost)}")
        else:
            lines.append("Cost: n/a (provider does not report cost)")
        lines.append(f"Time: {elapsed:.1f}s")
        return lines


def format_cost(cost: float) -> str:
    """Format a USD cost compactly, e.g. 3.11e-06 -> '$0.00000311'."""
    return f"${cost:.8f}".rstrip("0").rstrip(".")


class usage_collection:
    """Context manager binding a UsageCollector for the current context.

    LLM responses recorded while active are attributed to ``collector``
    (a new one when not provided).
    """

    def __init__(self, collector: UsageCollector | None = None):
        self._collector = collector or UsageCollector()
        self._token: object | None = None

    def __enter__(self) -> UsageCollector:
        self._token = _current_collector.set(self._collector)
        return self._collector

    def __exit__(self, *exc: object) -> None:
        if self._token is not None:
            _current_collector.reset(self._token)


def active_collector() -> UsageCollector | None:
    """The collector bound to the current context, if any."""
    return _current_collector.get()


class RecordingLLM:
    """Delegating wrapper that records usage from each ``invoke`` response.

    ChatOpenAI is a pydantic model and rejects instance-attribute patching,
    so get_llm returns this proxy instead when a collector is active. All
    other attributes delegate to the wrapped model.
    """

    def __init__(self, llm: Any, collector: UsageCollector):
        self._llm = llm
        self._collector = collector

    def invoke(self, *args, **kwargs):
        response = self._llm.invoke(*args, **kwargs)
        self._collector.record(response)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)
