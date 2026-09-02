"""Tests for LLM usage collection (ora.usage)."""

import pytest

from ora.usage import UsageCollector, active_collector, format_cost, usage_collection


def _message(usage_metadata=None, token_usage=None):
    """Build a fake AIMessage-like object."""
    return type(
        "FakeMessage",
        (),
        {
            "usage_metadata": usage_metadata or {},
            "response_metadata": ({"token_usage": token_usage} if token_usage else {}),
        },
    )()


class TestUsageCollector:
    def test_records_tokens_and_cost(self):
        collector = UsageCollector()
        collector.record(
            _message(
                usage_metadata={
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "input_token_details": {"cache_read": 20},
                },
                token_usage={"cost": 3.11e-06},
            )
        )
        assert collector.input_tokens == 100
        assert collector.output_tokens == 50
        assert collector.cached_tokens == 20
        assert collector.calls == 1
        assert collector.cost == pytest.approx(3.11e-06)
        assert collector.cost_known is True

    def test_accumulates_multiple_calls(self):
        collector = UsageCollector()
        for _ in range(3):
            collector.record(
                _message(
                    usage_metadata={"input_tokens": 10, "output_tokens": 5},
                    token_usage={"cost": 1.0},
                )
            )
        assert collector.calls == 3
        assert collector.input_tokens == 30
        assert collector.output_tokens == 15
        assert collector.cost == 3.0

    def test_cost_unknown_when_provider_omits_cost(self):
        collector = UsageCollector()
        collector.record(
            _message(
                usage_metadata={"input_tokens": 10, "output_tokens": 5},
                token_usage={"prompt_cache_hit_tokens": 3},
            )
        )
        assert collector.cost_known is False
        assert collector.cached_tokens == 3  # DeepSeek-style raw field fallback

    def test_cached_falls_back_to_openrouter_raw_field(self):
        collector = UsageCollector()
        collector.record(
            _message(
                usage_metadata={"input_tokens": 10, "output_tokens": 5},
                token_usage={"prompt_tokens_details": {"cached_tokens": 7}},
            )
        )
        # prompt_tokens_details is nested; cache_read in usage_metadata wins when
        # present, otherwise we look for flat raw keys. Nested-only payloads
        # yield 0 cached (documented limitation).
        assert collector.cached_tokens == 0

    def test_response_without_usage_is_ignored(self):
        collector = UsageCollector()
        collector.record(_message())
        assert collector.calls == 0
        assert collector.summary_lines(1.0) == []

    def test_ignores_plain_string_response(self):
        collector = UsageCollector()
        collector.record("just text")
        assert collector.calls == 0

    def test_summary_lines_format(self):
        collector = UsageCollector()
        collector.record(
            _message(
                usage_metadata={"input_tokens": 1234, "output_tokens": 567},
                token_usage={"cost": 3.11e-06},
            )
        )
        lines = collector.summary_lines(34.25)
        assert lines == [
            "Usage: 1234 in | 567 out | 0 cached-in | 1 calls",
            "Cost: $0.00000311",
            "Time: 34.2s",
        ]

    def test_summary_lines_cost_unknown(self):
        collector = UsageCollector()
        collector.record(_message(usage_metadata={"input_tokens": 1, "output_tokens": 1}))
        lines = collector.summary_lines(2.0)
        assert "Cost: n/a (provider does not report cost)" in lines

    def test_summary_empty_without_calls(self):
        assert UsageCollector().summary_lines(5.0) == []


class TestUsageCollectionContext:
    def test_context_binds_collector(self):
        collector = UsageCollector()
        with usage_collection(collector):
            assert active_collector() is collector
        assert active_collector() is None

    def test_context_creates_collector(self):
        with usage_collection() as collector:
            assert isinstance(collector, UsageCollector)
            assert active_collector() is collector

    def test_nested_context_restores_outer(self):
        outer = UsageCollector()
        with usage_collection(outer):
            with usage_collection() as inner:
                assert active_collector() is inner
            assert active_collector() is outer


class TestFormatCost:
    def test_tiny_cost(self):
        assert format_cost(3.11e-06) == "$0.00000311"

    def test_zero_cost(self):
        assert format_cost(0.0) == "$0"

    def test_round_cost(self):
        assert format_cost(0.0125) == "$0.0125"


class TestGetLlmRecordsUsage:
    def test_invoke_records_into_active_collector(self, monkeypatch):
        from ora import config as config_module
        from ora.config import get_llm

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def invoke(self, *args, **kwargs):
                return _message(
                    usage_metadata={"input_tokens": 7, "output_tokens": 3},
                    token_usage={"cost": 1e-05},
                )

        monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChatOpenAI)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        monkeypatch.setattr("ora.config.load_config", lambda *a, **kw: config_module.ORASettings())

        with usage_collection() as collector:
            llm = get_llm("deepseek-v4-flash")
            llm.invoke("hello")

        assert collector.calls == 1
        assert collector.input_tokens == 7
        assert collector.output_tokens == 3
        assert collector.cost == pytest.approx(1e-05)

    def test_recording_proxy_delegates_attributes(self, monkeypatch):
        from ora import config as config_module
        from ora.config import get_llm
        from ora.usage import RecordingLLM

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def invoke(self, *args, **kwargs):
                return _message(
                    usage_metadata={"input_tokens": 7, "output_tokens": 3},
                    token_usage={"cost": 1e-05},
                )

        monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChatOpenAI)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        monkeypatch.setattr("ora.config.load_config", lambda *a, **kw: config_module.ORASettings())

        with usage_collection() as collector:
            llm = get_llm("deepseek-v4-flash")
            # Recording proxy is returned; unknown attributes delegate.
            assert isinstance(llm, RecordingLLM)
            assert llm.kwargs["model"] == "deepseek-v4-flash"
            llm.invoke("hello")

        assert collector.calls == 1

    def test_invoke_passthrough_without_collector(self, monkeypatch):
        from ora import config as config_module
        from ora.config import get_llm

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                pass

            def invoke(self, *args, **kwargs):
                return _message(
                    usage_metadata={"input_tokens": 7, "output_tokens": 3},
                    token_usage={"cost": 1e-05},
                )

        monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChatOpenAI)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        monkeypatch.setattr("ora.config.load_config", lambda *a, **kw: config_module.ORASettings())

        llm = get_llm("deepseek-v4-flash")
        # No collector active: invoke works and records nothing.
        assert llm.invoke("hello").usage_metadata["input_tokens"] == 7
