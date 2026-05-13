"""Tests for progress event helper."""
from ora.progress import emit_progress


def test_emit_progress_no_config_is_noop():
    emit_progress(None, "hello", kind="info")


def test_emit_progress_without_callback_is_noop():
    emit_progress({"configurable": {}}, "hello", kind="info")


def test_emit_progress_with_none_configurable_is_noop():
    emit_progress({"configurable": None}, "hello", kind="info")


def test_emit_progress_calls_callback_with_event_dict():
    events = []
    config = {"configurable": {"progress_callback": events.append}}

    emit_progress(config, "Researcher: searching", kind="search")

    assert events == [{"message": "Researcher: searching", "kind": "search"}]


def test_emit_progress_callback_exception_does_not_break_research():
    called = []

    def callback(_event):
        called.append(True)
        raise RuntimeError("display failed")

    config = {"configurable": {"progress_callback": callback}}

    emit_progress(config, "Researcher: searching", kind="search")

    assert called == [True]


def test_emit_progress_defaults_kind_to_info():
    events = []
    config = {"configurable": {"progress_callback": events.append}}

    emit_progress(config, "hello")

    assert events == [{"message": "hello", "kind": "info"}]
