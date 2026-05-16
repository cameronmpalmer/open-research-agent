"""Tests for CLI interface."""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from ora.cli import main, _format_progress_event


def _fake_settings():
    return SimpleNamespace(
        models=SimpleNamespace(
            default="deepseek-v4-flash",
            researcher="deepseek-v4-flash",
            supervisor="deepseek-v4-pro",
            reviewer="deepseek-v4-pro",
        ),
        search=SimpleNamespace(provider="firecrawl", firecrawl_api_url="https://api.firecrawl.com"),
        output=SimpleNamespace(default_format="markdown", always_include_sources=True),
        limits=SimpleNamespace(max_revisions=3),
    )


class TestCLI:
    def test_format_progress_event_search(self):
        assert _format_progress_event({"kind": "search", "message": "Researcher: searching"}) == "🔎 Researcher: searching"

    def test_format_progress_event_unknown_kind_uses_bullet(self):
        assert _format_progress_event({"kind": "unexpected", "message": "Something happened"}) == "• Something happened"

    @pytest.mark.parametrize(
        "event, expected",
        [
            ({"kind": "search", "message": "Researcher: searching"}, "🔎 Researcher: searching"),
            ({"kind": "scrape", "message": "Researcher: scraping"}, "🌐 Researcher: scraping"),
            ({"kind": "success", "message": "Done"}, "✓ Done"),
            ({"kind": "error", "message": "Oops"}, "✗ Oops"),
            ({"kind": "write", "message": "Writer: drafting"}, "✍️ Writer: drafting"),
            ({"kind": "info", "message": "FYI"}, "• FYI"),
            ({"message": "Missing kind"}, "• Missing kind"),
        ],
    )
    def test_format_progress_event_icons(self, event, expected):
        assert _format_progress_event(event) == expected

    def test_cli_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "research" in result.output

    def test_python_module_entrypoint_help(self):
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "ora", "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert "Open Research Agent" in result.stdout
        assert "research" in result.stdout

    def test_top_level_help_lists_only_public_commands(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "config" in result.output
        assert "plan" in result.output
        assert "research" in result.output
        assert "bench" not in result.output

    def test_bench_command_is_not_registered(self):
        runner = CliRunner()
        result = runner.invoke(main, ["bench"])

        assert result.exit_code != 0
        assert "No such command" in result.output

    def test_research_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["research", "--help"])
        assert result.exit_code == 0
        assert "intensity" in result.output
        assert "quiet" in result.output

    def test_plan_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "--help"])
        assert result.exit_code == 0

    def test_plan_intensity_5_passes_through(self, monkeypatch):
        from ora import cli as cli_module

        received_states = []

        class FakePlanGraph:
            def invoke(self, state, config=None):
                received_states.append(state)
                return {"research_plan": "# Plan", "plan_approved": False, "messages": ["# Plan"]}

        monkeypatch.setattr(cli_module, "load_config", lambda: _fake_settings())
        monkeypatch.setattr(cli_module, "_spin", lambda func, message="Working...": func())
        monkeypatch.setattr(cli_module, "_print_markdown", lambda text: None)
        monkeypatch.setattr("ora.graph.build_plan_graph", lambda: FakePlanGraph())

        runner = CliRunner()
        result = runner.invoke(main, ["plan", "Rust vs Go", "--intensity", "5"])

        assert result.exit_code == 0
        assert received_states and received_states[0]["intensity"] == 5

    def test_config_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["config", "--help"])
        assert result.exit_code == 0

    def test_config_show_includes_intensity_table(self, monkeypatch):
        import ora.cli as cli_module

        monkeypatch.setattr(cli_module, "load_config", lambda: _fake_settings())
        monkeypatch.setattr("os.path.exists", lambda path: True)

        runner = CliRunner()
        result = runner.invoke(main, ["config", "--show"])

        assert result.exit_code == 0
        assert "Intensity Levels" in result.output
        assert "Quick" in result.output
        assert "Deep" in result.output
        assert "Exhaustive" in result.output

    def test_research_without_query_fails(self):
        runner = CliRunner()
        result = runner.invoke(main, ["research"])
        assert result.exit_code != 0

    def test_research_default_passes_progress_callback(self, monkeypatch):
        from ora import cli as cli_module

        received_configs = []

        class FakePlanGraph:
            def invoke(self, state, config=None):
                return {"research_plan": "# Plan", "plan_approved": False, "messages": ["# Plan"]}

        class FakeResearchGraph:
            def invoke(self, state, config=None):
                received_configs.append(config)
                callback = (config or {}).get("configurable", {}).get("progress_callback")
                if callback:
                    callback({"kind": "search", "message": 'Researcher: searching "Rust vs Go"'})
                    callback({"kind": "success", "message": "Writer: draft generated, 42 chars"})
                return {"draft_report": "# Research\nbody", "sources": [], "findings": []}

        monkeypatch.setattr(cli_module, "load_config", lambda: _fake_settings())
        monkeypatch.setattr(cli_module, "_spin", lambda func, message="Working...": func())
        monkeypatch.setattr(cli_module, "_print_markdown", lambda text: None)
        monkeypatch.setattr(cli_module.click, "prompt", lambda *a, **kw: "A")
        monkeypatch.setattr("ora.graph.build_plan_graph", lambda: FakePlanGraph())
        monkeypatch.setattr("ora.graph.build_research_graph", lambda *a, **kw: FakeResearchGraph())

        runner = CliRunner()
        result = runner.invoke(main, ["research", "Rust vs Go"])

        assert result.exit_code == 0
        assert received_configs and received_configs[0]["configurable"]["progress_callback"] == cli_module._print_progress_event
        assert "🔎 Researcher: searching \"Rust vs Go\"" in result.output
        assert "✓ Writer: draft generated, 42 chars" in result.output

    def test_research_quiet_does_not_pass_progress_callback(self, monkeypatch):
        from ora import cli as cli_module

        received_configs = []

        class FakePlanGraph:
            def invoke(self, state, config=None):
                return {"research_plan": "# Plan", "plan_approved": False, "messages": ["# Plan"]}

        class FakeResearchGraph:
            def invoke(self, state, config=None):
                received_configs.append(config)
                return {"draft_report": "# Research\nbody", "sources": [], "findings": []}

        monkeypatch.setattr(cli_module, "load_config", lambda: _fake_settings())
        monkeypatch.setattr(cli_module, "_spin", lambda func, message="Working...": func())
        monkeypatch.setattr(cli_module, "_print_markdown", lambda text: None)
        monkeypatch.setattr(cli_module.click, "prompt", lambda *a, **kw: "A")
        monkeypatch.setattr("ora.graph.build_plan_graph", lambda: FakePlanGraph())
        monkeypatch.setattr("ora.graph.build_research_graph", lambda *a, **kw: FakeResearchGraph())

        runner = CliRunner()
        result = runner.invoke(main, ["research", "Rust vs Go", "--quiet"])

        assert result.exit_code == 0
        assert received_configs == [None]

    def test_research_warns_when_reviewer_flags_are_ignored(self, monkeypatch):
        from ora import cli as cli_module

        class FakePlanGraph:
            def invoke(self, state, config=None):
                return {"research_plan": "# Plan", "plan_approved": False, "messages": ["# Plan"]}

        class FakeResearchGraph:
            def invoke(self, state, config=None):
                return {"draft_report": "# Research\nbody", "sources": [], "findings": []}

        monkeypatch.setattr(cli_module, "load_config", lambda: _fake_settings())
        monkeypatch.setattr(cli_module, "_spin", lambda func, message="Working...": func())
        monkeypatch.setattr(cli_module, "_print_markdown", lambda text: None)
        monkeypatch.setattr(cli_module.click, "prompt", lambda *a, **kw: "A")
        monkeypatch.setattr("ora.graph.build_plan_graph", lambda: FakePlanGraph())
        monkeypatch.setattr("ora.graph.build_research_graph", lambda *a, **kw: FakeResearchGraph())

        runner = CliRunner()
        result = runner.invoke(main, ["research", "Rust vs Go", "--no-review", "--max-revisions", "7"])

        assert result.exit_code == 0
        assert "--no-review is only relevant for intensity 3+" in result.output
        assert "--max-revisions is only relevant for intensity 3+" in result.output

    def test_research_edit_path_modifies_plan(self, monkeypatch):
        from ora import cli as cli_module

        received_plans = []
        prompt_calls = []

        def _fake_prompt(*a, **kw):
            prompt_calls.append(1)
            return "E" if len(prompt_calls) == 1 else "A"

        class FakePlanGraph:
            def invoke(self, state, config=None):
                return {"research_plan": "# Original\n\n## Section\n\ncontent", "plan_approved": False, "messages": ["# Plan"]}

        class FakeResearchGraph:
            def invoke(self, state, config=None):
                received_plans.append(state.get("research_plan"))
                return {"draft_report": "# Research\nbody", "sources": [], "findings": []}

        monkeypatch.setattr(cli_module, "load_config", lambda: _fake_settings())
        monkeypatch.setattr(cli_module, "_spin", lambda func, message="Working...": func())
        monkeypatch.setattr(cli_module, "_print_markdown", lambda text: None)
        monkeypatch.setattr(cli_module.click, "prompt", _fake_prompt)
        monkeypatch.setattr(cli_module.click, "edit", lambda text=None, extension=None: "# Edited\n\n## New Section\n\nedited content\n")
        monkeypatch.setattr("ora.graph.build_plan_graph", lambda: FakePlanGraph())
        monkeypatch.setattr("ora.graph.build_research_graph", lambda *a, **kw: FakeResearchGraph())

        runner = CliRunner()
        result = runner.invoke(main, ["research", "Rust vs Go"])

        assert result.exit_code == 0
        assert received_plans and "edited content" in received_plans[0]

    def test_research_revise_path_calls_supervisor(self, monkeypatch):
        from ora import cli as cli_module
        from ora.agents import supervisor as supervisor_module

        prompt_count = 0
        revise_calls = []

        def _fake_prompt(*a, **kw):
            nonlocal prompt_count
            prompt_count += 1
            if prompt_count == 1:
                return "R"
            elif prompt_count == 2:
                return "focus more on performance"
            return "A"

        def _fake_revise(query, intensity, plan, feedback):
            revise_calls.append((query, plan, feedback))
            return "# Revised\n\nMore on performance."

        class FakePlanGraph:
            def invoke(self, state, config=None):
                return {"research_plan": "# Original", "plan_approved": False, "messages": ["# Plan"]}

        class FakeResearchGraph:
            def invoke(self, state, config=None):
                return {"draft_report": "# Research\nbody", "sources": [], "findings": []}

        monkeypatch.setattr(cli_module, "load_config", lambda: _fake_settings())
        monkeypatch.setattr(cli_module, "_spin", lambda func, message="Working...": func())
        monkeypatch.setattr(cli_module, "_print_markdown", lambda text: None)
        monkeypatch.setattr(cli_module.click, "prompt", _fake_prompt)
        monkeypatch.setattr(supervisor_module, "revise_plan_text", _fake_revise)
        monkeypatch.setattr("ora.graph.build_plan_graph", lambda: FakePlanGraph())
        monkeypatch.setattr("ora.graph.build_research_graph", lambda *a, **kw: FakeResearchGraph())

        runner = CliRunner()
        result = runner.invoke(main, ["research", "Rust vs Go"])

        assert result.exit_code == 0
        assert len(revise_calls) == 1
        assert revise_calls[0][2] == "focus more on performance"

    def test_research_cancel_does_not_run_research(self, monkeypatch):
        from ora import cli as cli_module

        research_ran = []

        class FakePlanGraph:
            def invoke(self, state, config=None):
                return {"research_plan": "# Plan", "plan_approved": False, "messages": ["# Plan"]}

        class FakeResearchGraph:
            def invoke(self, state, config=None):
                research_ran.append(True)
                return {"draft_report": "# Research\nbody", "sources": [], "findings": []}

        monkeypatch.setattr(cli_module, "load_config", lambda: _fake_settings())
        monkeypatch.setattr(cli_module, "_spin", lambda func, message="Working...": func())
        monkeypatch.setattr(cli_module, "_print_markdown", lambda text: None)
        monkeypatch.setattr(cli_module.click, "prompt", lambda *a, **kw: "C")
        monkeypatch.setattr("ora.graph.build_plan_graph", lambda: FakePlanGraph())
        monkeypatch.setattr("ora.graph.build_research_graph", lambda *a, **kw: FakeResearchGraph())

        runner = CliRunner()
        result = runner.invoke(main, ["research", "Rust vs Go"])

        assert result.exit_code == 0
        assert research_ran == []

    def test_auto_approve_skips_prompt(self, monkeypatch):
        from ora import cli as cli_module

        prompt_called = []

        class FakePlanGraph:
            def invoke(self, state, config=None):
                return {"research_plan": "# Plan", "plan_approved": False, "messages": ["# Plan"]}

        class FakeResearchGraph:
            def invoke(self, state, config=None):
                return {"draft_report": "# Research\nbody", "sources": [], "findings": []}

        monkeypatch.setattr(cli_module, "load_config", lambda: _fake_settings())
        monkeypatch.setattr(cli_module, "_spin", lambda func, message="Working...": func())
        monkeypatch.setattr(cli_module, "_print_markdown", lambda text: None)
        monkeypatch.setattr(cli_module.click, "prompt", lambda *a, **kw: prompt_called.append(1) or "A")
        monkeypatch.setattr("ora.graph.build_plan_graph", lambda: FakePlanGraph())
        monkeypatch.setattr("ora.graph.build_research_graph", lambda *a, **kw: FakeResearchGraph())

        runner = CliRunner()
        result = runner.invoke(main, ["research", "Rust vs Go", "-y"])

        assert result.exit_code == 0
        assert prompt_called == []  # prompt was never invoked

    def test_hide_plan_on_autoapprove_suppresses_output(self, monkeypatch):
        from ora import cli as cli_module

        plan_rendered = []

        class FakePlanGraph:
            def invoke(self, state, config=None):
                return {"research_plan": "# Plan", "plan_approved": False, "messages": ["# Plan"]}

        class FakeResearchGraph:
            def invoke(self, state, config=None):
                return {"draft_report": "# Research\nbody", "sources": [], "findings": []}

        monkeypatch.setattr(cli_module, "load_config", lambda: _fake_settings())
        monkeypatch.setattr(cli_module, "_spin", lambda func, message="Working...": func())
        monkeypatch.setattr(cli_module, "_print_markdown", lambda text: plan_rendered.append(text))
        monkeypatch.setattr("ora.graph.build_plan_graph", lambda: FakePlanGraph())
        monkeypatch.setattr("ora.graph.build_research_graph", lambda *a, **kw: FakeResearchGraph())

        runner = CliRunner()
        result = runner.invoke(main, ["research", "Rust vs Go", "-y", "--hide-plan-on-autoapprove"])

        assert result.exit_code == 0
        assert plan_rendered == []  # plan never printed
