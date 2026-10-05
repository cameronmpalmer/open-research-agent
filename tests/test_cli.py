"""Tests for CLI interface."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from ora.cli import _format_progress_event, main


def _fake_settings():
    return SimpleNamespace(
        models=SimpleNamespace(
            default="deepseek-v4-flash",
            researcher="deepseek-v4-flash",
            supervisor="deepseek-v4-pro",
            writer=None,
            reviewer="deepseek-v4-pro",
        ),
        search=SimpleNamespace(provider="firecrawl", firecrawl_api_url="https://api.firecrawl.com"),
        output=SimpleNamespace(default_format="markdown", always_include_sources=True),
        limits=SimpleNamespace(max_revisions=3),
        provider=SimpleNamespace(default="deepseek"),
        providers={},
        deepseek_base_url="https://api.deepseek.com",
    )


class TestCLI:
    def test_format_progress_event_search(self):
        assert (
            _format_progress_event({"kind": "search", "message": "Researcher: searching"})
            == "🔎 Researcher: searching"
        )

    def test_format_progress_event_unknown_kind_uses_bullet(self):
        assert (
            _format_progress_event({"kind": "unexpected", "message": "Something happened"})
            == "• Something happened"
        )

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

    def test_config_show_describes_supervisor_as_planning(self, monkeypatch):
        import ora.cli as cli_module

        monkeypatch.setattr(cli_module, "load_config", lambda: _fake_settings())

        result = CliRunner().invoke(main, ["config", "--show"])

        assert result.exit_code == 0
        assert "Supervisor (planning): deepseek-v4-pro" in result.output.splitlines()

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

    def test_config_init_creates_file(self, tmp_path):
        """config --init must create ~/.ora/config.yaml on a fresh HOME."""
        runner = CliRunner()
        result = runner.invoke(main, ["config", "--init"], env={"HOME": str(tmp_path)})

        assert result.exit_code == 0
        assert "Config created" in result.output
        assert (tmp_path / ".ora" / "config.yaml").exists()
        content = (tmp_path / ".ora" / "config.yaml").read_text()
        assert "openrouter" in content
        assert "api_key_env" in content

    def test_config_show_lists_providers(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, ["config", "--show"], env={"HOME": str(tmp_path)})
        assert result.exit_code == 0
        assert "Default provider: deepseek" in result.output
        assert "deepseek" in result.output
        assert "openrouter" in result.output
        assert "key set:" in result.output

    def test_research_without_query_fails(self):
        runner = CliRunner()
        result = runner.invoke(main, ["research"])
        assert result.exit_code != 0

    def test_research_refuses_to_start_without_decodo_key(self, monkeypatch):
        """A missing Decodo key must fail before a plan is generated."""
        from ora import cli as cli_module

        plan_graph_calls = []

        class FakePlanGraph:
            def invoke(self, state, config=None):
                plan_graph_calls.append(state)
                return {"research_plan": "# Plan", "plan_approved": True, "messages": ["# Plan"]}

        settings = _fake_settings()
        settings.search = SimpleNamespace(
            provider="decodo",
            firecrawl_api_url="http://localhost:3002",
            decodo_api_key_env="DECODO_API_KEY",
        )
        monkeypatch.setattr(cli_module, "load_config", lambda: settings)
        monkeypatch.setattr("ora.graph.build_plan_graph", lambda: FakePlanGraph())
        monkeypatch.delenv("DECODO_API_KEY", raising=False)

        runner = CliRunner()
        result = runner.invoke(main, ["research", "Rust vs Go", "-y", "--no-save"])

        assert result.exit_code != 0
        assert "DECODO_API_KEY" in result.output
        assert plan_graph_calls == [], "a plan was generated despite the missing Decodo key"

    def test_research_refuses_to_start_when_decodo_key_is_rejected(self, monkeypatch):
        """A key Decodo rejects must fail before a plan is generated."""
        from ora import cli as cli_module
        from ora.tools import search as search_mod

        plan_graph_calls = []

        class FakePlanGraph:
            def invoke(self, state, config=None):
                plan_graph_calls.append(state)
                return {"research_plan": "# Plan", "plan_approved": True, "messages": ["# Plan"]}

        settings = _fake_settings()
        settings.search = SimpleNamespace(
            provider="decodo",
            firecrawl_api_url="http://localhost:3002",
            decodo_api_key_env="DECODO_API_KEY",
            decodo_api_url="https://scraper-api.decodo.com/v2/scrape",
        )
        monkeypatch.setattr(cli_module, "load_config", lambda: settings)
        monkeypatch.setattr("ora.graph.build_plan_graph", lambda: FakePlanGraph())
        monkeypatch.setenv("DECODO_API_KEY", "wrong-key")
        monkeypatch.setattr(
            search_mod.requests,
            "post",
            lambda *a, **kw: SimpleNamespace(status_code=401, json=lambda: {"status": "failed"}),
        )

        runner = CliRunner()
        result = runner.invoke(main, ["research", "Rust vs Go", "-y", "--no-save"])

        assert result.exit_code != 0
        assert "DECODO_API_KEY" in result.output
        assert plan_graph_calls == [], "a plan was generated despite the rejected key"

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
        result = runner.invoke(main, ["research", "Rust vs Go", "--no-save"])

        assert result.exit_code == 0
        assert (
            received_configs
            and received_configs[0]["configurable"]["progress_callback"]
            == cli_module._print_progress_event
        )
        assert '🔎 Researcher: searching "Rust vs Go"' in result.output
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
        result = runner.invoke(main, ["research", "Rust vs Go", "--quiet", "--no-save"])

        assert result.exit_code == 0
        assert received_configs == [None]

    def test_research_banner_lists_all_four_roles(self, monkeypatch):
        """The startup banner must name Supervisor, Researcher, Writer, and
        Reviewer with their resolved models, not just Researcher and Reviewer."""
        from ora import cli as cli_module

        settings = _fake_settings()
        settings.models = SimpleNamespace(
            default="m-default",
            researcher="m-researcher",
            supervisor="m-supervisor",
            writer="m-writer",
            reviewer="m-reviewer",
        )

        class FakePlanGraph:
            def invoke(self, state, config=None):
                return {"research_plan": "# Plan", "plan_approved": False, "messages": ["# Plan"]}

        class FakeResearchGraph:
            def invoke(self, state, config=None):
                return {"draft_report": "# Research\nbody", "sources": [], "findings": []}

        monkeypatch.setattr(cli_module, "load_config", lambda: settings)
        monkeypatch.setattr(cli_module, "_spin", lambda func, message="Working...": func())
        monkeypatch.setattr(cli_module, "_print_markdown", lambda text: None)
        # Cancel at the approval prompt so the real research phase never runs.
        monkeypatch.setattr(cli_module.click, "prompt", lambda *a, **kw: "C")
        monkeypatch.setattr("ora.graph.build_plan_graph", lambda: FakePlanGraph())
        monkeypatch.setattr("ora.graph.build_research_graph", lambda *a, **kw: FakeResearchGraph())

        runner = CliRunner()
        result = runner.invoke(main, ["research", "Rust vs Go", "--intensity", "3", "--no-save"])

        assert result.exit_code == 0
        assert "Supervisor: m-supervisor" in result.output
        assert "Researcher: m-researcher" in result.output
        assert "Writer:     m-writer" in result.output
        assert "Reviewer:   m-reviewer" in result.output

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
        result = runner.invoke(
            main, ["research", "Rust vs Go", "--no-review", "--max-revisions", "7", "--no-save"]
        )

        assert result.exit_code == 0
        assert "--no-review is only relevant for intensity 3+" in result.output
        assert "--max-revisions is only relevant for intensity 3+" in result.output

    def _patch_research_graph(self, monkeypatch, research_final_state):
        """Monkeypatch the CLI research flow to return a fixed final state
        from a fake research graph (follows the existing FakeGraph style)."""
        from ora import cli as cli_module

        class FakePlanGraph:
            def invoke(self, state, config=None):
                return {"research_plan": "# Plan", "plan_approved": False, "messages": ["# Plan"]}

        class FakeResearchGraph:
            def invoke(self, state, config=None):
                return research_final_state

        monkeypatch.setattr(cli_module, "load_config", lambda: _fake_settings())
        monkeypatch.setattr(cli_module, "_spin", lambda func, message="Working...": func())
        monkeypatch.setattr(cli_module, "_print_markdown", lambda text: None)
        monkeypatch.setattr(cli_module.click, "prompt", lambda *a, **kw: "A")
        monkeypatch.setattr("ora.graph.build_plan_graph", lambda: FakePlanGraph())
        monkeypatch.setattr("ora.graph.build_research_graph", lambda *a, **kw: FakeResearchGraph())

    def test_research_warns_on_revise_capped_end(self, monkeypatch):
        """A final REVISE verdict capped by the revision budget must surface a
        warning with the open/exhausted item counts."""
        from ora.state import ReviewVerdict

        self._patch_research_graph(
            monkeypatch,
            {
                "draft_report": "# Research\nbody\n\n## Changes made\n- partial.\n",
                "sources": [],
                "findings": [],
                "review_verdict": ReviewVerdict(verdict="REVISE", blocking=["still missing"]),
                "review_items": [
                    {"category": "blocking", "text": "still missing", "status": "open"}
                ],
                "revision_count": 3,
            },
        )

        runner = CliRunner()
        result = runner.invoke(main, ["research", "Rust vs Go", "--no-save"])

        assert result.exit_code == 0
        assert "Report finalized with unresolved review items: 1 open, 0 exhausted" in result.output

    def test_research_warns_on_pass_with_accepted_gaps(self, monkeypatch):
        """A final PASS that accepted evidence gaps (unresolvable_gaps) also
        surfaces the gap count so the user checks the report's evidence notes."""
        from ora.state import ReviewVerdict

        self._patch_research_graph(
            monkeypatch,
            {
                "draft_report": "# Research\nbody\n\n## Evidence gaps\nNot available.\n",
                "sources": [],
                "findings": [],
                "review_verdict": ReviewVerdict(
                    verdict="PASS",
                    unresolvable_gaps=["2026 market data unavailable"],
                ),
                "review_items": [],
                "revision_count": 2,
            },
        )

        runner = CliRunner()
        result = runner.invoke(main, ["research", "Rust vs Go", "--no-save"])

        assert result.exit_code == 0
        assert "Report finalized with unresolved review items: 0 open, 1 exhausted" in result.output
        # M4: the accepted evidence-gap text is printed so the fold path is
        # auditable from the CLI.
        assert "Accepted evidence gaps:" in result.output
        assert "- 2026 market data unavailable" in result.output

    def test_research_warning_truncates_long_gap_text(self, monkeypatch):
        """Gap text samples in the end warning are truncated to ~80 chars."""
        from ora.state import ReviewVerdict

        long_gap = "This accepted evidence gap has an extremely long explanatory sentence " * 2
        self._patch_research_graph(
            monkeypatch,
            {
                "draft_report": "# Research\nbody",
                "sources": [],
                "findings": [],
                "review_verdict": ReviewVerdict(verdict="PASS", unresolvable_gaps=[long_gap]),
                "review_items": [],
                "revision_count": 2,
            },
        )

        runner = CliRunner()
        result = runner.invoke(main, ["research", "Rust vs Go", "--no-save"])

        assert result.exit_code == 0
        assert "Accepted evidence gaps:" in result.output
        assert long_gap[:50] in result.output
        # Truncated with an ellipsis, not the full text.
        assert "..." in result.output
        assert long_gap not in result.output

    def test_research_edit_path_modifies_plan(self, monkeypatch):
        from ora import cli as cli_module

        received_plans = []
        prompt_calls = []

        def _fake_prompt(*a, **kw):
            prompt_calls.append(1)
            return "E" if len(prompt_calls) == 1 else "A"

        class FakePlanGraph:
            def invoke(self, state, config=None):
                return {
                    "research_plan": "# Original\n\n## Section\n\ncontent",
                    "plan_approved": False,
                    "messages": ["# Plan"],
                }

        class FakeResearchGraph:
            def invoke(self, state, config=None):
                received_plans.append(state.get("research_plan"))
                return {"draft_report": "# Research\nbody", "sources": [], "findings": []}

        monkeypatch.setattr(cli_module, "load_config", lambda: _fake_settings())
        monkeypatch.setattr(cli_module, "_spin", lambda func, message="Working...": func())
        monkeypatch.setattr(cli_module, "_print_markdown", lambda text: None)
        monkeypatch.setattr(cli_module.click, "prompt", _fake_prompt)
        monkeypatch.setattr(
            cli_module.click,
            "edit",
            lambda text=None, extension=None: "# Edited\n\n## New Section\n\nedited content\n",
        )
        monkeypatch.setattr("ora.graph.build_plan_graph", lambda: FakePlanGraph())
        monkeypatch.setattr("ora.graph.build_research_graph", lambda *a, **kw: FakeResearchGraph())

        runner = CliRunner()
        result = runner.invoke(main, ["research", "Rust vs Go", "--no-save"])

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
            return "# Revised\n\nMore on performance.", []

        class FakePlanGraph:
            def invoke(self, state, config=None):
                return {
                    "research_plan": "# Original",
                    "plan_approved": False,
                    "messages": ["# Plan"],
                }

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
        result = runner.invoke(main, ["research", "Rust vs Go", "--no-save"])

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
        result = runner.invoke(main, ["research", "Rust vs Go", "--no-save"])

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
        monkeypatch.setattr(
            cli_module.click, "prompt", lambda *a, **kw: prompt_called.append(1) or "A"
        )
        monkeypatch.setattr("ora.graph.build_plan_graph", lambda: FakePlanGraph())
        monkeypatch.setattr("ora.graph.build_research_graph", lambda *a, **kw: FakeResearchGraph())

        runner = CliRunner()
        result = runner.invoke(main, ["research", "Rust vs Go", "-y", "--no-save"])

        assert result.exit_code == 0
        assert prompt_called == []  # prompt was never invoked

    def test_cli_reparses_search_queries_after_edit(self, monkeypatch):
        """After user edits plan, search_queries should be re-extracted."""
        from ora import cli as cli_module

        prompt_values = ["E", "A"]  # Edit then approve

        def _fake_prompt(text, **kwargs):
            return prompt_values.pop(0)

        # Plan text with search_queries in it
        plan_text = '# Research Plan\n\nSome topics.\n\n```search_queries\n["edit query one"]\n```'

        class FakePlanGraph:
            def invoke(self, state, config=None):
                return {"research_plan": "# Original", "messages": ["# Original"]}

        research_invoked = []

        class FakeResearchGraph:
            def invoke(self, state, config=None):
                research_invoked.append(state)
                return {"draft_report": "# Report", "sources": [], "findings": []}

        monkeypatch.setattr(cli_module, "load_config", lambda: _fake_settings())
        monkeypatch.setattr(cli_module, "_spin", lambda func, message="Working...": func())
        monkeypatch.setattr(cli_module, "_print_markdown", lambda text: None)
        monkeypatch.setattr(cli_module.click, "prompt", _fake_prompt)
        monkeypatch.setattr(cli_module.click, "edit", lambda text=None, extension=".md": plan_text)
        monkeypatch.setattr("ora.graph.build_plan_graph", lambda: FakePlanGraph())
        monkeypatch.setattr("ora.graph.build_research_graph", lambda *a, **kw: FakeResearchGraph())

        runner = CliRunner()
        result = runner.invoke(main, ["research", "test query", "--no-save"])

        assert result.exit_code == 0
        assert len(research_invoked) == 1
        research_state = research_invoked[0]
        assert "search_queries" in research_state
        assert research_state["search_queries"] == ["edit query one"]

    def test_cli_reparses_search_queries_after_revise(self, monkeypatch):
        """After supervisor revises plan, search_queries should be re-extracted."""
        from ora import cli as cli_module
        from ora.agents import supervisor as supervisor_module

        prompt_values = ["R", "focus more on that", "A"]

        def _fake_prompt(text, **kwargs):
            return prompt_values.pop(0)

        revised_plan = '# Revised Plan\n\nUpdated content.\n\n```search_queries\n["revised query a", "revised query b"]\n```'
        original_plan = '# Original\n\n```search_queries\n["orig q1"]\n```'

        class FakePlanGraph:
            def invoke(self, state, config=None):
                return {
                    "research_plan": original_plan,
                    "search_queries": ["orig q1"],
                    "messages": ["# Original"],
                }

        research_invoked = []

        class FakeResearchGraph:
            def invoke(self, state, config=None):
                research_invoked.append(state)
                return {"draft_report": "# Report", "sources": [], "findings": []}

        monkeypatch.setattr(cli_module, "load_config", lambda: _fake_settings())
        monkeypatch.setattr(cli_module, "_spin", lambda func, message="Working...": func())
        monkeypatch.setattr(cli_module, "_print_markdown", lambda text: None)
        monkeypatch.setattr(cli_module.click, "prompt", _fake_prompt)
        monkeypatch.setattr(
            supervisor_module,
            "revise_plan_text",
            lambda q, i, p, f: (revised_plan, ["revised query a", "revised query b"]),
        )
        monkeypatch.setattr("ora.graph.build_plan_graph", lambda: FakePlanGraph())
        monkeypatch.setattr("ora.graph.build_research_graph", lambda *a, **kw: FakeResearchGraph())

        runner = CliRunner()
        result = runner.invoke(main, ["research", "test query", "--no-save"])

        assert result.exit_code == 0
        assert len(research_invoked) == 1
        research_state = research_invoked[0]
        assert research_state["search_queries"] == ["revised query a", "revised query b"]

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
        result = runner.invoke(
            main, ["research", "Rust vs Go", "-y", "--hide-plan-on-autoapprove", "--no-save"]
        )

        assert result.exit_code == 0
        # The plan must never be rendered under --hide-plan-on-autoapprove.
        # (The final report may be rendered to stdout when --no-save is used.)
        assert all("# Plan" not in text for text in plan_rendered)

    def test_model_flags_install_config_overrides(self, monkeypatch):
        """--model and --reviewer-model must install run-scoped config overrides."""
        from ora import cli as cli_module

        received_states = []
        override_calls = []
        clear_calls = []

        class FakePlanGraph:
            def invoke(self, state, config=None):
                return {"research_plan": "# Plan", "plan_approved": False, "messages": ["# Plan"]}

        class FakeResearchGraph:
            def invoke(self, state, config=None):
                received_states.append(state)
                return {"draft_report": "# Research\nbody", "sources": [], "findings": []}

        def record_set(role, model_name):
            override_calls.append((role, model_name))

        def record_clear():
            clear_calls.append(True)

        monkeypatch.setattr(cli_module, "set_model_override", record_set)
        monkeypatch.setattr(cli_module, "clear_model_overrides", record_clear)
        monkeypatch.setattr(cli_module, "load_config", lambda: _fake_settings())
        monkeypatch.setattr(cli_module, "_spin", lambda func, message="Working...": func())
        monkeypatch.setattr(cli_module, "_print_markdown", lambda text: None)
        monkeypatch.setattr(cli_module.click, "prompt", lambda *a, **kw: "A")
        monkeypatch.setattr("ora.graph.build_plan_graph", lambda: FakePlanGraph())
        monkeypatch.setattr("ora.graph.build_research_graph", lambda *a, **kw: FakeResearchGraph())

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "research",
                "Rust vs Go",
                "--no-save",
                "--model",
                "openrouter:qwen/qwen3.7-flash",
                "--reviewer-model",
                "openrouter:deepseek/deepseek-v4-pro",
            ],
        )

        assert result.exit_code == 0
        assert ("researcher", "openrouter:qwen/qwen3.7-flash") in override_calls
        assert ("reviewer", "openrouter:deepseek/deepseek-v4-pro") in override_calls
        assert clear_calls  # overrides cleared after the run
        # Model overrides travel via config, not via graph state.
        assert "researcher_model" not in received_states[0]
        assert "reviewer_model" not in received_states[0]

    def test_research_passes_max_revisions_flag_into_graph_state(self, monkeypatch):
        """An explicit --max-revisions flag must reach the research graph as
        plan_result['max_revisions'] (intensity 3+ so the cap is live)."""
        from ora import cli as cli_module

        received_states = []

        class FakePlanGraph:
            def invoke(self, state, config=None):
                return {"research_plan": "# Plan", "plan_approved": False, "messages": ["# Plan"]}

        class FakeResearchGraph:
            def invoke(self, state, config=None):
                received_states.append(state)
                return {"draft_report": "# Research\nbody", "sources": [], "findings": []}

        monkeypatch.setattr(cli_module, "load_config", lambda: _fake_settings())
        monkeypatch.setattr(cli_module, "_spin", lambda func, message="Working...": func())
        monkeypatch.setattr(cli_module, "_print_markdown", lambda text: None)
        monkeypatch.setattr("ora.graph.build_plan_graph", lambda: FakePlanGraph())
        monkeypatch.setattr("ora.graph.build_research_graph", lambda *a, **kw: FakeResearchGraph())

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "research",
                "Rust vs Go",
                "-y",
                "--intensity",
                "3",
                "--max-revisions",
                "5",
                "--no-save",
            ],
        )

        assert result.exit_code == 0
        assert received_states
        assert received_states[0]["max_revisions"] == 5

    def test_research_honors_config_limits_max_revisions_when_flag_default(self, monkeypatch):
        """When --max-revisions is not passed (flag None), the configured
        settings.limits.max_revisions value is wired into the graph state."""
        from ora import cli as cli_module

        settings = _fake_settings()
        settings.limits = SimpleNamespace(max_revisions=5, default_intensity=2)
        received_states = []

        class FakePlanGraph:
            def invoke(self, state, config=None):
                return {"research_plan": "# Plan", "plan_approved": False, "messages": ["# Plan"]}

        class FakeResearchGraph:
            def invoke(self, state, config=None):
                received_states.append(state)
                return {"draft_report": "# Research\nbody", "sources": [], "findings": []}

        monkeypatch.setattr(cli_module, "load_config", lambda: settings)
        monkeypatch.setattr(cli_module, "_spin", lambda func, message="Working...": func())
        monkeypatch.setattr(cli_module, "_print_markdown", lambda text: None)
        monkeypatch.setattr("ora.graph.build_plan_graph", lambda: FakePlanGraph())
        monkeypatch.setattr("ora.graph.build_research_graph", lambda *a, **kw: FakeResearchGraph())

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["research", "Rust vs Go", "-y", "--intensity", "3", "--no-save"],
        )

        assert result.exit_code == 0
        assert received_states
        assert received_states[0]["max_revisions"] == 5

    def test_explicit_max_revisions_three_overrides_config_five(self, monkeypatch):
        """--max-revisions 3 must beat config limits.max_revisions=5 (None sentinel)."""
        from ora import cli as cli_module

        received_states = []

        class FakePlanGraph:
            def invoke(self, state, config=None):
                return {"research_plan": "# Plan", "plan_approved": False, "messages": ["# Plan"]}

        class FakeResearchGraph:
            def invoke(self, state, config=None):
                received_states.append(state)
                return {"draft_report": "# Research\nbody", "sources": [], "findings": []}

        fake = _fake_settings()
        fake.limits = SimpleNamespace(max_revisions=5)
        monkeypatch.setattr(cli_module, "load_config", lambda: fake)
        monkeypatch.setattr(cli_module, "_spin", lambda func, message="Working...": func())
        monkeypatch.setattr(cli_module, "_print_markdown", lambda text: None)
        monkeypatch.setattr(cli_module.click, "prompt", lambda *a, **kw: "A")
        monkeypatch.setattr("ora.graph.build_plan_graph", lambda: FakePlanGraph())
        monkeypatch.setattr("ora.graph.build_research_graph", lambda *a, **kw: FakeResearchGraph())

        runner = CliRunner()
        result = runner.invoke(
            main, ["research", "Rust vs Go", "--no-save", "--max-revisions", "3"]
        )

        assert result.exit_code == 0
        assert received_states[0]["max_revisions"] == 3

    def test_research_accepts_explicit_single_revision_flag(self, monkeypatch):
        """An explicit --max-revisions 1 (single-audit budget) is accepted and
        reaches the graph state; IntRange lower bound is inclusive."""
        from ora import cli as cli_module

        received_states = []

        class FakePlanGraph:
            def invoke(self, state, config=None):
                return {"research_plan": "# Plan", "plan_approved": False, "messages": ["# Plan"]}

        class FakeResearchGraph:
            def invoke(self, state, config=None):
                received_states.append(state)
                return {"draft_report": "# Research\nbody", "sources": [], "findings": []}

        monkeypatch.setattr(cli_module, "load_config", lambda: _fake_settings())
        monkeypatch.setattr(cli_module, "_spin", lambda func, message="Working...": func())
        monkeypatch.setattr(cli_module, "_print_markdown", lambda text: None)
        monkeypatch.setattr("ora.graph.build_plan_graph", lambda: FakePlanGraph())
        monkeypatch.setattr("ora.graph.build_research_graph", lambda *a, **kw: FakeResearchGraph())

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "research",
                "Rust vs Go",
                "-y",
                "--intensity",
                "3",
                "--max-revisions",
                "1",
                "--no-save",
            ],
        )

        assert result.exit_code == 0
        assert received_states
        assert received_states[0]["max_revisions"] == 1

    @pytest.mark.parametrize("value", ["0", "-1"])
    def test_research_rejects_invalid_max_revisions_values(self, monkeypatch, value):
        """M1: 0 and negative budgets are rejected at option parse time
        (click.IntRange(min=1)) instead of being misinterpreted."""
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
        monkeypatch.setattr("ora.graph.build_plan_graph", lambda: FakePlanGraph())
        monkeypatch.setattr("ora.graph.build_research_graph", lambda *a, **kw: FakeResearchGraph())

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "research",
                "Rust vs Go",
                "-y",
                "--intensity",
                "3",
                "--max-revisions",
                value,
                "--no-save",
            ],
        )

        assert result.exit_code != 0
        assert "Invalid value" in result.output
