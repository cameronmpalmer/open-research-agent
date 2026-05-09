"""Tests for CLI interface."""
from click.testing import CliRunner
from ora.cli import main


class TestCLI:
    def test_cli_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "research" in result.output

    def test_research_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["research", "--help"])
        assert result.exit_code == 0
        assert "intensity" in result.output

    def test_plan_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "--help"])
        assert result.exit_code == 0

    def test_config_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["config", "--help"])
        assert result.exit_code == 0

    def test_research_without_query_fails(self):
        runner = CliRunner()
        result = runner.invoke(main, ["research"])
        assert result.exit_code != 0
