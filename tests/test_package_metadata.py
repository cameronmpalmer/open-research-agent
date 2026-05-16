"""Tests for public package metadata."""
from pathlib import Path


def _pyproject_text() -> str:
    return Path("pyproject.toml").read_text()


def test_project_name_uses_available_pypi_name():
    text = _pyproject_text()

    assert 'name = "open-research-agent"' in text
    assert 'name = "ora"' not in text


def test_console_scripts_include_primary_name_and_alias():
    text = _pyproject_text()

    assert '[project.scripts]' in text
    assert 'open-research-agent = "ora.cli:main"' in text
    assert 'ora = "ora.cli:main"' in text
