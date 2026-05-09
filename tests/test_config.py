"""Tests for configuration loading."""
import os
import tempfile
from ora.config import load_config, ORASettings


class TestORASettings:
    def test_defaults(self):
        settings = ORASettings()
        assert settings.models.default == "openai:gpt-4.1-mini"
        assert settings.search.provider == "firecrawl"
        assert settings.limits.max_revisions == 3
        assert settings.limits.default_intensity == 2

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("ORA_MODELS__DEFAULT", "anthropic:claude-sonnet-4-20250514")
        settings = ORASettings()
        assert settings.models.default == "anthropic:claude-sonnet-4-20250514"


class TestLoadConfig:
    def test_loads_yaml_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("limits:\n  default_intensity: 4\n")
            f.flush()
            config = load_config(f.name)
            assert config.limits.default_intensity == 4
            os.unlink(f.name)
