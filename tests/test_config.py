"""Tests for configuration loading."""
import os
import tempfile
from ora.config import load_config, ORASettings, get_researcher_model, get_reviewer_model, get_supervisor_model


class TestORASettings:
    def test_defaults(self):
        settings = ORASettings()
        assert settings.models.default == "deepseek-v4-flash"
        assert settings.search.provider == "firecrawl"
        assert settings.limits.max_revisions == 3
        assert settings.limits.default_intensity == 2
        assert settings.deepseek_base_url == "https://api.deepseek.com"
        assert settings.models.reviewer == "deepseek-v4-pro"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("ORA_MODELS__DEFAULT", "deepseek-v4-flash")
        settings = ORASettings()
        assert settings.models.default == "deepseek-v4-flash"


class TestLoadConfig:
    def test_loads_yaml_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("limits:\n  default_intensity: 4\n")
            f.flush()
            config = load_config(f.name)
            assert config.limits.default_intensity == 4
            os.unlink(f.name)

    def test_get_researcher_model_defaults_to_default(self):
        settings = ORASettings()
        assert get_researcher_model(settings) == "deepseek-v4-flash"

    def test_get_researcher_model_uses_researcher_override(self):
        from ora.config import ModelSettings
        settings = ORASettings(models=ModelSettings(researcher="deepseek-v4-pro"))
        assert get_researcher_model(settings) == "deepseek-v4-pro"

    def test_get_reviewer_model_defaults_to_v4_pro(self):
        settings = ORASettings()
        assert get_reviewer_model(settings) == "deepseek-v4-pro"

    def test_get_reviewer_model_fallback_when_reviewer_is_none(self):
        from ora.config import ModelSettings
        settings = ORASettings(models=ModelSettings(reviewer=None))
        assert get_reviewer_model(settings) == "deepseek-v4-pro"

    def test_get_supervisor_model_defaults_to_v4_pro(self):
        settings = ORASettings()
        assert get_supervisor_model(settings) == "deepseek-v4-pro"
