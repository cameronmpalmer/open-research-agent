"""Tests for configuration loading."""

import os
import tempfile

import pytest

from ora.config import (
    ORASettings,
    get_llm,
    get_researcher_model,
    get_reviewer_model,
    get_supervisor_model,
    load_config,
)


class TestORASettings:
    def test_defaults(self):
        settings = ORASettings()
        assert settings.models.default == "deepseek-v4-flash"
        assert settings.search.provider == "firecrawl"
        assert settings.limits.max_revisions == 3
        assert settings.limits.default_intensity == 2
        assert settings.deepseek_base_url == "https://api.deepseek.com"
        assert settings.provider.default == "deepseek"
        assert settings.providers == {}
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


class TestGetLlmWarning:
    def test_warns_on_colon_prefix(self, monkeypatch):
        """get_llm should warn when model name contains a provider:prefix."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        with pytest.warns(UserWarning, match="provider prefix"):
            get_llm("openai:gpt-4.1")

    def test_no_warn_without_prefix(self, monkeypatch):
        """get_llm should not warn for bare model names."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        import warnings

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            get_llm("deepseek-v4-flash")
        assert len(record) == 0, f"Unexpected warnings: {[str(w.message) for w in record]}"


class TestProviders:
    def test_builtin_defaults(self):
        from ora.config import DEFAULT_PROVIDERS

        assert DEFAULT_PROVIDERS["deepseek"].base_url == "https://api.deepseek.com"
        assert DEFAULT_PROVIDERS["deepseek"].api_key_env == "DEEPSEEK_API_KEY"
        assert DEFAULT_PROVIDERS["openrouter"].base_url == "https://openrouter.ai/api/v1"
        assert DEFAULT_PROVIDERS["openrouter"].api_key_env == "OPENROUTER_API_KEY"
        assert DEFAULT_PROVIDERS["openrouter"].headers["X-Title"] == "ORA"

    def test_loads_providers_from_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                "provider:\n"
                "  default: openrouter\n"
                "providers:\n"
                "  openrouter:\n"
                "    base_url: https://openrouter.ai/api/v1\n"
                "    api_key_env: OR_API_KEY\n"
                "    headers:\n"
                "      X-Title: Test\n"
            )
            f.flush()
            config = load_config(f.name)
            assert config.provider.default == "openrouter"
            assert config.providers["openrouter"].api_key_env == "OR_API_KEY"
            assert config.providers["openrouter"].headers["X-Title"] == "Test"
            os.unlink(f.name)

    def test_legacy_deepseek_base_url_migrates_without_providers_section(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("deepseek_base_url: https://legacy.example.com\n")
            f.flush()
            config = load_config(f.name)
            assert config.providers["deepseek"].base_url == "https://legacy.example.com"
            os.unlink(f.name)

    def test_providers_section_wins_over_legacy(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                "deepseek_base_url: https://legacy.example.com\n"
                "providers:\n"
                "  deepseek:\n"
                "    base_url: https://new.example.com\n"
            )
            f.flush()
            config = load_config(f.name)
            assert config.providers["deepseek"].base_url == "https://new.example.com"
            os.unlink(f.name)

    def test_env_default_provider_override(self, monkeypatch):
        monkeypatch.setenv("ORA_PROVIDER__DEFAULT", "openrouter")
        settings = ORASettings()
        assert settings.provider.default == "openrouter"

    def test_env_legacy_deepseek_base_url_applies_when_no_providers_configured(self, monkeypatch):
        monkeypatch.setenv("ORA_DEEPSEEK_BASE_URL", "https://env-legacy.example.com")
        config = load_config("/nonexistent-config.yaml")
        assert config.deepseek_base_url == "https://env-legacy.example.com"
