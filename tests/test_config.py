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
        assert settings.models.reviewer is None  # unset roles fall back to models.default

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

    def test_get_reviewer_model_defaults_to_default(self):
        settings = ORASettings()
        assert get_reviewer_model(settings) == settings.models.default

    def test_get_reviewer_model_fallback_when_reviewer_is_none(self):
        from ora.config import ModelSettings

        settings = ORASettings(models=ModelSettings(default="my-default", reviewer=None))
        assert get_reviewer_model(settings) == "my-default"

    def test_get_reviewer_model_uses_reviewer_override(self):
        from ora.config import ModelSettings

        settings = ORASettings(
            models=ModelSettings(default="my-default", reviewer="deepseek-v4-pro")
        )
        assert get_reviewer_model(settings) == "deepseek-v4-pro"

    def test_get_supervisor_model_defaults_to_default(self):
        from ora.config import ModelSettings

        settings = ORASettings(models=ModelSettings(default="my-default"))
        assert get_supervisor_model(settings) == "my-default"

    def test_get_supervisor_model_uses_supervisor_override(self):
        from ora.config import ModelSettings

        settings = ORASettings(
            models=ModelSettings(default="my-default", supervisor="deepseek-v4-pro")
        )
        assert get_supervisor_model(settings) == "deepseek-v4-pro"


class FakeChatOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class TestGetLlmRouting:
    @pytest.fixture(autouse=True)
    def _isolate_from_user_config(self, monkeypatch):
        """get_llm calls load_config(); keep routing tests independent of the
        developer's real ~/.ora/config.yaml by substituting pure defaults."""
        monkeypatch.setattr("ora.config.load_config", lambda *a, **kw: ORASettings())

    def _capture_chat_openai(self, monkeypatch):
        captures = {}

        def factory(**kwargs):
            captures.update(kwargs)
            return FakeChatOpenAI(**kwargs)

        # get_llm imports ChatOpenAI locally from langchain_openai, so patch
        # the class there (ora.config has no module-level ChatOpenAI attribute).
        monkeypatch.setattr("langchain_openai.ChatOpenAI", factory)
        return captures

    def test_no_prefix_uses_default_provider_deepseek(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
        captures = self._capture_chat_openai(monkeypatch)
        get_llm("deepseek-v4-flash")
        assert captures["model"] == "deepseek-v4-flash"
        assert captures["base_url"] == "https://api.deepseek.com"
        assert captures["api_key"] == "sk-deepseek"
        assert captures["default_headers"] is None

    def test_openrouter_prefix_routes_to_openrouter(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
        captures = self._capture_chat_openai(monkeypatch)
        get_llm("openrouter:anthropic/claude-3.5-sonnet")
        assert captures["model"] == "anthropic/claude-3.5-sonnet"
        assert captures["base_url"] == "https://openrouter.ai/api/v1"
        assert captures["api_key"] == "sk-or"
        assert captures["default_headers"]["X-Title"] == "ORA"

    def test_unknown_provider_warns_and_falls_back(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
        captures = self._capture_chat_openai(monkeypatch)
        with pytest.warns(UserWarning, match="Unknown provider"):
            get_llm("openai:gpt-4.1")
        assert captures["model"] == "openai:gpt-4.1"  # prefix kept for unknown providers
        assert captures["base_url"] == "https://api.deepseek.com"

    def test_missing_key_raises_value_error(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="deepseek"):
            get_llm("deepseek-v4-flash")

    def test_deepseek_falls_back_to_openai_key(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-legacy")
        captures = self._capture_chat_openai(monkeypatch)
        get_llm("deepseek-v4-flash")
        assert captures["api_key"] == "sk-openai-legacy"

    def test_custom_provider_without_base_url_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MY_API_KEY", "sk-custom")
        config_file = tmp_path / "config.yaml"
        config_file.write_text("providers:\n  custom:\n    api_key_env: MY_API_KEY\n")

        monkeypatch.setattr(
            "ora.config.load_config", lambda *a, **kw: load_config(str(config_file))
        )
        with pytest.raises(ValueError, match="base_url"):
            get_llm("custom:my-model")


class TestSplitProvider:
    def test_no_prefix(self):
        from ora.config import _split_provider

        assert _split_provider("deepseek-v4-flash") == (None, "deepseek-v4-flash")

    def test_openrouter_prefix_with_slashes(self):
        from ora.config import _split_provider

        assert _split_provider("openrouter:anthropic/claude-3.5-sonnet") == (
            "openrouter",
            "anthropic/claude-3.5-sonnet",
        )


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
