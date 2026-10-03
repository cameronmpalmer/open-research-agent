"""Tests for configuration loading."""

import os
import tempfile

import pytest

import ora.config
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
        # The Firecrawl fallback is opt-in; silently defaulting it on hides
        # Decodo failures.
        assert settings.search.fallback_to_firecrawl is False
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

    def test_yaml_limits_max_revisions_is_used(self):
        """A YAML limits.max_revisions value must surface on the loaded
        settings so the CLI can wire it into the research graph."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("limits:\n  max_revisions: 5\n")
            f.flush()
            config = load_config(f.name)
            assert config.limits.max_revisions == 5
            os.unlink(f.name)

    def test_yaml_llm_bounds_surface_on_settings(self, monkeypatch):
        monkeypatch.delenv("ORA_LLM_TIMEOUT_SECONDS", raising=False)
        monkeypatch.delenv("ORA_LLM_MAX_RETRIES", raising=False)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("llm_timeout_seconds: 42.5\nllm_max_retries: 0\n")
            f.flush()
            config = load_config(f.name)
            assert config.llm_timeout_seconds == 42.5
            assert config.llm_max_retries == 0
            os.unlink(f.name)

    def test_yaml_without_llm_bounds_uses_defaults(self, monkeypatch):
        monkeypatch.delenv("ORA_LLM_TIMEOUT_SECONDS", raising=False)
        monkeypatch.delenv("ORA_LLM_MAX_RETRIES", raising=False)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("limits:\n  max_revisions: 3\n")
            f.flush()
            config = load_config(f.name)
            assert config.llm_timeout_seconds == 300.0
            assert config.llm_max_retries == 2

    def test_yaml_search_block_preserves_env_provider(self, monkeypatch):
        """A partial search: block must not clobber keys it omits.

        Replacing the whole block silently reset an env ORA_SEARCH__PROVIDER
        back to the default, so a block holding only firecrawl_api_url would
        quietly disable Decodo.
        """
        monkeypatch.setenv("ORA_SEARCH__PROVIDER", "decodo")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("search:\n  firecrawl_api_url: http://fc.example:3002\n")
            f.flush()
            config = load_config(f.name)
            assert config.search.provider == "decodo"
            assert config.search.firecrawl_api_url == "http://fc.example:3002"
            os.unlink(f.name)

    def test_yaml_search_provider_overrides_env(self, monkeypatch):
        monkeypatch.setenv("ORA_SEARCH__PROVIDER", "decodo")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("search:\n  provider: firecrawl\n")
            f.flush()
            config = load_config(f.name)
            assert config.search.provider == "firecrawl"
            os.unlink(f.name)

    def test_yaml_without_search_block_keeps_env_provider(self, monkeypatch):
        monkeypatch.setenv("ORA_SEARCH__PROVIDER", "decodo")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("limits:\n  max_revisions: 3\n")
            f.flush()
            config = load_config(f.name)
            assert config.search.provider == "decodo"
            os.unlink(f.name)

    def test_yaml_models_block_preserves_env_default(self, monkeypatch):
        """A partial models: block must not clobber ORA_MODELS__DEFAULT.

        Replacing the whole block discarded the env default whenever the YAML
        defined only a per-role override.
        """
        monkeypatch.setenv("ORA_MODELS__DEFAULT", "env-default-model")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("models:\n  researcher: yaml-researcher-model\n")
            f.flush()
            config = load_config(f.name)
            assert config.models.default == "env-default-model"
            assert config.models.researcher == "yaml-researcher-model"
            os.unlink(f.name)

    def test_yaml_output_block_preserves_env_key(self, monkeypatch):
        monkeypatch.setenv("ORA_OUTPUT__ALWAYS_INCLUDE_SOURCES", "false")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("output:\n  default_format: html\n")
            f.flush()
            config = load_config(f.name)
            assert config.output.always_include_sources is False
            assert config.output.default_format == "html"
            os.unlink(f.name)

    def test_yaml_limits_block_preserves_env_key(self, monkeypatch):
        monkeypatch.setenv("ORA_LIMITS__MAX_REVISIONS", "7")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("limits:\n  default_intensity: 4\n")
            f.flush()
            config = load_config(f.name)
            assert config.limits.max_revisions == 7
            assert config.limits.default_intensity == 4
            os.unlink(f.name)

    def test_yaml_blocks_still_override_env_for_keys_they_define(self, monkeypatch):
        """YAML still wins for the keys it defines."""
        monkeypatch.setenv("ORA_MODELS__DEFAULT", "env-default-model")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("models:\n  default: yaml-default-model\n")
            f.flush()
            config = load_config(f.name)
            assert config.models.default == "yaml-default-model"
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


def test_get_llm_passes_timeout_and_retries(monkeypatch):
    captured = {}

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return object()

    # get_llm imports ChatOpenAI lazily from langchain_openai, so patch there.
    # Isolate load_config so a real ~/.ora/config.yaml cannot leak in.
    monkeypatch.setattr("langchain_openai.ChatOpenAI", fake_chat_openai)
    monkeypatch.setattr("ora.config.load_config", lambda *a, **kw: ORASettings())
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    # Clear ambient ORA_LLM_* so an exported value cannot override the defaults.
    monkeypatch.delenv("ORA_LLM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("ORA_LLM_MAX_RETRIES", raising=False)

    ora.config.get_llm("deepseek/deepseek-v4-flash", temperature=0)

    assert captured["max_retries"] == 2
    # Pin the canonical field name: "timeout" is only an alias, so accepting
    # either would not catch a regression to the wrong kwarg.
    assert captured["request_timeout"] == 300.0


def test_get_llm_honours_configured_timeout_and_retries(monkeypatch):
    captured = {}

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("langchain_openai.ChatOpenAI", fake_chat_openai)
    settings = ORASettings()
    settings.llm_timeout_seconds = 42.5
    settings.llm_max_retries = 0
    monkeypatch.setattr("ora.config.load_config", lambda *a, **kw: settings)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    # Ambient ORA_LLM_* cannot affect this test (attributes are set directly),
    # but clear them so the test reads the same regardless of the environment.
    monkeypatch.delenv("ORA_LLM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("ORA_LLM_MAX_RETRIES", raising=False)

    ora.config.get_llm("deepseek/deepseek-v4-flash", temperature=0)

    assert captured["max_retries"] == 0
    # Pin the canonical field name: "timeout" is only an alias, so accepting
    # either would not catch a regression to the wrong kwarg.
    assert captured["request_timeout"] == 42.5


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


class TestRunModelOverrides:
    """Run-scoped overrides set by CLI flags (ora.config.set_model_override)."""

    @pytest.fixture(autouse=True)
    def _clear_overrides(self):
        from ora.config import clear_model_overrides

        yield
        clear_model_overrides()

    def test_researcher_override_beats_config(self):
        from ora.config import ModelSettings, set_model_override

        settings = ORASettings(models=ModelSettings(researcher="config-model"))
        set_model_override("researcher", "openrouter:qwen/qwen3.7-flash")
        assert get_researcher_model(settings) == "openrouter:qwen/qwen3.7-flash"

    def test_reviewer_override_beats_config(self):
        from ora.config import ModelSettings, set_model_override

        settings = ORASettings(models=ModelSettings(reviewer="config-model"))
        set_model_override("reviewer", "openrouter:deepseek/deepseek-v4-pro")
        assert get_reviewer_model(settings) == "openrouter:deepseek/deepseek-v4-pro"

    def test_clear_restores_config_fallback(self):
        from ora.config import ModelSettings, clear_model_overrides, set_model_override

        settings = ORASettings(models=ModelSettings(default="my-default"))
        set_model_override("researcher", "openrouter:qwen/qwen3.7-flash")
        clear_model_overrides()
        assert get_researcher_model(settings) == "my-default"

    def test_supervisor_unaffected_by_overrides(self):
        from ora.config import ModelSettings, set_model_override

        settings = ORASettings(models=ModelSettings(default="my-default"))
        set_model_override("researcher", "openrouter:qwen/qwen3.7-flash")
        assert get_supervisor_model(settings) == "my-default"


class TestProviderBlockMerge:
    """A partial provider/providers block must not discard env values.

    CONFIG.md promises that a YAML key only overrides its own env var, and this
    holds for every nested block including provider and providers.
    """

    def test_provider_block_preserves_env_default(self, monkeypatch):
        monkeypatch.setenv("ORA_PROVIDER__DEFAULT", "openrouter")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("provider: {}\n")
            f.flush()
            config = load_config(f.name)
            assert config.provider.default == "openrouter"
            os.unlink(f.name)

    def test_yaml_provider_default_overrides_env(self, monkeypatch):
        monkeypatch.setenv("ORA_PROVIDER__DEFAULT", "openrouter")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("provider:\n  default: deepseek\n")
            f.flush()
            config = load_config(f.name)
            assert config.provider.default == "deepseek"
            os.unlink(f.name)

    def test_partial_providers_entry_preserves_env_fields(self, monkeypatch):
        monkeypatch.setenv("ORA_PROVIDERS__OPENROUTER__API_KEY_ENV", "MY_KEY")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("providers:\n  openrouter:\n    base_url: http://x.example/v1\n")
            f.flush()
            config = load_config(f.name)
            assert config.providers["openrouter"].base_url == "http://x.example/v1"
            assert config.providers["openrouter"].api_key_env == "MY_KEY"
            os.unlink(f.name)
