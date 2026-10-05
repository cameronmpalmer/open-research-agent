"""Configuration loading via YAML, env vars, and pydantic-settings."""

import os
from typing import TypeVar

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LimitSettings(BaseModel):
    # At least one writer-reviewer revision cycle: 0 (or a negative value)
    # would disable revisions entirely and is rejected loudly at load rather
    # than silently coerced at the cap sites.
    max_revisions: int = Field(default=3, ge=1)
    default_intensity: int = 2


class SearchSettings(BaseModel):
    provider: str = "firecrawl"
    firecrawl_api_key: str | None = None
    firecrawl_api_url: str = "https://api.firecrawl.com"
    # Decodo SERP API (https://scraper-api.decodo.com/v2/scrape). Credentials are
    # read from process env only, never from config.yaml.
    decodo_api_url: str = "https://scraper-api.decodo.com/v2/scrape"
    # Single Basic-auth credential issued by the Decodo dashboard (API
    # Playground). Username/password are deliberately not supported: the key is
    # the credential Decodo issues, and the account password is a broader
    # secret than the API requires.
    decodo_api_key_env: str = "DECODO_API_KEY"
    decodo_domain: str = "com"
    decodo_locale: str = "en-us"
    # Off by default: Firecrawl search is unreliable and in practice often
    # returns nothing, so silently falling back hides Decodo failures. The
    # fallback is still available, but must be opted into with
    # `search.fallback_to_firecrawl: true` (or ORA_SEARCH__FALLBACK_TO_FIRECRAWL).
    fallback_to_firecrawl: bool = False


class ModelSettings(BaseModel):
    default: str = "deepseek-v4-flash"
    researcher: str | None = None
    supervisor: str | None = None
    writer: str | None = None
    reviewer: str | None = None


class ProviderDefaultSettings(BaseModel):
    default: str = "deepseek"


class ProviderSettings(BaseModel):
    base_url: str | None = None
    api_key_env: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)


class OutputSettings(BaseModel):
    default_format: str = "markdown"
    always_include_sources: bool = True


class ORASettings(BaseSettings):
    """ORA configuration, loaded from env vars with ORA_ prefix."""

    model_config = SettingsConfigDict(
        env_prefix="ORA_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    models: ModelSettings = Field(default_factory=ModelSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)
    limits: LimitSettings = Field(default_factory=LimitSettings)
    provider: ProviderDefaultSettings = Field(default_factory=ProviderDefaultSettings)
    providers: dict[str, ProviderSettings] = Field(default_factory=dict)
    deepseek_base_url: str = "https://api.deepseek.com"  # legacy; kept for backward compat

    # Per-call LLM bounds. Without these the OpenAI SDK defaults apply
    # (read timeout 600s, max_retries=2), so a single stalled call can block
    # a run for ~30 minutes. 300s leaves headroom over the slowest observed
    # legitimate call (155s); with max_retries=2 the worst case for one
    # logical call is about 900s (3 attempts x 300s), not ~30 minutes.
    llm_timeout_seconds: float = 300.0
    llm_max_retries: int = 2


DEFAULT_PROVIDERS: dict[str, ProviderSettings] = {
    "deepseek": ProviderSettings(
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
    ),
    "openrouter": ProviderSettings(
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        headers={
            "HTTP-Referer": "https://github.com/cameronmpalmer/open-research-agent",
            "X-Title": "ORA",
        },
    ),
}


_BlockT = TypeVar("_BlockT", bound=BaseModel)


def _merge_block(base: _BlockT, overrides: dict, model_cls: type[_BlockT]) -> _BlockT:
    """Merge a YAML block over env/defaults field by field.

    Replacing the whole block would silently discard env-provided values for
    keys the YAML omits (e.g. a ``search:`` block with only
    ``firecrawl_api_url`` would reset an env ``ORA_SEARCH__PROVIDER=decodo``
    back to the default). Only the keys actually present in the YAML block
    override, so YAML still wins for the keys it defines.

    A bare ``key:`` in YAML parses to ``None``; treat that as an empty block so
    no call site has to guard it.
    """
    overrides = overrides or {}
    data = base.model_dump()
    data.update({key: value for key, value in overrides.items() if key in data})
    return model_cls(**data)


def _merge_search(base: SearchSettings, overrides: dict) -> SearchSettings:
    """Merge a YAML ``search:`` block over env/defaults field by field."""
    return _merge_block(base, overrides, SearchSettings)


def search_config_error(settings: ORASettings) -> str | None:
    """Return an error message if the configured search backend cannot run.

    Called before any research work starts so a missing search credential fails
    fast with a clear cause, rather than generating a plan and producing a
    report with no sources. Returns None when the backend is usable.
    """
    provider = (settings.search.provider or "firecrawl").lower()
    if provider == "decodo":
        env_name = settings.search.decodo_api_key_env
        if not os.environ.get(env_name, "").strip():
            return (
                f"Search provider 'decodo' requires {env_name} to be set; "
                "set it, or choose another search.provider."
            )
    return None


def load_config(config_path: str | None = None) -> ORASettings:
    """Load ORA configuration from YAML file and environment.

    Priority: YAML file > env vars > defaults (fields present in the YAML
    file override env-var values).
    """
    settings = ORASettings()

    # Try YAML file
    path = config_path or os.path.expanduser("~/.ora/config.yaml")
    if os.path.exists(path):
        with open(path) as f:
            yaml_data = yaml.safe_load(f)
        if yaml_data:
            if "models" in yaml_data:
                settings.models = _merge_block(settings.models, yaml_data["models"], ModelSettings)
            if "search" in yaml_data:
                settings.search = _merge_search(settings.search, yaml_data["search"])
            if "output" in yaml_data:
                settings.output = _merge_block(settings.output, yaml_data["output"], OutputSettings)
            if "limits" in yaml_data:
                settings.limits = _merge_block(settings.limits, yaml_data["limits"], LimitSettings)
            if "provider" in yaml_data:
                settings.provider = _merge_block(
                    settings.provider, yaml_data["provider"], ProviderDefaultSettings
                )
            if "providers" in yaml_data:
                # Merge per provider key so a partial YAML entry (for example
                # only base_url) does not discard an env-provided api_key_env.
                merged_providers = dict(settings.providers)
                for name, cfg in (yaml_data["providers"] or {}).items():
                    base = merged_providers.get(name) or ProviderSettings()
                    merged_providers[name] = _merge_block(base, cfg, ProviderSettings)
                settings.providers = merged_providers
            elif "deepseek_base_url" in yaml_data:
                # Legacy config: no providers map, honor the old flat key.
                settings.providers["deepseek"] = ProviderSettings(
                    base_url=yaml_data["deepseek_base_url"]
                )
            if "deepseek_base_url" in yaml_data:
                settings.deepseek_base_url = yaml_data["deepseek_base_url"]
            if yaml_data.get("llm_timeout_seconds") is not None:
                settings.llm_timeout_seconds = float(yaml_data["llm_timeout_seconds"])
            if yaml_data.get("llm_max_retries") is not None:
                settings.llm_max_retries = int(yaml_data["llm_max_retries"])

    return settings


# Run-scoped model overrides set by CLI flags (--model / --reviewer-model).
# The get_*_model helpers consult these first, so every agent and internal
# helper resolves its model through one path and no call site can bypass the
# configured model. The CLI sets them for a run and clears them afterwards.
_run_model_overrides: dict[str, str] = {}


def set_model_override(role: str, model_name: str) -> None:
    """Override a role's model for the current process run.

    Args:
        role: One of "researcher" or "reviewer" (supervisor has no flag).
        model_name: Full model name, optionally with a provider: prefix.
    """
    _run_model_overrides[role] = model_name


def clear_model_overrides() -> None:
    """Clear all run-scoped model overrides."""
    _run_model_overrides.clear()


def get_researcher_model(settings: ORASettings) -> str:
    """Get the researcher model: run override, else config, else default."""
    return (
        _run_model_overrides.get("researcher")
        or settings.models.researcher
        or settings.models.default
    )


def get_supervisor_model(settings: ORASettings) -> str:
    """Get the supervisor model, falling back to the default model."""
    return settings.models.supervisor or settings.models.default


def get_writer_model(settings: ORASettings) -> str:
    """Get the writer model.

    Resolution: a run override for "writer", else ``models.writer``, else the
    researcher's resolution (the writer has always shared it, so an unset
    writer keeps its previous behaviour), else the default.
    """
    return (
        _run_model_overrides.get("writer")
        or settings.models.writer
        or _run_model_overrides.get("researcher")
        or settings.models.researcher
        or settings.models.default
    )


def get_reviewer_model(settings: ORASettings) -> str:
    """Get the reviewer model: run override, else config, else default."""
    return (
        _run_model_overrides.get("reviewer") or settings.models.reviewer or settings.models.default
    )


def _split_provider(model_name: str) -> tuple[str | None, str]:
    """Split a model name into (provider, model). No colon -> (None, name)."""
    if ":" in model_name:
        provider, _, rest = model_name.partition(":")
        return provider, rest
    return None, model_name


def _resolve_provider(settings: ORASettings, name: str) -> ProviderSettings | None:
    """Effective provider settings: user config merged over built-in defaults.

    Returns None for unknown providers. The legacy deepseek_base_url field
    (YAML or ORA_DEEPSEEK_BASE_URL env) applies when the user has not
    explicitly configured providers.deepseek.
    """
    defaults = DEFAULT_PROVIDERS.get(name)
    if defaults is None and name not in settings.providers:
        return None
    merged = ProviderSettings()
    if defaults is not None:
        merged.base_url = defaults.base_url
        merged.api_key_env = defaults.api_key_env
        merged.headers = dict(defaults.headers)
    user = settings.providers.get(name)
    if user is not None:
        if user.base_url:
            merged.base_url = user.base_url
        if user.api_key_env:
            merged.api_key_env = user.api_key_env
        merged.headers.update(user.headers)
    if name == "deepseek" and name not in settings.providers and settings.deepseek_base_url:
        merged.base_url = settings.deepseek_base_url
    return merged


def get_firecrawl_client():
    """Get a configured FirecrawlApp instance.

    Uses FIRECRAWL_API_KEY env var or config, and api_url from config.
    For self-hosted Firecrawl (no auth), leave api_key empty and set
    FIRECRAWL_API_URL=http://localhost:3002.
    """
    from firecrawl import FirecrawlApp

    settings = load_config()
    api_key = os.environ.get("FIRECRAWL_API_KEY", settings.search.firecrawl_api_key or "")
    api_url = os.environ.get("FIRECRAWL_API_URL", settings.search.firecrawl_api_url)
    return FirecrawlApp(api_key=api_key, api_url=api_url)


def get_llm(model_name: str, temperature: float = 0.0):
    """Get a ChatOpenAI instance routed to the right provider.

    A `provider:model` prefix (e.g. `openrouter:anthropic/claude-3.5-sonnet`)
    selects the provider; without a prefix, the default provider is used.
    An unknown provider prefix warns and falls back to the default provider.
    Raises ValueError if the resolved provider has no API key configured.
    """
    from langchain_openai import ChatOpenAI

    settings = load_config()
    provider_name, clean_name = _split_provider(model_name)

    if provider_name is None:
        provider_name = settings.provider.default or "deepseek"
        clean_name = model_name
    elif _resolve_provider(settings, provider_name) is None:
        import warnings

        warnings.warn(
            f"Unknown provider '{provider_name}' in model name '{model_name}'; "
            f"falling back to default provider '{settings.provider.default or 'deepseek'}'.",
            stacklevel=2,
        )
        provider_name = settings.provider.default or "deepseek"
        clean_name = model_name  # keep the full name; prefix not stripped

    provider = _resolve_provider(settings, provider_name)
    if provider is None:
        raise ValueError(
            f"Unknown default provider '{provider_name}'. Check the 'provider.default' setting."
        )
    if not provider.base_url:
        raise ValueError(
            f"Provider '{provider_name}' has no base_url configured. "
            f"Set providers.{provider_name}.base_url in config.yaml."
        )

    api_key = os.environ.get(provider.api_key_env or "", "")
    if not api_key and provider_name == "deepseek":
        api_key = os.environ.get("OPENAI_API_KEY", "")  # legacy fallback
    if not api_key:
        env_hint = provider.api_key_env or "the provider's api_key_env var"
        raise ValueError(f"No API key for provider '{provider_name}'. Set {env_hint}.")

    llm = ChatOpenAI(
        model=clean_name,
        temperature=temperature,
        base_url=provider.base_url,
        api_key=api_key,
        default_headers=provider.headers or None,
        # request_timeout is the canonical langchain-openai field ("timeout" is
        # its alias); kept as request_timeout for compatibility with older
        # langchain-openai releases.
        request_timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )

    # When a usage collector is active (ora.usage.usage_collection), return a
    # recording proxy so tokens/cost are captured without changing agent
    # call sites. ChatOpenAI is a pydantic model, so we cannot patch it.
    from ora.usage import RecordingLLM, active_collector

    collector = active_collector()
    if collector is not None:
        return RecordingLLM(llm, collector)
    return llm
