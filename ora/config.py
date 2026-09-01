"""Configuration loading via YAML, env vars, and pydantic-settings."""

import os

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LimitSettings(BaseModel):
    max_revisions: int = 3
    default_intensity: int = 2


class SearchSettings(BaseModel):
    provider: str = "firecrawl"
    firecrawl_api_key: str | None = None
    firecrawl_api_url: str = "https://api.firecrawl.com"


class ModelSettings(BaseModel):
    default: str = "deepseek-v4-flash"
    researcher: str | None = None
    supervisor: str | None = None
    reviewer: str | None = "deepseek-v4-pro"


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
                settings.models = ModelSettings(**yaml_data["models"])
            if "search" in yaml_data:
                settings.search = SearchSettings(**yaml_data["search"])
            if "output" in yaml_data:
                settings.output = OutputSettings(**yaml_data["output"])
            if "limits" in yaml_data:
                settings.limits = LimitSettings(**yaml_data["limits"])
            if "provider" in yaml_data:
                settings.provider = ProviderDefaultSettings(**yaml_data["provider"])
            if "providers" in yaml_data:
                settings.providers = {
                    name: ProviderSettings(**cfg)
                    for name, cfg in yaml_data["providers"].items()
                }
            elif "deepseek_base_url" in yaml_data:
                # Legacy config: no providers map, honor the old flat key.
                settings.providers["deepseek"] = ProviderSettings(
                    base_url=yaml_data["deepseek_base_url"]
                )
            if "deepseek_base_url" in yaml_data:
                settings.deepseek_base_url = yaml_data["deepseek_base_url"]

    return settings


def get_researcher_model(settings: ORASettings) -> str:
    """Get the researcher model, falling back to default."""
    return settings.models.researcher or settings.models.default


def get_supervisor_model(settings: ORASettings) -> str:
    """Get the supervisor model, falling back to deepseek-v4-pro."""
    return settings.models.supervisor or "deepseek-v4-pro"


def get_reviewer_model(settings: ORASettings) -> str:
    """Get the reviewer model, falling back to deepseek-v4-pro."""
    return settings.models.reviewer or "deepseek-v4-pro"


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

    api_key = os.environ.get(provider.api_key_env or "", "")
    if not api_key and provider_name == "deepseek":
        api_key = os.environ.get("OPENAI_API_KEY", "")  # legacy fallback
    if not api_key:
        env_hint = provider.api_key_env or "the provider's api_key_env var"
        raise ValueError(f"No API key for provider '{provider_name}'. Set {env_hint}.")

    return ChatOpenAI(
        model=clean_name,
        temperature=temperature,
        base_url=provider.base_url,
        api_key=api_key,
        default_headers=provider.headers or None,
    )
