"""Configuration loading via YAML, env vars, and pydantic-settings."""
import os
from typing import Optional
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LimitSettings(BaseModel):
    max_revisions: int = 3
    default_intensity: int = 2


class SearchSettings(BaseModel):
    provider: str = "firecrawl"
    firecrawl_api_key: Optional[str] = None


class ModelSettings(BaseModel):
    default: str = "deepseek-chat"
    researcher: Optional[str] = None
    reviewer: Optional[str] = "deepseek-reasoner"


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
    deepseek_base_url: str = "https://api.deepseek.com"


def load_config(config_path: Optional[str] = None) -> ORASettings:
    """Load ORA configuration from YAML file and environment.

    Priority: env vars > YAML file > defaults.
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
            if "deepseek_base_url" in yaml_data:
                settings.deepseek_base_url = yaml_data["deepseek_base_url"]

    return settings


def get_researcher_model(settings: ORASettings) -> str:
    """Get the researcher model, falling back to default."""
    return settings.models.researcher or settings.models.default


def get_reviewer_model(settings: ORASettings) -> str:
    """Get the reviewer model, falling back to deepseek-reasoner."""
    return settings.models.reviewer or "deepseek-reasoner"


def get_llm(model_name: str, temperature: float = 0.0):
    """Get a DeepSeek-configured ChatOpenAI instance.

    Strips any 'provider:' prefix from model_name for backward compatibility.
    Uses DEEPSEEK_API_KEY with fallback to OPENAI_API_KEY.
    """
    from langchain_openai import ChatOpenAI

    settings = load_config()
    # Strip legacy provider: prefix (e.g. "openai:gpt-4.1-mini" -> "gpt-4.1-mini")
    clean_name = model_name.split(":", 1)[-1] if ":" in model_name else model_name
    return ChatOpenAI(
        model=clean_name,
        temperature=temperature,
        base_url=settings.deepseek_base_url,
        api_key=os.environ.get("DEEPSEEK_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
    )
