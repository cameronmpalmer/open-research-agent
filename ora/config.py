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
    default: str = "openai:gpt-4.1-mini"
    researcher: Optional[str] = None
    reviewer: Optional[str] = None


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

    return settings


def get_researcher_model(settings: ORASettings) -> str:
    """Get the researcher model, falling back to default."""
    return settings.models.researcher or settings.models.default


def get_reviewer_model(settings: ORASettings) -> str:
    """Get the reviewer model, falling back with cross-model logic."""
    if settings.models.reviewer:
        return settings.models.reviewer
    # Default: pick opposite provider from researcher
    researcher = get_researcher_model(settings)
    if researcher.startswith("openai:"):
        return "anthropic:claude-sonnet-4-20250514"
    return "openai:gpt-4.1"
