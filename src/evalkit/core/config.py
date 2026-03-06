"""Configuration management for evalkit.

Supports loading from YAML files, environment variables, and direct construction.
All secrets (API keys) are read exclusively from environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from evalkit.core.models import VotingStrategy


class LLMProviderConfig(BaseModel):
    """Configuration for an LLM provider (OpenAI, Anthropic, etc.).

    Attributes:
        provider: Provider name ("openai" or "anthropic").
        model: Model identifier (e.g. "gpt-4o", "claude-sonnet-4-20250514").
        api_key_env_var: Name of the env var holding the API key.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the response.
        timeout_seconds: Request timeout.
    """

    model_config = ConfigDict(frozen=True)

    provider: str = Field(default="openai")
    model: str = Field(default="gpt-4o")
    api_key_env_var: str = Field(default="OPENAI_API_KEY")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1)
    timeout_seconds: int = Field(default=60, ge=1)

    @property
    def api_key(self) -> str:
        """Retrieve the API key from the environment.

        Raises:
            EnvironmentError: If the env var is not set.
        """
        key = os.environ.get(self.api_key_env_var)
        if not key:
            raise OSError(
                f"API key not found. Set the {self.api_key_env_var} environment variable."
            )
        return key


class JudgeConfig(BaseModel):
    """Configuration for a single judge instance.

    Attributes:
        judge_id: Unique identifier for the judge.
        judge_type: Type of judge ("llm", "ensemble").
        llm: LLM provider config (required for llm type).
        rubric_name: Name of the rubric to use.
        weight: Weight when used inside an ensemble.
    """

    model_config = ConfigDict(frozen=True)

    judge_id: str
    judge_type: str = Field(default="llm")
    llm: LLMProviderConfig = Field(default_factory=LLMProviderConfig)
    rubric_name: str = Field(default="default")
    weight: float = Field(default=1.0, ge=0.0)


class EnsembleConfig(BaseModel):
    """Configuration for an ensemble of judges.

    Attributes:
        judges: List of judge configurations.
        voting_strategy: How to aggregate judge verdicts.
        regression_threshold: Delta below which a score change is a regression.
    """

    model_config = ConfigDict(frozen=True)

    judges: list[JudgeConfig] = Field(default_factory=list)
    voting_strategy: VotingStrategy = Field(default=VotingStrategy.WEIGHTED_AVERAGE)
    regression_threshold: float = Field(default=-0.1)


class StorageConfig(BaseModel):
    """Configuration for the DuckDB storage backend.

    Attributes:
        database_path: File path for the DuckDB database. Use ":memory:" for in-memory.
    """

    model_config = ConfigDict(frozen=True)

    database_path: str = Field(default=":memory:")


class EvalConfig(BaseModel):
    """Top-level configuration for an evalkit session.

    Attributes:
        project_name: Human-readable project name.
        ensemble: Ensemble configuration.
        storage: Storage backend configuration.
        default_model_id: Default model identifier when not specified per-eval.
    """

    model_config = ConfigDict(frozen=True)

    project_name: str = Field(default="evalkit")
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    default_model_id: str = Field(default="default-model")

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Parsed EvalConfig instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            ValueError: If the YAML content is invalid.
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            raw: Any = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ValueError(f"Expected a YAML mapping, got {type(raw).__name__}")

        return cls.model_validate(raw)

    def to_yaml(self, path: str | Path) -> None:
        """Write configuration to a YAML file.

        Args:
            path: Destination file path.
        """
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = self.model_dump(mode="json")
        with open(config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
