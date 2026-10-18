"""Versioned API contracts and training configuration validation."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RunCreateRequest(BaseModel):
    """Public contract shared by beginner and advanced training workflows."""

    model_config = ConfigDict(extra="forbid")

    agent: str = Field(min_length=1, max_length=80)
    env: str = Field(min_length=1, max_length=120)
    name: Optional[str] = Field(default=None, max_length=160)
    description: str = Field(default="", max_length=4000)
    tags: list[str] = Field(default_factory=list, max_length=30)
    params: dict[str, Any] = Field(default_factory=dict)
    episodes: int = Field(default=500, ge=1, le=10_000_000)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    eval_freq: int = Field(default=25, ge=1)
    eval_episodes: int = Field(default=10, ge=1, le=10_000)
    checkpoint_freq: int = Field(default=100, ge=1)
    max_steps: Optional[int] = Field(default=None, ge=1, le=10_000_000)
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    save_best: bool = True
    save_final: bool = True
    auto_eval: bool = True
    early_stopping_patience: Optional[int] = Field(default=None, ge=1)
    log_tensorboard: bool = True
    log_wandb: bool = False
    preset: str = Field(default="custom", max_length=80)
    mode: Literal["beginner", "advanced"] = "beginner"
    # When given, one run is created per seed and they share a group id.
    # ``seed`` is ignored in that case.
    seeds: Optional[list[int]] = Field(default=None, min_length=1, max_length=20)

    @field_validator("seeds")
    @classmethod
    def unique_seeds(cls, seeds):
        if seeds is None:
            return seeds
        for s in seeds:
            if s < 0 or s > 2**32 - 1:
                raise ValueError("seeds must be between 0 and 2^32-1")
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be unique")
        return seeds

    @field_validator("tags")
    @classmethod
    def clean_tags(cls, tags: list[str]) -> list[str]:
        cleaned = []
        for tag in tags:
            value = tag.strip().lower()
            if value and value not in cleaned:
                cleaned.append(value[:48])
        return cleaned


class ModelUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = Field(default=None, min_length=1, max_length=160)
    description: Optional[str] = Field(default=None, max_length=4000)
    tags: Optional[list[str]] = Field(default=None, max_length=30)
    is_favorite: Optional[bool] = None
    is_archived: Optional[bool] = None


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    episodes: int = Field(default=20, ge=1, le=10_000)


class RunUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = Field(default=None, min_length=1, max_length=160)
    notes: Optional[str] = Field(default=None, max_length=4000)
    tags: Optional[list[str]] = Field(default=None, max_length=30)

    @field_validator("tags")
    @classmethod
    def clean_tags(cls, tags):
        if tags is None:
            return tags
        return RunCreateRequest.clean_tags(tags)
