"""Safe, validated YAML configuration. No executable imports or credentials in the registry."""

from pathlib import Path
from typing import Self

import yaml
from harness_model_router import ModelProfile, Price
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    defaults: dict[str, str]
    prices: dict[str, Price]
    profiles: list[ModelProfile]
    bindings: dict[str, str] = Field(default_factory=dict)  # profile alias -> optional .env setting
    parser_kwargs: dict[str, dict[str, object]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_profiles(self) -> Self:
        ids = [p.id for p in self.profiles]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate profile aliases")
        if not set(self.bindings) <= set(ids):
            raise ValueError("Binding references an unknown profile")
        return self


def load_registry(path: Path) -> ModelRegistry:
    with path.open() as source:
        return ModelRegistry.model_validate(yaml.safe_load(source))
