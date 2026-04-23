"""Config validation with Pydantic. Validates YAML configs before the pipeline runs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator


VALID_CHECK_TYPES = {"raw-qc"}
VALID_AGGREGATION_TYPES = {"mean"}


class DerivativeConfig(BaseModel):
    name: str
    path: Path
    bids: bool
    proc: str | None = None  # e.g. "filt", "clean" — required for MBP-style derivatives

    @field_validator("path")
    @classmethod
    def validate_path_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Path does not exist: {v}")
        return v

    model_config = {"populate_by_name": True}


class PipelineConfig(BaseModel):
    check_type: str       # global metric type — applies to all derivatives
    aggregation_type: str  # how to aggregate per-file metrics into one row per derivative
    derivatives: list[DerivativeConfig]

    @field_validator("check_type")
    @classmethod
    def validate_check_type(cls, v: str) -> str:
        if v not in VALID_CHECK_TYPES:
            raise ValueError(
                f"Unknown check-type '{v}'. Valid values: {sorted(VALID_CHECK_TYPES)}"
            )
        return v

    @field_validator("aggregation_type")
    @classmethod
    def validate_aggregation_type(cls, v: str) -> str:
        if v not in VALID_AGGREGATION_TYPES:
            raise ValueError(
                f"Unknown aggregation-type '{v}'. Valid values: {sorted(VALID_AGGREGATION_TYPES)}"
            )
        return v

    @model_validator(mode="after")
    def at_least_two_derivatives(self) -> "PipelineConfig":
        if len(self.derivatives) < 2:
            raise ValueError("Config must list at least 2 derivatives to compare.")
        names = [d.name for d in self.derivatives]
        duplicates = [n for n in names if names.count(n) > 1]
        if duplicates:
            raise ValueError(f"Duplicate derivative names: {list(set(duplicates))}")
        return self


def load_config(config_path: str | Path) -> PipelineConfig:
    """Load and validate a YAML pipeline config. Raises on invalid input."""
    import yaml

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("Config file must be a YAML mapping.")

    # Normalize hyphenated keys → underscored for Pydantic
    for hyphen_key, under_key in [("check-type", "check_type"), ("aggregation-type", "aggregation_type")]:
        if hyphen_key in raw:
            raw[under_key] = raw.pop(hyphen_key)

    return PipelineConfig.model_validate(raw)
