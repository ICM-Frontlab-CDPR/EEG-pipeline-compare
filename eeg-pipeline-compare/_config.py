"""Config validation with Pydantic. Validates YAML configs before the pipeline runs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator


VALID_CHECK_TYPES = {"raw-qc"}


class DerivativeConfig(BaseModel):
    name: str
    path: Path
    bids: bool
    check_type: str  # mapped from "check-type" in YAML
    proc: str | None = None  # e.g. "filt", "clean" — required for MBP-style derivatives

    @field_validator("check_type")
    @classmethod
    def validate_check_type(cls, v: str) -> str:
        if v not in VALID_CHECK_TYPES:
            raise ValueError(
                f"Unknown check-type '{v}'. Valid values: {sorted(VALID_CHECK_TYPES)}"
            )
        return v

    @field_validator("path")
    @classmethod
    def validate_path_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Path does not exist: {v}")
        return v

    model_config = {"populate_by_name": True}


class PipelineConfig(BaseModel):
    derivatives: list[DerivativeConfig]

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

    # Normalize "check-type" → "check_type" for Pydantic
    for deriv in raw.get("derivatives", []):
        if "check-type" in deriv:
            deriv["check_type"] = deriv.pop("check-type")

    return PipelineConfig.model_validate(raw)
