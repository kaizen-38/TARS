from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ToolConfig(BaseModel):
    fast_downward: str | None = None
    val: str | None = None
    python: str = "python"


class DataConfig(BaseModel):
    repo_root: Path = Path(".")
    domains_root: Path | None = None
    generated_plans_root: Path | None = None
    generations_root: Path | None = None
    splits_file: Path | None = None
    output_root: Path = Path("results")


class ExperimentConfig(BaseModel):
    train_domains: list[str] = Field(default_factory=lambda: [
        "blocksworld", "gripper", "ferry", "delivery",
        "childsnack", "floortile", "rovers", "spanner",
    ])
    heldout_domains: list[str] = Field(default_factory=lambda: [
        "miconic", "sokoban", "transport", "satellite",
    ])
    representations: list[str] = Field(
        default_factory=lambda: ["standard", "anonymized", "compact"]
    )
    model_ids: list[str] = Field(
        default_factory=lambda: ["base", "sft514", "sft1558"]
    )
    random_seed: int = 574
    max_seconds_per_instance: int = 30
    max_steps_factor: float = 3.0
    max_steps_cap: int = 200
    val_call_cap: int = 100


class PivotConfig(BaseModel):
    tools: ToolConfig = Field(default_factory=ToolConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)


def load_config(path: Path) -> PivotConfig:
    if not path.exists():
        return PivotConfig()
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    return PivotConfig(**raw)


def save_config(config: PivotConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)
