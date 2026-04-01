"""Write LLaMA-Factory training YAML configs programmatically.

This ensures configs are always generated from canonical values,
preventing drift between the locked spec and the actual training run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.io import dump_yaml, load_yaml
from utils.logging import get_logger

logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS_DIR = _REPO_ROOT / "configs" / "train"
_RUNS_DIR = _REPO_ROOT / "runs"


def _base_config(
    model_name_or_path: str,
    finetuning_type: str,
    dataset_names: list[str],
    output_dir: Path,
    **overrides: Any,
) -> dict[str, Any]:
    """Return the canonical base config dict."""
    cfg: dict[str, Any] = {
        # Model
        "model_name_or_path": model_name_or_path,
        "trust_remote_code": True,
        # Method
        "stage": "sft",
        "do_train": True,
        "finetuning_type": finetuning_type,
        # Dataset
        "dataset": ",".join(dataset_names),
        "dataset_dir": "data/datasets",
        "template": "qwen",
        "cutoff_len": 4096,
        "overwrite_cache": False,
        "preprocessing_num_workers": 4,
        # Output
        "output_dir": str(output_dir),
        "logging_dir": str(output_dir / "logs"),
        "overwrite_output_dir": False,
        "save_steps": 500,
        "logging_steps": 10,
        "plot_loss": True,
        # Training
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 1.0e-5,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "num_train_epochs": 3.0,
        "max_steps": -1,
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "ddp_timeout": 180000,
        "dataloader_num_workers": 2,
        # Reproducibility
        "seed": 42,
        "data_seed": 42,
    }
    cfg.update(overrides)
    return cfg


def write_full_config(
    output_path: Path | None = None,
    model_name_or_path: str = "Qwen/Qwen3-1.7B",
    dataset_names: list[str] | None = None,
    run_name: str = "sft_qwen3_phase1_full",
    **overrides: Any,
) -> Path:
    """Write the reproduction (full fine-tuning) config.

    Args:
        output_path: Where to write the YAML. Defaults to runs/<run_name>/train_config.yaml.
        model_name_or_path: HF model identifier.
        dataset_names: LLaMA-Factory dataset names to train on.
        run_name: Used to derive output_dir and default output_path.
        **overrides: Any key-value pairs to override in the base config.

    Returns:
        Path to the written YAML file.
    """
    dataset_names = dataset_names or ["phase1_standard", "phase1_anonymized", "phase1_compact"]
    output_dir = _RUNS_DIR / run_name

    cfg = _base_config(
        model_name_or_path=model_name_or_path,
        finetuning_type="full",
        dataset_names=dataset_names,
        output_dir=output_dir,
        **overrides,
    )

    if output_path is None:
        output_path = output_dir / "train_config.yaml"

    dump_yaml(cfg, output_path)
    logger.info("Wrote full SFT config -> %s", output_path)
    return output_path


def write_lora_debug_config(
    output_path: Path | None = None,
    model_name_or_path: str = "Qwen/Qwen3-1.7B",
    dataset_names: list[str] | None = None,
    run_name: str = "sft_qwen3_phase1_lora_debug",
    **overrides: Any,
) -> Path:
    """Write the LoRA smoke-test config.

    Same as full config but with:
    - finetuning_type: lora
    - max_steps: 20 (tiny smoke run)
    - cutoff_len: 1024
    - gradient_checkpointing: false
    - overwrite_output_dir: true
    """
    dataset_names = dataset_names or ["phase1_standard"]
    output_dir = _RUNS_DIR / run_name

    lora_extras: dict[str, Any] = {
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target": "all",
        "cutoff_len": 1024,
        "overwrite_output_dir": True,
        "save_steps": 50,
        "logging_steps": 5,
        "plot_loss": False,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1.0e-4,
        "num_train_epochs": 1.0,
        "max_steps": 20,
        "gradient_checkpointing": False,
    }
    lora_extras.update(overrides)

    cfg = _base_config(
        model_name_or_path=model_name_or_path,
        finetuning_type="lora",
        dataset_names=dataset_names,
        output_dir=output_dir,
        **lora_extras,
    )

    if output_path is None:
        output_path = output_dir / "train_config.yaml"

    dump_yaml(cfg, output_path)
    logger.info("Wrote LoRA debug SFT config -> %s", output_path)
    return output_path
