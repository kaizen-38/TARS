"""Launch SFT training via LLaMA-Factory.

Generates the training config YAML (if not already present) and calls
LLaMA-Factory's `llamafactory-cli train` entry point.

Usage:
    python src/training/launch_sft.py --mode full
    python src/training/launch_sft.py --mode lora_debug
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

from training.write_llamafactory_yaml import write_full_config, write_lora_debug_config
from utils.logging import get_logger

logger = get_logger(__name__)
app = typer.Typer(add_completion=False)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LF_ROOT = _REPO_ROOT / "third_party" / "LLaMAFactory"


def _find_llamafactory_cli() -> list[str]:
    """Locate the LLaMA-Factory CLI entry point."""
    # Prefer installed CLI
    try:
        result = subprocess.run(
            ["llamafactory-cli", "--help"], capture_output=True, timeout=30
        )
        if result.returncode == 0:
            return ["llamafactory-cli"]
    except FileNotFoundError:
        pass

    # Fall back to running from the submodule
    src_script = _LF_ROOT / "src" / "llamafactory" / "launcher.py"
    if src_script.exists():
        return [sys.executable, str(src_script)]

    main_script = _LF_ROOT / "src" / "train.py"
    if main_script.exists():
        return [sys.executable, str(main_script)]

    raise RuntimeError(
        "LLaMA-Factory CLI not found. Run `pip install -e third_party/LLaMAFactory` "
        "or `make setup`."
    )


@app.command()
def main(
    mode: str = typer.Option("lora_debug", help="'full' or 'lora_debug'"),
    model: str = typer.Option("Qwen/Qwen3-1.7B", help="HF model identifier"),
    dry_run: bool = typer.Option(False, help="Print command without executing"),
) -> None:
    """Launch SFT training via LLaMA-Factory."""

    if mode == "full":
        config_path = write_full_config(model_name_or_path=model)
    elif mode == "lora_debug":
        config_path = write_lora_debug_config(model_name_or_path=model)
    else:
        typer.echo(f"Unknown mode '{mode}'. Choose 'full' or 'lora_debug'.", err=True)
        raise typer.Exit(1)

    lf_cmd = _find_llamafactory_cli()
    cmd = lf_cmd + ["train", str(config_path)]

    logger.info("Launch SFT: %s", " ".join(cmd))
    if dry_run:
        typer.echo("DRY RUN: " + " ".join(cmd))
        return

    # Run from repo root so relative paths in config resolve correctly
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    if result.returncode != 0:
        logger.error("Training failed with exit code %d", result.returncode)
        raise typer.Exit(result.returncode)

    logger.info("Training complete. Outputs in %s", config_path.parent)


if __name__ == "__main__":
    app()
