"""Run greedy evaluation of a fine-tuned model on PDDL instances.

For each instance:
  1. Generate a plan (greedy decoding, enable_thinking=False)
  2. Clean the raw output to extract plan text
  3. Parse the plan into a ParsedPlan
  4. Write the plan to a temp file and run VAL
  5. Log all fields to the run log

Usage:
    python src/inference/run_greedy_eval.py \\
        --checkpoint-path runs/sft_qwen3_phase1_full \\
        --split heldout \\
        --representations standard anonymized compact \\
        --eval-config configs/eval/greedy_eval.yaml
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Optional

import typer

from inference.generate_plan import load_model_and_tokenizer, generate_plan
from generation.validate_with_val import validate_plan
from pddl_ops.parse_utils import parse_plan_from_text, plan_to_file
from pddl_ops.decode_compact_plan import extract_compact_plan_from_text, decode_compact_plan
from utils.io import load_yaml, load_json, ensure_dir
from utils.logging import get_logger, RunLogger
from utils.seeds import set_global_seed

logger = get_logger(__name__)
app = typer.Typer(add_completion=False)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data"


@app.command()
def main(
    checkpoint_path: str = typer.Option(..., help="Path to fine-tuned checkpoint or HF model id"),
    split: str = typer.Option("heldout", help="'train' or 'heldout'"),
    representations: list[str] = typer.Option(
        ["standard", "anonymized", "compact"], help="Representations to evaluate"
    ),
    eval_config_path: Path = typer.Option(
        _REPO_ROOT / "configs" / "eval" / "greedy_eval.yaml",
        help="Evaluation config YAML",
    ),
    instances_dir: Path = typer.Option(
        _DATA_DIR / "generated" / "instances", help="Directory of generated instances"
    ),
    output_dir: Path = typer.Option(
        _REPO_ROOT / "runs" / "eval_results", help="Directory to write eval results"
    ),
    seed: int = typer.Option(42, help="Global random seed"),
    model_name: str = typer.Option("Qwen/Qwen3-1.7B", help="Model name for logging"),
    run_id: Optional[str] = typer.Option(None, help="Run ID (auto-generated if not set)"),
    domain: Optional[str] = typer.Option(None, help="Filter to a single domain (optional)"),
) -> None:
    """Run greedy evaluation on PDDL instances."""
    set_global_seed(seed)
    run_id = run_id or str(uuid.uuid4())[:8]

    eval_cfg = load_yaml(eval_config_path)
    decoding_cfg = eval_cfg.get("decoding", {})

    # enforce enable_thinking=False at the config level
    decoding_cfg["enable_thinking"] = False

    ensure_dir(output_dir)
    run_logger = RunLogger(output_dir / "run_log.jsonl")
    tmp_dir = ensure_dir(output_dir / "tmp_plans")

    logger.info("Starting greedy eval run_id=%s split=%s", run_id, split)

    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=model_name,
        checkpoint_path=checkpoint_path if Path(checkpoint_path).exists() else None,
    )

    # Discover instances for the split
    instance_files = list(instances_dir.rglob("*_meta.json"))
    instance_files = [
        f for f in instance_files
        if f"_{split}_" in f.name
    ]

    if domain:
        instance_files = [f for f in instance_files if f"_{domain}_" in f.name or f.name.startswith(domain + "_")]
        logger.info("Filtered to domain '%s': %d files", domain, len(instance_files))
    logger.info("Found %d instance meta files for split '%s'", len(instance_files), split)

    for meta_path in sorted(instance_files):
        meta = load_json(meta_path)
        instance_id = meta["instance_id"]
        inst_domain = meta["domain"]

        domain_file = _REPO_ROOT / meta["domain_file"]
        problem_file = _REPO_ROOT / meta["problem_file"]

        if not domain_file.exists() or not problem_file.exists():
            logger.warning("Missing PDDL files for %s, skipping", instance_id)
            continue

        domain_text = domain_file.read_text(encoding="utf-8")
        problem_text = problem_file.read_text(encoding="utf-8")

        for repr_name in representations:
            total_t0 = time.perf_counter()
            error_type: Optional[str] = None
            valid_plan: Optional[bool] = None
            goal_reached: Optional[bool] = None
            generated_tokens: Optional[int] = None
            gen_time: Optional[float] = None
            val_time: Optional[float] = None
            num_actions: int = 0

            try:
                gen_result = generate_plan(
                    domain_text=domain_text,
                    problem_text=problem_text,
                    representation=repr_name,
                    model=model,
                    tokenizer=tokenizer,
                    decoding_config=decoding_cfg,
                    model_name=model_name,
                    checkpoint_path=checkpoint_path,
                )
                generated_tokens = gen_result.generated_tokens
                gen_time = gen_result.generation_time_sec

                # Save raw generation
                raw_out_path = output_dir / "raw" / f"{instance_id}_{repr_name}.txt"
                ensure_dir(raw_out_path.parent)
                raw_out_path.write_text(gen_result.raw_output, encoding="utf-8")

                # Extract and parse plan
                if repr_name == "compact":
                    compact_text = extract_compact_plan_from_text(gen_result.raw_output)
                    parsed = decode_compact_plan(compact_text)
                else:
                    parsed = parse_plan_from_text(gen_result.raw_output)

                num_actions = len(parsed.actions)

                if num_actions == 0:
                    valid_plan = False
                    goal_reached = False
                    logger.warning(
                        "Empty plan for %s/%s (0 actions parsed from %d tokens)",
                        instance_id, repr_name, generated_tokens,
                    )
                else:
                    # Write plan for VAL
                    plan_path = tmp_dir / f"{instance_id}_{repr_name}.pddl"
                    plan_to_file(parsed, plan_path, timed=False)

                    # Validate
                    val_timeout = eval_cfg.get("val_timeout_sec", 30)
                    val_result = validate_plan(
                        problem_id=f"{instance_id}_{repr_name}",
                        domain_file=domain_file,
                        problem_file=problem_file,
                        plan_file=plan_path,
                        output_dir=output_dir / "val_results",
                        timeout=val_timeout,
                    )
                    valid_plan = val_result.parsed_validity
                    goal_reached = val_result.parsed_goal_reached
                    val_time = val_result.wall_clock_sec

                    # Treat unparseable VAL output as invalid
                    if valid_plan is None:
                        valid_plan = False
                    if goal_reached is None and valid_plan is False:
                        goal_reached = False

            except Exception as exc:
                error_type = type(exc).__name__
                logger.warning("Error on %s/%s: %s", instance_id, repr_name, exc)

            total_time = time.perf_counter() - total_t0

            run_logger.log_run_result(
                run_id=run_id,
                seed=seed,
                domain=inst_domain,
                problem_id=instance_id,
                representation=repr_name,
                split=split,
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                planner_backend="none",
                valid_plan=valid_plan,
                goal_reached=goal_reached,
                max_new_tokens=decoding_cfg.get("max_new_tokens", 2048),
                generated_tokens=generated_tokens,
                generation_time_sec=gen_time,
                val_time_sec=val_time,
                total_time_sec=total_time,
                error_type=error_type,
                num_actions=num_actions,
            )

    logger.info("Eval complete. Results in %s", output_dir)


if __name__ == "__main__":
    app()
