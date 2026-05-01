#!/usr/bin/env python3
"""GRPO training using VAL validation as reward function.

Uses TRL (Transformer Reinforcement Learning) for policy optimization.
Reward: VAL validation result (0 for invalid, 1 for valid, 10 for goal-reached).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from generation.validate_with_val import validate_plan
from pddl_ops.decode_compact_plan import decode_compact_plan
from utils.logging import get_logger

logger = get_logger(__name__)


def compute_plan_reward(
    problem_id: str,
    domain_file: Path,
    problem_file: Path,
    generated_plan: str,
) -> tuple[float, dict[str, Any]]:
    """Compute reward by validating generated plan with VAL.

    Returns:
        reward: float in [0, 10]
        info: dict with validation details
    """
    # Parse actions from generated text
    if generated_plan.strip().startswith("("):
        # Standard PDDL format
        actions = [line.strip() for line in generated_plan.strip().split("\n") if line.strip()]
    else:
        # Compact format
        try:
            parsed_plan = decode_compact_plan(generated_plan)
            actions = parsed_plan.to_pddl_lines()
        except Exception:
            actions = []

    if not actions:
        return 0.0, {"valid": False, "goal": False, "reason": "empty_plan"}

    # Write temporary plan file
    temp_plan = Path(f"/tmp/grpo_plan_{problem_id}.pddl")
    temp_plan.write_text("\n".join(actions))

    # Validate with VAL
    try:
        result = validate_plan(
            problem_id=problem_id,
            domain_file=domain_file,
            problem_file=problem_file,
            plan_file=temp_plan,
            output_dir=Path("/tmp"),
            timeout=10,
        )

        # Compute reward
        if result.parsed_goal_reached:
            reward = 10.0
        elif result.parsed_validity:
            reward = 1.0
        else:
            reward = 0.0

        info = {
            "valid": result.parsed_validity or False,
            "goal": result.parsed_goal_reached or False,
            "num_actions": len(actions),
            "exit_code": result.exit_code,
        }

        return reward, info

    except Exception as e:
        logger.warning(f"Reward computation failed for {problem_id}: {e}")
        return 0.0, {"valid": False, "goal": False, "reason": "error"}
    finally:
        temp_plan.unlink(missing_ok=True)


def load_training_data() -> Dataset:
    """Load PDDL problems for GRPO training."""
    # Load tuples (contain domain, problem, reference plan)
    tuples_dir = _REPO_ROOT / "data/generated/tuples_standard"
    tuples = []

    for tuple_file in tuples_dir.glob("*_train_*_tuple.json"):
        data = json.loads(tuple_file.read_text())

        # Create prompt (same format as SFT)
        instruction = "Generate a valid PDDL plan to solve this planning problem."
        problem_text = data["problem_text"]
        prompt = f"{instruction}\n\n{problem_text}\n\nPlan:"

        tuples.append({
            "instance_id": data["instance_id"],
            "prompt": prompt,
            "domain_text": data["domain_text"],
            "problem_text": data["problem_text"],
            # Store paths for VAL validation
            "domain_file": str(_REPO_ROOT / "data/generated/instances" /
                             f"{data['instance_id'].split('_')[0]}/train/{data['instance_id']}_domain.pddl"),
            "problem_file": str(_REPO_ROOT / "data/generated/instances" /
                              f"{data['instance_id'].split('_')[0]}/train/{data['instance_id']}_problem.pddl"),
        })

    logger.info(f"Loaded {len(tuples)} training problems")
    return Dataset.from_list(tuples)


def main():
    model_path = _REPO_ROOT / "runs/qwen3_mini_direct"
    output_dir = _REPO_ROOT / "runs/qwen3_grpo"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SFT checkpoint as base policy
    logger.info(f"Loading SFT model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        padding_side="left",  # Important for generation
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # PPO config
    ppo_config = PPOConfig(
        learning_rate=1e-6,
        batch_size=4,
        mini_batch_size=1,
        gradient_accumulation_steps=4,
        ppo_epochs=4,
        optimize_cuda_cache=True,
        early_stopping=False,
        target_kl=0.1,
        seed=42,
    )

    # Load training data
    dataset = load_training_data()

    # Tokenize prompts
    def tokenize_fn(examples):
        return tokenizer(
            examples["prompt"],
            truncation=True,
            max_length=2048,
            padding=False,
        )

    dataset = dataset.map(tokenize_fn, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # PPO Trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
    )

    # Training loop
    generation_kwargs = {
        "max_new_tokens": 512,
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 0.7,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    logger.info("Starting GRPO training...")

    for epoch in range(3):
        logger.info(f"Epoch {epoch + 1}/3")

        for batch_idx, batch in enumerate(ppo_trainer.dataloader):
            # Get prompts for this batch
            batch_instances = [dataset[i] for i in range(batch_idx * ppo_config.batch_size,
                                                         min((batch_idx + 1) * ppo_config.batch_size, len(dataset)))]

            # Generate plans
            query_tensors = batch["input_ids"]
            response_tensors = ppo_trainer.generate(
                query_tensors,
                return_prompt=False,
                **generation_kwargs,
            )

            # Decode generations
            batch_texts = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

            # Compute rewards using VAL
            rewards = []
            stats = []

            for instance, generated_text in zip(batch_instances, batch_texts):
                reward, info = compute_plan_reward(
                    problem_id=instance["instance_id"],
                    domain_file=Path(instance["domain_file"]),
                    problem_file=Path(instance["problem_file"]),
                    generated_plan=generated_text,
                )
                rewards.append(torch.tensor(reward))
                stats.append(info)

            # PPO update
            stats_dict = ppo_trainer.step(query_tensors, response_tensors, rewards)

            # Log stats
            if batch_idx % 10 == 0:
                avg_reward = sum(r.item() for r in rewards) / len(rewards)
                valid_rate = sum(s["valid"] for s in stats) / len(stats)
                goal_rate = sum(s["goal"] for s in stats) / len(stats)

                logger.info(
                    f"Batch {batch_idx}: avg_reward={avg_reward:.3f} "
                    f"valid={valid_rate:.2%} goal={goal_rate:.2%}"
                )

        # Save checkpoint after each epoch
        checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch + 1}"
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        logger.info(f"Saved checkpoint to {checkpoint_dir}")

    # Save final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Training complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
