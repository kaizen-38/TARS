#!/usr/bin/env python3
"""GRPO training using VAL validation as reward - TRL 0.24.0 compatible.

This uses TRL's PPOTrainer with external reward computation via VAL.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    """Compute reward by validating generated plan with VAL."""
    # Parse actions from generated text
    if generated_plan.strip().startswith("("):
        actions = [line.strip() for line in generated_plan.strip().split("\n") if line.strip()]
    else:
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
        }

        return reward, info

    except Exception as e:
        logger.warning(f"Reward computation failed for {problem_id}: {e}")
        return 0.0, {"valid": False, "goal": False, "reason": "error"}
    finally:
        temp_plan.unlink(missing_ok=True)


def load_training_data() -> tuple[Dataset, dict]:
    """Load PDDL problems and metadata for GRPO training."""
    tuples_dir = _REPO_ROOT / "data/generated/tuples_standard"
    data_list = []
    metadata = {}

    for tuple_file in tuples_dir.glob("*_train_*_tuple.json"):
        data = json.loads(tuple_file.read_text())
        instance_id = data["instance_id"]

        # Create prompt
        instruction = "Generate a valid PDDL plan to solve this planning problem."
        problem_text = data["problem_text"]
        prompt = f"{instruction}\n\n{problem_text}\n\nPlan:"

        data_list.append({
            "query": prompt,  # TRL expects "query" field
            "instance_id": instance_id,
        })

        # Store metadata for reward computation
        domain_name = instance_id.split("_")[0]
        metadata[instance_id] = {
            "domain_file": str(_REPO_ROOT / "data/generated/instances" / domain_name / "train" / f"{instance_id}_domain.pddl"),
            "problem_file": str(_REPO_ROOT / "data/generated/instances" / domain_name / "train" / f"{instance_id}_problem.pddl"),
        }

    logger.info(f"Loaded {len(data_list)} training problems")
    return Dataset.from_list(data_list), metadata


def main():
    model_path = _REPO_ROOT / "runs/qwen3_mini_direct"
    output_dir = _REPO_ROOT / "runs/qwen3_grpo"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading SFT model from {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model for policy
    base_model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Wrap with value head for PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)

    # Copy generation_config from base model (required by PPOTrainer)
    if hasattr(base_model, 'generation_config'):
        model.generation_config = base_model.generation_config

    # Create reference model (frozen copy for KL divergence)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # PPO config
    ppo_config = PPOConfig(
        output_dir=str(output_dir),
        learning_rate=1e-6,
        batch_size=4,
        mini_batch_size=1,
        gradient_accumulation_steps=4,
        num_ppo_epochs=4,
        response_length=512,
        temperature=0.7,
        kl_coef=0.05,
        cliprange=0.2,
        vf_coef=0.1,
        gamma=1.0,
        lam=0.95,
        seed=42,
        report_to="none",
    )

    # Load training data
    dataset, metadata = load_training_data()

    # PPO Trainer - TRL 0.24.0 API
    ppo_trainer = PPOTrainer(
        args=ppo_config,  # Changed from config to args
        processing_class=tokenizer,  # Changed from tokenizer
        model=model,
        ref_model=ref_model,
        reward_model=None,  # We compute rewards externally
        train_dataset=dataset,
        value_model=None,  # Value head is in model
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
            # Get queries (prompts)
            query_tensors = [torch.tensor(ex["input_ids"]) for ex in batch]

            # Generate responses
            response_tensors = []
            for query in query_tensors:
                response = ppo_trainer.generate(
                    query.unsqueeze(0),
                    **generation_kwargs,
                )
                response_tensors.append(response.squeeze())

            # Decode generations
            generated_texts = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

            # Compute rewards using VAL
            rewards = []
            stats = []

            for i, (generated_text, batch_item) in enumerate(zip(generated_texts, batch)):
                instance_id = batch_item["instance_id"]
                meta = metadata[instance_id]

                reward, info = compute_plan_reward(
                    problem_id=instance_id,
                    domain_file=Path(meta["domain_file"]),
                    problem_file=Path(meta["problem_file"]),
                    generated_plan=generated_text,
                )
                rewards.append(torch.tensor(reward))
                stats.append(info)

            # PPO update
            train_stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

            # Log stats
            if batch_idx % 10 == 0:
                avg_reward = sum(r.item() for r in rewards) / len(rewards)
                valid_rate = sum(s["valid"] for s in stats) / len(stats)
                goal_rate = sum(s["goal"] for s in stats) / len(stats)

                logger.info(
                    f"Batch {batch_idx}: reward={avg_reward:.3f} "
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
