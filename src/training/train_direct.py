#!/usr/bin/env python3
"""Direct training script bypassing LLaMAFactory CLI issues.

Uses Transformers + datasets directly for robust training.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from utils.logging import get_logger

logger = get_logger(__name__)


def main():
    model_name = "Qwen/Qwen3-1.7B"
    output_dir = _REPO_ROOT / "runs" / "qwen3_mini_direct"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    dataset_files = {
        "train": [
            str(_REPO_ROOT / "data/datasets/alpaca/phase1_standard_train.jsonl"),
            str(_REPO_ROOT / "data/datasets/alpaca/phase1_anonymized_train.jsonl"),
            str(_REPO_ROOT / "data/datasets/alpaca/phase1_compact_train.jsonl"),
        ]
    }

    logger.info("Loading datasets...")
    dataset = load_dataset("json", data_files=dataset_files)
    logger.info(f"Loaded {len(dataset['train'])} training examples")

    # Load tokenizer and model
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False

    # Tokenize function for Alpaca format
    def tokenize_function(examples):
        # Alpaca format: instruction + input (problem) -> output (plan)
        prompts = []
        responses = []
        for inst, inp, out in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            prompt = f"{inst}\n\n{inp}"
            prompts.append(prompt)
            responses.append(out)

        # Tokenize prompt + response together
        full_texts = [p + r for p, r in zip(prompts, responses)]
        tokenized = tokenizer(
            full_texts,
            truncation=True,
            max_length=4096,
            padding=False,
        )

        # Also tokenize just prompts to mask them in loss
        prompt_tokenized = tokenizer(
            prompts,
            truncation=True,
            max_length=4096,
            padding=False,
        )

        # Create labels (mask prompt, only train on response)
        labels = []
        for i, input_ids in enumerate(tokenized["input_ids"]):
            prompt_len = len(prompt_tokenized["input_ids"][i])
            label = [-100] * prompt_len + input_ids[prompt_len:]
            labels.append(label)

        tokenized["labels"] = labels
        return tokenized

    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=4,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=500,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        seed=42,
        data_seed=42,
        report_to="none",
        save_total_limit=2,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
