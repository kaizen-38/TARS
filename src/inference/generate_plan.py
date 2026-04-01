"""Low-level plan generation using a local HF model.

Always sets enable_thinking=False for Phase 1 baseline.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

from utils.logging import get_logger
from utils.io import load_yaml

logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class GenerationResult:
    prompt: str
    raw_output: str
    generated_tokens: int
    generation_time_sec: float
    model_name: str
    checkpoint_path: Optional[str]
    decoding_config: dict[str, Any]


def _build_prompt(
    domain_text: str,
    problem_text: str,
    representation: str,
    system_prompt: str,
) -> str:
    """Build the chat-formatted prompt for the model."""
    if representation == "compact":
        task_instruction = (
            "Solve the following PDDL planning problem. "
            "Return a valid plan in compact format: one action per line, "
            "without parentheses."
        )
    elif representation == "anonymized":
        task_instruction = (
            "Solve the following anonymized PDDL planning problem. "
            "All identifiers have been renamed. "
            "Return a valid plan as a sequence of PDDL actions, one per line."
        )
    else:
        task_instruction = (
            "Solve the following PDDL planning problem. "
            "Return a valid plan as a sequence of PDDL actions, one per line."
        )

    user_content = (
        f"{task_instruction}\n\n"
        f"DOMAIN:\n{domain_text.strip()}\n\n"
        f"PROBLEM:\n{problem_text.strip()}"
    )
    return user_content


def load_model_and_tokenizer(
    model_name_or_path: str,
    checkpoint_path: Optional[str] = None,
    torch_dtype: str = "bfloat16",
    use_flash_attention_2: bool = False,
):
    """Load HF model and tokenizer.

    Args:
        model_name_or_path: HF model identifier or local path.
        checkpoint_path: Path to a fine-tuned checkpoint (overrides model_name_or_path).
        torch_dtype: 'bfloat16', 'float16', or 'float32'.
        use_flash_attention_2: Whether to enable Flash Attention 2.

    Returns:
        (model, tokenizer) tuple.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_path = checkpoint_path or model_name_or_path
    logger.info("Loading model from %s", load_path)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    attn_kwargs: dict[str, Any] = {}
    if use_flash_attention_2:
        attn_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        **attn_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
    model.eval()
    return model, tokenizer


def generate_plan(
    domain_text: str,
    problem_text: str,
    representation: str,
    model,
    tokenizer,
    decoding_config: dict[str, Any],
    model_name: str,
    checkpoint_path: Optional[str] = None,
    system_prompt: str = (
        "You are an expert AI planning assistant. "
        "Given a PDDL domain and problem, produce a valid sequential plan. "
        "Output ONLY the plan actions, one per line, with no additional explanation."
    ),
) -> GenerationResult:
    """Generate a plan for a single PDDL instance.

    Args:
        domain_text: PDDL domain string.
        problem_text: PDDL problem string.
        representation: 'standard', 'anonymized', or 'compact'.
        model: Loaded HF model.
        tokenizer: Loaded HF tokenizer.
        decoding_config: Dict with keys like max_new_tokens, do_sample, etc.
        model_name: Name for logging.
        checkpoint_path: Checkpoint path for logging.
        system_prompt: System message.

    Returns:
        GenerationResult with raw output and timing.
    """
    import torch

    # ALWAYS force enable_thinking=False per Phase 1 spec
    config = dict(decoding_config)
    config.pop("enable_thinking", None)  # remove if present; handled via tokenizer

    user_content = _build_prompt(domain_text, problem_text, representation, system_prompt)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # apply_chat_template with enable_thinking=False is the Qwen3 convention
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Phase 1 requirement
        )
    except TypeError:
        # Older tokenizers may not support enable_thinking
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": config.get("max_new_tokens", 2048),
        "do_sample": config.get("do_sample", False),
        "pad_token_id": tokenizer.eos_token_id,
    }
    if gen_kwargs["do_sample"]:
        for key in ("temperature", "top_p", "top_k"):
            if key in config:
                gen_kwargs[key] = config[key]
        if "presence_penalty" in config:
            gen_kwargs["repetition_penalty"] = 1.0 + config["presence_penalty"] * 0.1
    else:
        gen_kwargs["temperature"] = config.get("temperature", 0.01)

    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)
    elapsed = time.perf_counter() - t0

    new_ids = output_ids[0][input_len:]
    raw_output = tokenizer.decode(new_ids, skip_special_tokens=True)
    generated_tokens = len(new_ids)

    logger.info(
        "Generated %d tokens in %.2fs for %s/%s",
        generated_tokens, elapsed, representation, model_name
    )

    return GenerationResult(
        prompt=text,
        raw_output=raw_output,
        generated_tokens=generated_tokens,
        generation_time_sec=elapsed,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        decoding_config=config,
    )
