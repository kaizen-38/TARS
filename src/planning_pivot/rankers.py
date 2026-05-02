from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any, Protocol

from planning_pivot.sas import SasTask, SasOperator, apply_operator, is_goal


class ActionRanker(Protocol):
    name: str

    def score(
        self,
        task: SasTask,
        state: tuple[int, ...],
        candidates: list[SasOperator],
        context: dict[str, Any],
    ) -> list[float]:
        ...


class RandomRanker:
    name = "random"

    def __init__(self, seed: int = 574):
        self._rng = random.Random(seed)

    def score(
        self,
        task: SasTask,
        state: tuple[int, ...],
        candidates: list[SasOperator],
        context: dict[str, Any],
    ) -> list[float]:
        return [self._rng.random() for _ in candidates]


class GoalCountRanker:
    name = "goal_count"

    def score(
        self,
        task: SasTask,
        state: tuple[int, ...],
        candidates: list[SasOperator],
        context: dict[str, Any],
    ) -> list[float]:
        scores: list[float] = []
        for op in candidates:
            next_state = apply_operator(task, state, op)
            goal_count = sum(1 for var, val in task.goals if next_state[var] == val)
            changed = sum(1 for a, b in zip(state, next_state) if a != b)
            scores.append(goal_count + changed * 0.01)
        return scores


class TeacherFrequencyRanker:
    name = "teacher_frequency"

    def __init__(self, teacher_plan_paths: list[Path]):
        self._freq: dict[str, int] = {}
        for p in teacher_plan_paths:
            if not p.exists():
                continue
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith(";"):
                    continue
                schema = self._extract_schema(line)
                if schema:
                    self._freq[schema] = self._freq.get(schema, 0) + 1

    @staticmethod
    def _extract_schema(action_line: str) -> str | None:
        inner = action_line.strip("() ").lower()
        parts = inner.split()
        return parts[0] if parts else None

    def score(
        self,
        task: SasTask,
        state: tuple[int, ...],
        candidates: list[SasOperator],
        context: dict[str, Any],
    ) -> list[float]:
        scores: list[float] = []
        for op in candidates:
            schema = op.name.split()[0].lower()
            scores.append(float(self._freq.get(schema, 0)))
        return scores


def _try_import_hf():
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer, AutoModelForCausalLM
        return torch, F, AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        return None


class HFLogprobRanker:
    name = "hf_logprob"

    def __init__(
        self,
        model_name_or_path: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_prompt_chars: int = 6000,
    ):
        imports = _try_import_hf()
        if imports is None:
            raise ImportError("torch and transformers are required for HFLogprobRanker")
        torch, F, AutoTokenizer, AutoModelForCausalLM = imports
        self._torch = torch
        self._F = F
        self._max_prompt_chars = max_prompt_chars

        dtype = getattr(torch, torch_dtype) if torch_dtype != "auto" else "auto"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map=device_map, torch_dtype=dtype,
        )
        self._model.eval()

    def make_prompt(
        self,
        task: SasTask,
        state: tuple[int, ...],
        candidates: list[SasOperator],
        context: dict[str, Any],
    ) -> str:
        goal_desc = ", ".join(
            f"{task.variables[v].name}={task.variables[v].values[val]}"
            for v, val in task.goals
        )
        state_desc = ", ".join(
            f"{task.variables[i].name}={task.variables[i].values[v]}"
            for i, v in enumerate(state) if v != 0
        )
        prefix_actions = context.get("prefix_actions", [])
        prefix_str = "\n".join(prefix_actions[-10:]) if prefix_actions else "(start)"

        prompt = (
            f"Goal: {goal_desc}\n"
            f"State: {state_desc}\n"
            f"Previous actions:\n{prefix_str}\n"
            f"Next action:"
        )
        return prompt[-self._max_prompt_chars:]

    def score_prompt_candidates(self, prompt: str, candidates: list[str]) -> list[float]:
        scores: list[float] = []
        prompt_ids = self._tokenizer.encode(prompt, return_tensors="pt").to(self._model.device)

        for cand in candidates:
            cand_ids = self._tokenizer.encode(cand, add_special_tokens=False, return_tensors="pt").to(self._model.device)
            input_ids = self._torch.cat([prompt_ids, cand_ids], dim=-1)

            with self._torch.no_grad():
                outputs = self._model(input_ids)
                logits = outputs.logits[0]

            prompt_len = prompt_ids.shape[-1]
            total_logprob = 0.0
            for i in range(cand_ids.shape[-1]):
                pos = prompt_len - 1 + i
                log_probs = self._F.log_softmax(logits[pos], dim=-1)
                token_id = cand_ids[0, i]
                total_logprob += log_probs[token_id].item()

            avg_logprob = total_logprob / max(cand_ids.shape[-1], 1)
            scores.append(avg_logprob)

        return scores

    def score(
        self,
        task: SasTask,
        state: tuple[int, ...],
        candidates: list[SasOperator],
        context: dict[str, Any],
    ) -> list[float]:
        from planning_pivot.sas import canonical_operator_plan_line
        prompt = self.make_prompt(task, state, candidates, context)
        cand_strings = [canonical_operator_plan_line(op.name) for op in candidates]
        return self.score_prompt_candidates(prompt, cand_strings)
