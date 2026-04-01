"""Generate PDDL domain/problem instances using AI-Planning/pddl-generators.

Each domain has a corresponding generator script under
  third_party/pddl-generators/<domain>/

We run the generator as a subprocess, capture the output, and save both
the raw PDDL files and a metadata JSON alongside them.
"""
from __future__ import annotations

import json
import subprocess
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from utils.logging import get_logger
from utils.io import ensure_dir, dump_json

logger = get_logger(__name__)

# Root of the repo (two levels up from this file at src/generation/)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_GENERATORS_ROOT = _REPO_ROOT / "third_party" / "pddl-generators"


@dataclass
class InstanceMetadata:
    instance_id: str
    domain: str
    split: str
    seed: int
    generator_args: list[str]
    generator_cmd: str
    domain_file: str          # relative to data/generated/instances/
    problem_file: str         # relative to data/generated/instances/
    generated_at: float = field(default_factory=time.time)
    generator_version: str = "unknown"
    notes: str = ""


def _find_generator_script(domain: str) -> Optional[Path]:
    """Locate the generator script for a domain.

    Tries common naming conventions used in pddl-generators.
    """
    candidates = [
        _GENERATORS_ROOT / domain / "generator.py",
        _GENERATORS_ROOT / domain / f"generate_{domain}.py",
        _GENERATORS_ROOT / domain / "generate.py",
        _GENERATORS_ROOT / domain / "generator.sh",
        _GENERATORS_ROOT / domain / f"generate_{domain}.sh",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Try top-level script variants
    for suffix in [".py", ".sh"]:
        top = _GENERATORS_ROOT / f"generate_{domain}{suffix}"
        if top.exists():
            return top
    return None


def _generator_args_for_domain(domain: str, seed: int, instance_idx: int) -> list[str]:
    """Return generator CLI args for a given domain.

    These are best-effort defaults; domain-specific overrides can be added here.
    Generators in pddl-generators typically take positional args for size params.
    """
    common: dict[str, list[str]] = {
        "blocksworld": ["--num-blocks", "5", "--seed", str(seed + instance_idx)],
        "gripper": ["--num-balls", "4", "--seed", str(seed + instance_idx)],
        "ferry": ["--num-cars", "4", "--num-locations", "3", "--seed", str(seed + instance_idx)],
        "delivery": ["--num-packages", "3", "--num-cities", "4", "--seed", str(seed + instance_idx)],
        "childsnack": ["--num-children", "3", "--seed", str(seed + instance_idx)],
        "floortile": ["--rows", "3", "--cols", "3", "--seed", str(seed + instance_idx)],
        "rovers": ["--num-rovers", "2", "--num-waypoints", "4", "--seed", str(seed + instance_idx)],
        "spanner": ["--num-spanners", "3", "--num-nuts", "3", "--seed", str(seed + instance_idx)],
        "miconic": ["--num-floors", "5", "--num-passengers", "3", "--seed", str(seed + instance_idx)],
        "sokoban": ["--size", "5", "--num-boxes", "2", "--seed", str(seed + instance_idx)],
        "transport": ["--num-cities", "4", "--num-trucks", "2", "--seed", str(seed + instance_idx)],
        "satellite": ["--num-satellites", "2", "--num-targets", "3", "--seed", str(seed + instance_idx)],
    }
    return common.get(domain, ["--seed", str(seed + instance_idx)])


def generate_instance(
    domain: str,
    split: str,
    instance_idx: int,
    seed: int,
    output_dir: Path,
    timeout: int = 30,
) -> InstanceMetadata:
    """Generate one PDDL instance and save files + metadata.

    Returns:
        InstanceMetadata for the generated instance.

    Raises:
        RuntimeError if the generator fails or cannot be found.
    """
    instance_id = f"{domain}_{split}_{instance_idx:04d}_{seed}"
    inst_dir = ensure_dir(output_dir / domain / split)

    domain_file = inst_dir / f"{instance_id}_domain.pddl"
    problem_file = inst_dir / f"{instance_id}_problem.pddl"

    script = _find_generator_script(domain)
    if script is None:
        raise RuntimeError(
            f"No generator found for domain '{domain}' under {_GENERATORS_ROOT}. "
            "Run `make build-tools` to ensure submodules are initialised."
        )

    args = _generator_args_for_domain(domain, seed, instance_idx)

    if script.suffix == ".py":
        cmd = ["python", str(script)] + args + [
            "--domain-file", str(domain_file),
            "--problem-file", str(problem_file),
        ]
    else:
        cmd = ["bash", str(script)] + args + [
            str(domain_file), str(problem_file),
        ]

    logger.info("Generating instance %s: %s", instance_id, " ".join(cmd))
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        raise RuntimeError(
            f"Generator failed for {instance_id} (rc={result.returncode}):\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    # Some generators write to stdout instead of files; handle that case.
    if not domain_file.exists() and result.stdout.strip():
        # Assume stdout contains domain then problem separated by blank line
        parts = result.stdout.split("\n\n", 1)
        domain_file.write_text(parts[0])
        if len(parts) > 1:
            problem_file.write_text(parts[1])
        else:
            problem_file.write_text("")

    logger.info("Generated %s in %.2fs", instance_id, elapsed)

    meta = InstanceMetadata(
        instance_id=instance_id,
        domain=domain,
        split=split,
        seed=seed,
        generator_args=args,
        generator_cmd=" ".join(cmd),
        domain_file=str(domain_file.relative_to(output_dir.parent)),
        problem_file=str(problem_file.relative_to(output_dir.parent)),
    )
    dump_json(asdict(meta), inst_dir / f"{instance_id}_meta.json")
    return meta


def generate_domain_split(
    domain: str,
    split: str,
    n_instances: int,
    seed: int,
    output_dir: Path,
    timeout: int = 30,
) -> list[InstanceMetadata]:
    """Generate n_instances for one (domain, split) pair."""
    metas = []
    for idx in range(n_instances):
        meta = generate_instance(
            domain=domain,
            split=split,
            instance_idx=idx,
            seed=seed,
            output_dir=output_dir,
            timeout=timeout,
        )
        metas.append(meta)
    return metas
