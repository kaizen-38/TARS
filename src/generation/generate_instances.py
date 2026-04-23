"""Generate PDDL domain/problem instances using AI-Planning/pddl-generators.

Each domain has a corresponding generator script under
  third_party/pddl-generators/<domain>/

We run the generator as a subprocess, capture stdout (PDDL), and save both
the raw PDDL files and a metadata JSON alongside them.
"""
from __future__ import annotations

import json
import random
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from utils.logging import get_logger
from utils.io import ensure_dir, dump_json

logger = get_logger(__name__)

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
    domain_file: str
    problem_file: str
    generated_at: float = field(default_factory=time.time)
    generator_version: str = "unknown"
    notes: str = ""


# ---------------------------------------------------------------------------
# Per-domain generator functions
# Each returns (domain_pddl_text, problem_pddl_text) or raises RuntimeError.
# ---------------------------------------------------------------------------

def _run(cmd: str, cwd: Path, timeout: int = 30) -> str:
    """Run a shell command and return stdout. Raises RuntimeError on failure."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True,
        cwd=cwd, timeout=timeout
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (rc={result.returncode}):\n"
            f"CMD: {cmd}\nSTDERR: {result.stderr[:500]}"
        )
    return result.stdout


def _gen_blocksworld(seed: int, idx: int) -> tuple[str, str]:
    rng = random.Random(seed)
    blocks = rng.randint(3, 8)
    cwd = _GENERATORS_ROOT / "blocksworld"
    problem = _run(f"./blocksworld 4 {blocks} {seed}", cwd)
    domain = (cwd / "4ops" / "domain.pddl").read_text()
    return domain, problem


def _gen_gripper(seed: int, idx: int) -> tuple[str, str]:
    rng = random.Random(seed)
    balls = rng.randint(2, 6)
    cwd = _GENERATORS_ROOT / "gripper"
    problem = _run(f"./gripper -n {balls}", cwd)
    domain = (cwd / "4ops" / "domain.pddl").read_text()
    return domain, problem


def _gen_ferry(seed: int, idx: int) -> tuple[str, str]:
    rng = random.Random(seed)
    locs = rng.randint(2, 4)
    cars = rng.randint(2, 5)
    cwd = _GENERATORS_ROOT / "ferry"
    problem = _run(f"./ferry -l {locs} -c {cars} -s {seed}", cwd)
    domain = (cwd / "4ops" / "domain.pddl").read_text()
    return domain, problem


def _gen_childsnack(seed: int, idx: int) -> tuple[str, str]:
    rng = random.Random(seed)
    children = rng.randint(3, 4)
    trays = 2
    cwd = _GENERATORS_ROOT / "childsnack"
    problem = _run(
        f"python child-snack-generator.py pool {seed} {children} {trays} 0.4 1.3",
        cwd
    )
    domain = (cwd / "4ops" / "domain.pddl").read_text()
    return domain, problem


def _gen_floortile(seed: int, idx: int) -> tuple[str, str]:
    rng = random.Random(seed)
    rows = 2
    cols = rng.randint(2, 3)
    cwd = _GENERATORS_ROOT / "floortile"
    problem = _run(
        f"python floortile-generator.py p{idx:04d} {rows} {cols} 2 seq {seed}",
        cwd
    )
    domain = (cwd / "4ops" / "domain.pddl").read_text()
    return domain, problem


def _gen_rovers(seed: int, idx: int) -> tuple[str, str]:
    rng = random.Random(seed)
    waypoints = rng.randint(3, 4)
    goals = rng.randint(1, 2)
    cwd = _GENERATORS_ROOT / "rovers"
    problem = _run(f"./rovgen {seed} 1 {waypoints} 1 1 {goals}", cwd)
    domain = (cwd / "4ops" / "domain.pddl").read_text()
    return domain, problem


def _gen_spanner(seed: int, idx: int) -> tuple[str, str]:
    rng = random.Random(seed)
    spanners = rng.randint(2, 5)
    nuts = rng.randint(1, spanners)
    locations = rng.randint(2, 4)
    cwd = _GENERATORS_ROOT / "spanner"
    problem = _run(
        f"python spanner-generator.py {spanners} {nuts} {locations} --seed {seed}",
        cwd
    )
    domain = (cwd / "4ops" / "domain.pddl").read_text()
    return domain, problem
def _gen_delivery(seed: int, idx: int) -> tuple[str, str]:
    static_dir = _REPO_ROOT / "data" / "static" / "delivery" / "train"
    instances = sorted(static_dir.glob("instance_*.pddl"))
    if not instances:
        raise RuntimeError(
            "No pre-generated delivery instances found in data/static/delivery/train/. "
            "Run: cd third_party/pddl-generators/delivery && "
            "uv run generate.py --grid_size_splits=3,5,7 --max_nr_packages=2 --nr_instances_per_setup=15"
        )
    domain_file = static_dir / "domain.pddl"
    instance_file = instances[idx % len(instances)]
    return domain_file.read_text(), instance_file.read_text()
    
def _gen_miconic(seed: int, idx: int) -> tuple[str, str]:
    rng = random.Random(seed)
    floors = rng.randint(2, 6)
    passengers = rng.randint(1, floors)
    cwd = _GENERATORS_ROOT / "miconic"
    problem = _run(f"./miconic -f {floors} -p {passengers} -r {seed}", cwd)
    domain = (cwd / "4ops" / "domain.pddl").read_text()
    return domain, problem


def _gen_sokoban(seed: int, idx: int) -> tuple[str, str]:
    rng = random.Random(seed)
    grid = rng.randint(6, 8)
    boxes = rng.randint(1, 2)
    walls = rng.randint(3, 6)
    cwd = _GENERATORS_ROOT / "sokoban" / "random"
    problem = _run(
        f"./sokoban-generator-typed -n {grid} -b {boxes} -w {walls} -s {seed}",
        cwd
    )
    if not problem.strip():
        raise RuntimeError("Sokoban generator produced empty output")
    domain = (_GENERATORS_ROOT / "sokoban" / "domain.pddl").read_text()
    return domain, problem


def _gen_transport(seed: int, idx: int) -> tuple[str, str]:
    rng = random.Random(seed)
    nodes = rng.randint(4, 7)
    trucks = rng.randint(1, 2)
    packages = rng.randint(1, 3)
    cwd = _GENERATORS_ROOT / "transport"
    problem = _run(
        f"python city-generator.py {nodes} 1000 3 100 {trucks} {packages} {seed}",
        cwd
    )
    domain = (cwd / "4ops" / "domain.pddl").read_text()
    return domain, problem


def _gen_satellite(seed: int, idx: int) -> tuple[str, str]:
    rng = random.Random(seed)
    targets = rng.randint(5, 8)
    obs = rng.randint(3, targets)
    cwd = _GENERATORS_ROOT / "satellite"
    problem = _run(f"./satgen {seed} 2 3 3 {targets} {obs}", cwd)
    domain = (cwd / "4ops" / "domain.pddl").read_text()
    return domain, problem


# ---------------------------------------------------------------------------
# Domain registry
# ---------------------------------------------------------------------------

_GENERATORS: dict[str, callable] = {
    "blocksworld": _gen_blocksworld,
    "gripper":     _gen_gripper,
    "ferry":       _gen_ferry,
    "childsnack":  _gen_childsnack,
    "floortile":   _gen_floortile,
    "rovers":      _gen_rovers,
    "spanner":     _gen_spanner,
    "miconic":     _gen_miconic,
    "sokoban":     _gen_sokoban,
    "transport":   _gen_transport,
    "satellite":   _gen_satellite,
    "delivery":    _gen_delivery,
}


# ---------------------------------------------------------------------------
# Public API (matches Rhythm's expected interface)
# ---------------------------------------------------------------------------

def generate_instance(
    domain: str,
    split: str,
    instance_idx: int,
    seed: int,
    output_dir: Path,
    timeout: int = 30,
) -> InstanceMetadata:
    """Generate one PDDL instance and save files + metadata."""

    gen_fn = _GENERATORS.get(domain)
    if gen_fn is None:
        raise RuntimeError(
            f"No generator implemented for domain '{domain}'. "
            f"Available: {list(_GENERATORS.keys())}"
        )

    instance_id = f"{domain}_{split}_{instance_idx:04d}_{seed}"
    inst_dir = ensure_dir(output_dir / domain / split)

    domain_file = inst_dir / f"{instance_id}_domain.pddl"
    problem_file = inst_dir / f"{instance_id}_problem.pddl"

    logger.info("Generating %s", instance_id)
    t0 = time.perf_counter()

    domain_text, problem_text = gen_fn(seed, instance_idx)
    elapsed = time.perf_counter() - t0

    domain_file.write_text(domain_text, encoding="utf-8")
    problem_file.write_text(problem_text, encoding="utf-8")

    logger.info("Generated %s in %.2fs", instance_id, elapsed)

    meta = InstanceMetadata(
        instance_id=instance_id,
        domain=domain,
        split=split,
        seed=seed,
        generator_args=[],
        generator_cmd=f"{domain} seed={seed} idx={instance_idx}",
        domain_file=str(domain_file.resolve().relative_to(_REPO_ROOT)),
        problem_file=str(problem_file.resolve().relative_to(_REPO_ROOT)),
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
    rng = random.Random(seed)
    metas = []
    for idx in range(n_instances):
        instance_seed = rng.randint(1, 999999)
        try:
            meta = generate_instance(
                domain=domain,
                split=split,
                instance_idx=idx,
                seed=instance_seed,
                output_dir=output_dir,
                timeout=timeout,
            )
            metas.append(meta)
        except Exception as exc:
            logger.warning("FAILED %s idx=%d: %s", domain, idx, exc)
    return metas