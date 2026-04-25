"""Solve PDDL problems using Fast Downward (default) or a pluggable backend."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from utils.logging import get_logger
from utils.io import ensure_dir, dump_json

logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FD_ROOT = _REPO_ROOT / "third_party" / "downward"


@dataclass
class SolveResult:
    problem_id: str
    domain_file: str
    problem_file: str
    backend: str
    success: bool
    exit_code: int
    wall_clock_sec: float
    raw_plan_file: Optional[str]
    normalized_plan_file: Optional[str]
    action_sequence: list[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None


class PlannerBackend(ABC):
    @abstractmethod
    def solve(
        self,
        domain_file: Path,
        problem_file: Path,
        timeout: int,
    ) -> tuple[int, str, str, list[str]]:
        pass


class FastDownwardBackend(PlannerBackend):
    DEFAULT_SEARCH = "--search lazy_greedy([ff()], preferred=[ff()])"

    def __init__(self, fd_root: Path = _FD_ROOT, search_config: Optional[str] = None) -> None:
        self.fd_root = fd_root
        self.search_config = search_config or self.DEFAULT_SEARCH
        self._fd_bin: Optional[Path] = None

    def _get_binary(self) -> Path:
        if self._fd_bin is not None:
            return self._fd_bin
        candidates = [
            self.fd_root / "fast-downward.py",
            self.fd_root / "src" / "fast-downward.py",
        ]
        for c in candidates:
            if c.exists():
                self._fd_bin = c
                return c
        raise RuntimeError(
            f"Fast Downward binary not found under {self.fd_root}. "
            "Run `make build-tools`."
        )

    def solve(
        self,
        domain_file: Path,
        problem_file: Path,
        timeout: int,
    ) -> tuple[int, str, str, list[str]]:
        binary = self._get_binary()
        _tmp_dir = Path(tempfile.mkdtemp(prefix="tars_fd_"))

        try:
            if self.search_config.startswith("--search "):
                search_expr = self.search_config[len("--search "):]
                cmd = [
                    "python", str(binary.resolve()),
                    str(domain_file.resolve()), str(problem_file.resolve()),
                    "--search", search_expr,
                ]
            elif self.search_config.startswith("--alias "):
                alias = self.search_config[len("--alias "):]
                cmd = [
                    "python", str(binary.resolve()),
                    str(domain_file.resolve()), str(problem_file.resolve()),
                    "--alias", alias,
                ]
            else:
                cmd = (
                    ["python", str(binary.resolve()),
                     str(domain_file.resolve()), str(problem_file.resolve())]
                    + self.search_config.split()
                )

            logger.info("FD solve: %s", " ".join(cmd))
            fd_env = os.environ.copy()
            conda_lib = os.path.join(os.environ.get("CONDA_PREFIX", ""), "lib")
            fd_env["LD_LIBRARY_PATH"] = conda_lib + ":" + fd_env.get("LD_LIBRARY_PATH", "")
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
                cwd=str(_tmp_dir), env=fd_env
            )

            plan_files = sorted(_tmp_dir.glob("sas_plan*"))
            actions = _parse_fd_plan(plan_files[-1].read_text()) if plan_files else []

            return result.returncode, result.stdout, result.stderr, actions
        finally:
            shutil.rmtree(_tmp_dir, ignore_errors=True)


def _parse_fd_plan(text: str) -> list[str]:
    actions: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        if stripped.startswith("("):
            if "[" in stripped:
                stripped = stripped[: stripped.rfind("[")].strip()
            if stripped.endswith(")"):
                actions.append(stripped.lower())
    return actions


PLANNER_REGISTRY: dict[str, type[PlannerBackend]] = {
    "fd": FastDownwardBackend,
    "fast-downward": FastDownwardBackend,
}


def get_backend(name: str) -> PlannerBackend:
    cls = PLANNER_REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown planner backend '{name}'. "
            f"Available: {list(PLANNER_REGISTRY.keys())}"
        )
    return cls()


def solve_instance(
    problem_id: str,
    domain_file: Path,
    problem_file: Path,
    output_dir: Path,
    backend: str = "fd",
    timeout: int = 120,
) -> SolveResult:
    ensure_dir(output_dir)
    planner = get_backend(backend)

    raw_plan_file = output_dir / f"{problem_id}.raw_plan.txt"
    norm_plan_file = output_dir / f"{problem_id}.plan.pddl"
    meta_file = output_dir / f"{problem_id}.solve_meta.json"

    t0 = time.perf_counter()
    try:
        exit_code, stdout, stderr, actions = planner.solve(
            domain_file, problem_file, timeout
        )
        elapsed = time.perf_counter() - t0
        success = exit_code == 0 and len(actions) > 0

        raw_plan_file.write_text(stdout, encoding="utf-8")
        if actions:
            norm_plan_file.write_text("\n".join(actions) + "\n", encoding="utf-8")

        result = SolveResult(
            problem_id=problem_id,
            domain_file=str(domain_file),
            problem_file=str(problem_file),
            backend=backend,
            success=success,
            exit_code=exit_code,
            wall_clock_sec=elapsed,
            raw_plan_file=str(raw_plan_file) if raw_plan_file.exists() else None,
            normalized_plan_file=str(norm_plan_file) if norm_plan_file.exists() else None,
            action_sequence=actions,
            stdout=stdout,
            stderr=stderr,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        result = SolveResult(
            problem_id=problem_id,
            domain_file=str(domain_file),
            problem_file=str(problem_file),
            backend=backend,
            success=False,
            exit_code=-1,
            wall_clock_sec=elapsed,
            raw_plan_file=None,
            normalized_plan_file=None,
            error=f"Timeout after {timeout}s",
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        result = SolveResult(
            problem_id=problem_id,
            domain_file=str(domain_file),
            problem_file=str(problem_file),
            backend=backend,
            success=False,
            exit_code=-2,
            wall_clock_sec=elapsed,
            raw_plan_file=None,
            normalized_plan_file=None,
            error=str(exc),
        )

    dump_json(asdict(result), meta_file)
    log_fn = logger.info if result.success else logger.warning
    log_fn(
        "Solve %s: success=%s backend=%s time=%.2fs",
        problem_id,
        result.success,
        backend,
        result.wall_clock_sec,
    )
    return result
