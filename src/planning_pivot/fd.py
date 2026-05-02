from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from planning_pivot.shell import run_cmd

_PYTHON = sys.executable


@dataclass(frozen=True)
class FDResult:
    domain_path: Path
    problem_path: Path
    mode: str
    solved: bool
    plan_path: Path | None
    returncode: int
    stdout: str
    stderr: str
    seconds: float


_SEARCH_CONFIGS: dict[str, list[str]] = {
    "lama_first": ["--alias", "lama-first"],
    "lama2011": ["--alias", "seq-sat-lama-2011"],
    "lmcut": ["--search", "astar(lmcut())"],
    "ff_greedy": [
        "--search",
        "let(hff, ff(), lazy_greedy([hff], preferred=[hff]))",
    ],
}


def run_fd_translate(
    domain_path: Path,
    problem_path: Path,
    fd_bin: Path,
    out_dir: Path,
    timeout: int = 60,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        _PYTHON, str(fd_bin),
        "--translate",
        str(domain_path.resolve()),
        str(problem_path.resolve()),
    ]
    result = run_cmd(cmd, cwd=out_dir, timeout=timeout)
    sas_path = out_dir / "output.sas"
    if not sas_path.exists():
        raise RuntimeError(
            f"FD translate failed (rc={result.returncode}): {result.stderr[:500]}"
        )
    return sas_path


def run_fd_search(
    domain_path: Path,
    problem_path: Path,
    fd_bin: Path,
    out_dir: Path,
    mode: str,
    timeout: int = 60,
) -> FDResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode not in _SEARCH_CONFIGS:
        raise ValueError(f"Unknown FD mode '{mode}'. Available: {list(_SEARCH_CONFIGS)}")

    search_args = _SEARCH_CONFIGS[mode]
    if search_args[0] == "--alias":
        cmd = [
            _PYTHON, str(fd_bin),
            "--alias", search_args[1],
            str(domain_path.resolve()),
            str(problem_path.resolve()),
        ]
    else:
        cmd = [
            _PYTHON, str(fd_bin),
            str(domain_path.resolve()),
            str(problem_path.resolve()),
        ] + search_args

    result = run_cmd(cmd, cwd=out_dir, timeout=timeout)
    plan_path = find_fd_plan_file(out_dir)
    solved = plan_path is not None and result.returncode in (0, 12)

    return FDResult(
        domain_path=domain_path,
        problem_path=problem_path,
        mode=mode,
        solved=solved,
        plan_path=plan_path,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        seconds=result.seconds,
    )


def find_fd_plan_file(out_dir: Path) -> Path | None:
    candidates: list[Path] = []
    candidates.extend(sorted(out_dir.glob("sas_plan*")))
    candidates.extend(sorted(out_dir.glob("plan.out*")))
    candidates.extend(sorted(out_dir.glob("*.plan")))
    candidates.extend(sorted(out_dir.glob("*.soln")))

    valid: list[tuple[int, Path]] = []
    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            text = p.read_text()
            action_lines = [
                l for l in text.splitlines()
                if l.strip() and not l.strip().startswith(";")
            ]
            if action_lines:
                valid.append((len(action_lines), p))

    if not valid:
        return None
    valid.sort(key=lambda x: x[0])
    return valid[0][1]
