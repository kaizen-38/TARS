from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommandResult:
    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str
    seconds: float


def run_cmd(
    cmd: list[str],
    cwd: Path | None = None,
    timeout: int | float | None = None,
    env: dict[str, str] | None = None,
) -> CommandResult:
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd else None,
            timeout=timeout,
            env=env,
        )
        elapsed = time.perf_counter() - t0
        return CommandResult(
            cmd=cmd,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            seconds=elapsed,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        return CommandResult(
            cmd=cmd,
            returncode=-1,
            stdout="",
            stderr=f"Command timed out after {timeout}s",
            seconds=elapsed,
        )
    except FileNotFoundError:
        elapsed = time.perf_counter() - t0
        return CommandResult(
            cmd=cmd,
            returncode=-2,
            stdout="",
            stderr=f"Command not found: {cmd[0]}",
            seconds=elapsed,
        )


def which_any(names: list[str]) -> Path | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None
