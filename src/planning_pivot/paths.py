from __future__ import annotations

import os
import shutil
from pathlib import Path


def discover_fast_downward(configured: str | None = None) -> Path:
    if configured:
        p = Path(configured)
        if p.exists():
            return p

    env_val = os.environ.get("FAST_DOWNWARD")
    if env_val:
        p = Path(env_val)
        if p.exists():
            return p

    found = shutil.which("fast-downward.py")
    if found:
        return Path(found)

    repo_candidates = [
        Path("fast-downward.py"),
        Path("fast-downward/fast-downward.py"),
        Path("downward/fast-downward.py"),
        Path("third_party/downward/fast-downward.py"),
    ]
    for c in repo_candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError(
        "Fast Downward not found. Set FAST_DOWNWARD env var, pass --fd-path, "
        "or place fast-downward.py in the repo root or third_party/downward/."
    )


def discover_val(configured: str | None = None) -> Path:
    if configured:
        p = Path(configured)
        if p.exists():
            return p

    env_val = os.environ.get("VAL_BIN")
    if env_val:
        p = Path(env_val)
        if p.exists():
            return p

    for name in ["Validate", "validate", "VAL"]:
        found = shutil.which(name)
        if found:
            return Path(found)

    repo_candidates = [
        Path("third_party/VAL/build/bin/Validate"),
        Path("third_party/VAL/bin/Validate"),
        Path("third_party/VAL/Validate"),
        Path("VAL/build/bin/Validate"),
        Path("val/build/bin/Validate"),
        Path("bin/Validate"),
        Path("bin/validate"),
    ]
    for c in repo_candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError(
        "VAL Validate binary not found. Set VAL_BIN env var, pass --val-path, "
        "or build VAL under third_party/VAL/build/bin/Validate."
    )
