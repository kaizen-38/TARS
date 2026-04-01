"""Seed management utilities.

Centralises seed setting so that all components (torch, numpy, random,
transformers) are seeded consistently from a single integer.
"""
from __future__ import annotations

import random


def set_global_seed(seed: int) -> None:
    """Set seeds for all relevant libraries."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        from transformers import set_seed as hf_set_seed
        hf_set_seed(seed)
    except ImportError:
        pass


def derive_seed(base: int, domain: str, problem_id: str) -> int:
    """Derive a deterministic per-instance seed from base seed + identifiers.

    Keeps instance-level randomness independent while remaining reproducible.
    """
    combined = f"{base}:{domain}:{problem_id}"
    return int(abs(hash(combined)) % (2**31))
