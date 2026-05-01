#!/usr/bin/env python3
"""Test GRPO setup before running full training."""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

print("Testing GRPO setup...")
print("=" * 50)

# Test 1: TRL import
try:
    from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
    print("✓ TRL imports successful")
except Exception as e:
    print(f"❌ TRL import failed: {e}")
    print("   Run: pip install trl")
    sys.exit(1)

# Test 2: Internal imports
try:
    from generation.validate_with_val import validate_plan
    from pddl_ops.decode_compact_plan import decode_compact_plan
    from utils.logging import get_logger
    print("✓ Internal imports successful")
except Exception as e:
    print(f"❌ Internal imports failed: {e}")
    sys.exit(1)

# Test 3: Check SFT checkpoint exists
sft_path = _REPO_ROOT / "runs/qwen3_mini_direct"
if not sft_path.exists():
    print(f"❌ SFT checkpoint not found at {sft_path}")
    print("   Run full baseline first: bash scripts/submit_full.sh")
    sys.exit(1)
print(f"✓ SFT checkpoint exists: {sft_path}")

# Test 4: Check training data exists
tuples_dir = _REPO_ROOT / "data/generated/tuples_standard"
train_files = list(tuples_dir.glob("*_train_*_tuple.json"))
if not train_files:
    print(f"❌ No training tuples found in {tuples_dir}")
    sys.exit(1)
print(f"✓ Found {len(train_files)} training tuples")

# Test 5: Test decode_compact_plan
try:
    parsed = decode_compact_plan("move a b")
    assert len(parsed.to_pddl_lines()) == 1
    print("✓ decode_compact_plan works")
except Exception as e:
    print(f"❌ decode_compact_plan failed: {e}")
    sys.exit(1)

print("=" * 50)
print("✅ All checks passed! Ready for GRPO training.")
