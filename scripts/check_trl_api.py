#!/usr/bin/env python3
"""Check TRL API and print valid PPOConfig parameters."""

try:
    import trl
    print(f"TRL version: {trl.__version__}")

    from trl import PPOConfig
    import inspect

    # Get PPOConfig signature
    sig = inspect.signature(PPOConfig.__init__)
    params = [p for p in sig.parameters.keys() if p != 'self']

    print(f"\nValid PPOConfig parameters ({len(params)}):")
    print("=" * 60)
    for param in params:
        param_obj = sig.parameters[param]
        default = param_obj.default
        if default == inspect.Parameter.empty:
            print(f"  {param} (required)")
        else:
            print(f"  {param} = {default}")

except ImportError:
    print("TRL not installed. Run: pip install trl")
    print("\nExpected parameters based on TRL 0.8.x:")
    print("  - batch_size")
    print("  - mini_batch_size")
    print("  - learning_rate")
    print("  - gradient_accumulation_steps")
    print("  - adap_kl_ctrl")
    print("  - init_kl_coef")
    print("  - target")
    print("  - horizon")
    print("  - gamma")
    print("  - lam")
    print("  - cliprange")
    print("  - cliprange_value")
    print("  - vf_coef")
    print("  - seed")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
