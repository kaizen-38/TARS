#!/usr/bin/env python3
"""Check PPOTrainer API."""

try:
    from trl import PPOTrainer
    import inspect

    sig = inspect.signature(PPOTrainer.__init__)
    params = [p for p in sig.parameters.keys() if p != 'self']

    print(f"Valid PPOTrainer.__init__ parameters ({len(params)}):")
    print("=" * 60)
    for param in params:
        param_obj = sig.parameters[param]
        default = param_obj.default
        if default == inspect.Parameter.empty:
            print(f"  {param} (required)")
        else:
            print(f"  {param} = {repr(default)[:50]}")

except ImportError:
    print("TRL not installed")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
