"""
Deprecated modules from WANN SDK.

These modules are kept for backwards compatibility but are no longer
actively maintained. Use the main wann_sdk modules instead.

Deprecated:
    - ray_env: Use wann_sdk.environments.BraxEnv instead
    - gymnasium_env: Use wann_sdk.environments.BraxEnv instead

Migration Guide:
    Old (deprecated):
        from wann_sdk.deprecated.ray_env import GymnasiumEnvWrapper
        env = GymnasiumEnvWrapper("BipedalWalker-v3")

    New (recommended):
        from wann_sdk import BraxEnv
        env = BraxEnv("humanoid")  # Use Brax equivalent
"""

import warnings

def _warn_deprecated(module_name: str):
    warnings.warn(
        f"{module_name} is deprecated. Use wann_sdk.environments.BraxEnv instead.",
        DeprecationWarning,
        stacklevel=3,
    )


# Re-export with warnings
def get_gymnasium_wrapper():
    """Get deprecated GymnasiumEnvWrapper with warning."""
    _warn_deprecated("GymnasiumEnvWrapper")
    from .ray_env import GymnasiumEnvWrapper
    return GymnasiumEnvWrapper


def get_distributed_pool():
    """Get deprecated DistributedEnvPool with warning."""
    _warn_deprecated("DistributedEnvPool")
    from .ray_env import DistributedEnvPool
    return DistributedEnvPool
