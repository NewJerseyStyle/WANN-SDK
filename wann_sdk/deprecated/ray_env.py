"""
DEPRECATED: Ray Environment Service and Gymnasium Integration

This module is deprecated. Please use wann_sdk.environments.BraxEnv instead.

Migration:
    Old: from wann_sdk.deprecated.ray_env import GymnasiumEnvWrapper
    New: from wann_sdk import BraxEnv
"""

import warnings

warnings.warn(
    "wann_sdk.deprecated.ray_env is deprecated. "
    "Use wann_sdk.environments.BraxEnv instead.",
    DeprecationWarning,
    stacklevel=2,
)

import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Callable
from dataclasses import dataclass

try:
    import gymnasium as gym
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False

try:
    import ray
    from ray import serve
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


class GymnasiumEnvWrapper:
    """
    DEPRECATED: Wrapper for Gymnasium environments.

    Use wann_sdk.environments.BraxEnv instead for better performance.
    """

    def __init__(
        self,
        env_name: str,
        render_mode: Optional[str] = None,
        **env_kwargs
    ):
        warnings.warn(
            "GymnasiumEnvWrapper is deprecated. Use BraxEnv instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if not GYMNASIUM_AVAILABLE:
            raise ImportError("gymnasium is required. Install with: pip install gymnasium")

        self.env_name = env_name
        self.env = gym.make(env_name, render_mode=render_mode, **env_kwargs)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.obs_is_discrete = isinstance(
            self.observation_space,
            gym.spaces.Discrete
        )
        self.action_is_discrete = isinstance(
            self.action_space,
            gym.spaces.Discrete
        )

        self.obs_dim = self._get_space_dim(self.observation_space)
        self.action_dim = self._get_space_dim(self.action_space)

    def _get_space_dim(self, space) -> int:
        if isinstance(space, gym.spaces.Discrete):
            return space.n
        elif isinstance(space, gym.spaces.Box):
            return int(np.prod(space.shape))
        elif isinstance(space, gym.spaces.MultiDiscrete):
            return int(np.sum(space.nvec))
        else:
            raise ValueError(f"Unsupported space type: {type(space)}")

    def reset(self, seed: Optional[int] = None) -> Tuple[jnp.ndarray, Dict]:
        obs, info = self.env.reset(seed=seed)
        return jnp.array(obs.flatten()), info

    def step(self, action: jnp.ndarray) -> Tuple[jnp.ndarray, float, bool, bool, Dict]:
        if isinstance(action, jnp.ndarray):
            action = np.array(action)
        if self.action_is_discrete:
            action = int(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return jnp.array(obs.flatten()), reward, terminated, truncated, info

    def get_env_info(self) -> Dict[str, Any]:
        return {
            "env_name": self.env_name,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "obs_is_discrete": self.obs_is_discrete,
            "action_is_discrete": self.action_is_discrete,
        }

    def close(self):
        self.env.close()


class DistributedEnvPool:
    """
    DEPRECATED: Distributed environment pool using Ray.

    Use wann_sdk.environments.BraxEnv with batch_size > 1 instead.
    """

    def __init__(self, env_name: str, num_workers: int = 4, **env_kwargs):
        warnings.warn(
            "DistributedEnvPool is deprecated. "
            "Use BraxEnv(batch_size=N) instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if not RAY_AVAILABLE:
            raise ImportError("Ray is required. Install with: pip install ray")

        self.env_name = env_name
        self.num_workers = num_workers
        self.env_kwargs = env_kwargs

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Create reference env for info
        self._ref_env = GymnasiumEnvWrapper(env_name, **env_kwargs)

    def get_env_info(self) -> Dict[str, Any]:
        return self._ref_env.get_env_info()

    def close_all(self):
        self._ref_env.close()
