"""
Environment Wrappers for WANN SDK

Provides high-performance JAX-native environments for WANN training:
- BraxEnv: Physics simulation (Ant, Humanoid, HalfCheetah, etc.)
- GymnaxEnv: Classic control (CartPole, Pendulum, MountainCar, etc.)

Both use JAX for automatic GPU acceleration.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass

# Brax imports
try:
    import brax
    from brax import envs as brax_envs
    BRAX_AVAILABLE = True
except ImportError:
    BRAX_AVAILABLE = False

# Gymnax imports
try:
    import gymnax
    from gymnax.environments import environment
    GYMNAX_AVAILABLE = True
except ImportError:
    GYMNAX_AVAILABLE = False


# Available environments
ENVIRONMENTS = {
    "humanoid": {
        "description": "Humanoid robot locomotion",
        "obs_dim": 244,
        "action_dim": 17,
    },
    "ant": {
        "description": "Ant robot locomotion",
        "obs_dim": 87,
        "action_dim": 8,
    },
    "halfcheetah": {
        "description": "Half cheetah locomotion",
        "obs_dim": 18,
        "action_dim": 6,
    },
    "hopper": {
        "description": "Hopper locomotion",
        "obs_dim": 11,
        "action_dim": 3,
    },
    "walker2d": {
        "description": "Walker 2D locomotion",
        "obs_dim": 17,
        "action_dim": 6,
    },
    "inverted_pendulum": {
        "description": "Inverted pendulum balance",
        "obs_dim": 4,
        "action_dim": 1,
    },
    "inverted_double_pendulum": {
        "description": "Inverted double pendulum balance",
        "obs_dim": 9,
        "action_dim": 1,
    },
    "reacher": {
        "description": "Reacher arm control",
        "obs_dim": 11,
        "action_dim": 2,
    },
    "pusher": {
        "description": "Pusher arm control",
        "obs_dim": 23,
        "action_dim": 7,
    },
}


def list_environments() -> Dict[str, Dict[str, Any]]:
    """
    List all available environments.

    Returns:
        Dictionary mapping environment names to their info

    Example:
        >>> envs = list_environments()
        >>> for name, info in envs.items():
        ...     print(f"{name}: {info['description']}")
    """
    return ENVIRONMENTS.copy()


class BraxEnv:
    """
    High-performance Brax environment wrapper.

    Provides a unified interface for Brax environments with support for:
    - Single environment interaction
    - Batched (vectorized) environments
    - Automatic GPU acceleration via JAX

    Args:
        env_name: Name of the environment (e.g., 'humanoid', 'ant')
        batch_size: Number of parallel environments
        episode_length: Maximum steps per episode
        seed: Random seed

    Example:
        >>> env = BraxEnv("humanoid", batch_size=32)
        >>> obs, state = env.reset(jax.random.PRNGKey(0))
        >>> action = jnp.zeros(env.action_dim)
        >>> obs, state, reward, done, info = env.step(state, action)
    """

    def __init__(
        self,
        env_name: str,
        batch_size: int = 1,
        episode_length: int = 1000,
        seed: int = 0,
    ):
        if not BRAX_AVAILABLE:
            raise ImportError(
                "Brax is required for BraxEnv. "
                "Install with: pip install wann-sdk[brax]"
            )

        if env_name not in ENVIRONMENTS:
            available = ", ".join(ENVIRONMENTS.keys())
            raise ValueError(
                f"Unknown environment: {env_name}. "
                f"Available: {available}"
            )

        self.env_name = env_name
        self.batch_size = batch_size
        self.episode_length = episode_length
        self.seed = seed

        # Create Brax environment
        self.env = brax_envs.get_environment(env_name)
        self.obs_dim = self.env.observation_size
        self.action_dim = self.env.action_size

        # JIT compile core functions
        self._reset_fn = jit(self._reset_impl)
        self._step_fn = jit(self._step_impl)
        self._batch_reset_fn = jit(vmap(self._reset_impl))
        self._batch_step_fn = jit(vmap(self._step_impl))

    def _reset_impl(self, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, Any]:
        """Reset implementation for single environment."""
        state = self.env.reset(key)
        return state.obs, state

    def _step_impl(
        self,
        state: Any,
        action: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, Any, jnp.ndarray, jnp.ndarray, Dict]:
        """Step implementation for single environment."""
        next_state = self.env.step(state, action)
        return (
            next_state.obs,
            next_state,
            next_state.reward,
            next_state.done,
            {"truncation": next_state.info.get("truncation", False)},
        )

    def reset(
        self, key: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[jnp.ndarray, Any]:
        """
        Reset environment(s).

        Args:
            key: Random key (uses self.seed if None)

        Returns:
            obs: Observation(s)
            state: Environment state(s)
        """
        if key is None:
            key = jax.random.PRNGKey(self.seed)

        if self.batch_size == 1:
            return self._reset_fn(key)
        else:
            keys = jax.random.split(key, self.batch_size)
            return self._batch_reset_fn(keys)

    def step(
        self,
        state: Any,
        action: jnp.ndarray,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Tuple[jnp.ndarray, Any, jnp.ndarray, jnp.ndarray, Dict]:
        """
        Step environment(s).

        Args:
            state: Current state(s)
            action: Action(s) to take
            key: Random key

        Returns:
            obs: Next observation(s)
            next_state: Next state(s)
            reward: Reward(s)
            done: Done flag(s)
            info: Additional info
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        if self.batch_size == 1:
            return self._step_fn(state, action, key)
        else:
            keys = jax.random.split(key, self.batch_size)
            return self._batch_step_fn(state, action, keys)

    def get_info(self) -> Dict[str, Any]:
        """
        Get environment information.

        Returns:
            Dictionary with environment details
        """
        return {
            "env_name": self.env_name,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "batch_size": self.batch_size,
            "episode_length": self.episode_length,
            "description": ENVIRONMENTS[self.env_name]["description"],
        }

    def rollout(
        self,
        policy_fn: Callable,
        policy_params: Any,
        key: jax.random.PRNGKey,
        num_steps: Optional[int] = None,
    ) -> Dict[str, jnp.ndarray]:
        """
        Collect a complete rollout.

        Args:
            policy_fn: Policy function (params, obs) -> action
            policy_params: Policy parameters
            key: Random key
            num_steps: Number of steps (uses episode_length if None)

        Returns:
            Dictionary with rollout data
        """
        num_steps = num_steps or self.episode_length

        def scan_fn(carry, _):
            state, obs, key, done = carry

            key, subkey = jax.random.split(key)

            # Get action from policy
            action = policy_fn(policy_params, obs)

            # Step environment
            next_obs, next_state, reward, next_done, info = self.step(
                state, action, subkey
            )

            # Auto-reset on done
            key, reset_key = jax.random.split(key)
            reset_obs, reset_state = self.reset(reset_key)

            next_obs = jnp.where(next_done, reset_obs, next_obs)
            next_state = jax.tree.map(
                lambda r, n: jnp.where(next_done, r, n),
                reset_state,
                next_state,
            )

            return (next_state, next_obs, key, next_done), (obs, action, reward, done)

        # Initialize
        key, reset_key = jax.random.split(key)
        obs, state = self.reset(reset_key)
        done = jnp.zeros(
            self.batch_size if self.batch_size > 1 else (), dtype=bool
        )

        # Run rollout
        _, (observations, actions, rewards, dones) = jax.lax.scan(
            scan_fn,
            (state, obs, key, done),
            None,
            length=num_steps,
        )

        return {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "total_reward": jnp.sum(rewards, axis=0),
        }

    def __repr__(self) -> str:
        return (
            f"BraxEnv('{self.env_name}', "
            f"obs_dim={self.obs_dim}, action_dim={self.action_dim}, "
            f"batch_size={self.batch_size})"
        )


# Gymnax environments
GYMNAX_ENVIRONMENTS = {
    "CartPole-v1": {
        "description": "Classic cart-pole balance",
        "obs_dim": 4,
        "action_dim": 2,
        "continuous": False,
    },
    "Pendulum-v1": {
        "description": "Inverted pendulum swing-up",
        "obs_dim": 3,
        "action_dim": 1,
        "continuous": True,
    },
    "MountainCar-v0": {
        "description": "Mountain car control",
        "obs_dim": 2,
        "action_dim": 3,
        "continuous": False,
    },
    "MountainCarContinuous-v0": {
        "description": "Mountain car (continuous)",
        "obs_dim": 2,
        "action_dim": 1,
        "continuous": True,
    },
    "Acrobot-v1": {
        "description": "Acrobot swing-up",
        "obs_dim": 6,
        "action_dim": 3,
        "continuous": False,
    },
    "Asterix-MinAtar": {
        "description": "MinAtar Asterix game",
        "obs_dim": 100,  # 10x10 with channels flattened
        "action_dim": 5,
        "continuous": False,
    },
    "Breakout-MinAtar": {
        "description": "MinAtar Breakout game",
        "obs_dim": 100,
        "action_dim": 3,
        "continuous": False,
    },
    "SpaceInvaders-MinAtar": {
        "description": "MinAtar Space Invaders",
        "obs_dim": 100,
        "action_dim": 4,
        "continuous": False,
    },
}


def list_gymnax_environments() -> Dict[str, Dict[str, Any]]:
    """
    List all available Gymnax environments.

    Returns:
        Dictionary mapping environment names to their info

    Example:
        >>> envs = list_gymnax_environments()
        >>> for name, info in envs.items():
        ...     print(f"{name}: {info['description']}")
    """
    return GYMNAX_ENVIRONMENTS.copy()


class GymnaxEnv:
    """
    JAX-native Gymnax environment wrapper.

    Provides classic control and MinAtar environments with:
    - Pure JAX implementation (no Python loops)
    - Automatic GPU/TPU acceleration
    - Vectorized environment support

    Args:
        env_name: Name of the environment (e.g., 'CartPole-v1', 'Pendulum-v1')
        batch_size: Number of parallel environments
        seed: Random seed

    Example:
        >>> env = GymnaxEnv("CartPole-v1")
        >>> obs, state = env.reset(jax.random.PRNGKey(0))
        >>> action = 0  # Discrete action
        >>> obs, state, reward, done, info = env.step(state, action)

    Note:
        Install gymnax with: pip install gymnax
    """

    def __init__(
        self,
        env_name: str,
        batch_size: int = 1,
        seed: int = 0,
    ):
        if not GYMNAX_AVAILABLE:
            raise ImportError(
                "Gymnax is required for GymnaxEnv. "
                "Install with: pip install gymnax"
            )

        self.env_name = env_name
        self.batch_size = batch_size
        self.seed = seed

        # Create Gymnax environment
        self.env, self.env_params = gymnax.make(env_name)

        # Get dimensions
        self.obs_dim = self.env.observation_space(self.env_params).shape[0]

        # Handle discrete vs continuous action spaces
        action_space = self.env.action_space(self.env_params)
        if hasattr(action_space, 'n'):
            # Discrete action space
            self.action_dim = action_space.n
            self.continuous = False
        else:
            # Continuous action space
            self.action_dim = action_space.shape[0]
            self.continuous = True

        # JIT compile core functions
        self._reset_fn = jit(self._reset_impl)
        self._step_fn = jit(self._step_impl)
        if batch_size > 1:
            self._batch_reset_fn = jit(vmap(self._reset_impl))
            self._batch_step_fn = jit(vmap(self._step_impl, in_axes=(0, 0, 0)))

    def _reset_impl(self, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, Any]:
        """Reset implementation for single environment."""
        obs, state = self.env.reset(key, self.env_params)
        return obs, state

    def _step_impl(
        self,
        state: Any,
        action: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, Any, jnp.ndarray, jnp.ndarray, Dict]:
        """Step implementation for single environment."""
        # Handle action format
        if not self.continuous:
            # Discrete: convert network output to action index
            if action.shape != ():
                action = jnp.argmax(action)

        obs, state, reward, done, info = self.env.step(
            key, state, action, self.env_params
        )
        return obs, state, reward, done, info

    def reset(
        self, key: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[jnp.ndarray, Any]:
        """
        Reset environment(s).

        Args:
            key: Random key (uses self.seed if None)

        Returns:
            obs: Observation(s)
            state: Environment state(s)
        """
        if key is None:
            key = jax.random.PRNGKey(self.seed)

        if self.batch_size == 1:
            return self._reset_fn(key)
        else:
            keys = jax.random.split(key, self.batch_size)
            return self._batch_reset_fn(keys)

    def step(
        self,
        state: Any,
        action: jnp.ndarray,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Tuple[jnp.ndarray, Any, jnp.ndarray, jnp.ndarray, Dict]:
        """
        Step environment(s).

        Args:
            state: Current state(s)
            action: Action(s) to take
            key: Random key

        Returns:
            obs: Next observation(s)
            next_state: Next state(s)
            reward: Reward(s)
            done: Done flag(s)
            info: Additional info
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        if self.batch_size == 1:
            return self._step_fn(state, action, key)
        else:
            keys = jax.random.split(key, self.batch_size)
            return self._batch_step_fn(state, action, keys)

    def get_info(self) -> Dict[str, Any]:
        """
        Get environment information.

        Returns:
            Dictionary with environment details
        """
        return {
            "env_name": self.env_name,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "batch_size": self.batch_size,
            "continuous": self.continuous,
        }

    def __repr__(self) -> str:
        action_type = "continuous" if self.continuous else "discrete"
        return (
            f"GymnaxEnv('{self.env_name}', "
            f"obs_dim={self.obs_dim}, action_dim={self.action_dim}, "
            f"{action_type}, batch_size={self.batch_size})"
        )
