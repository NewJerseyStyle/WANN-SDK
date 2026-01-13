"""
Training Methods for WANN

Provides Evolution Strategies (ES) based training for neural networks.
Fully JAX-native for GPU acceleration.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
import pickle
from pathlib import Path
import time

from .environments import BraxEnv


@dataclass
class TrainingConfig:
    """
    Configuration for ES training.

    Args:
        pop_size: Population size for ES
        learning_rate: Learning rate for parameter updates
        noise_std: Standard deviation for parameter noise
        hidden_sizes: Hidden layer sizes for the network
        max_episode_steps: Maximum steps per episode
        num_eval_episodes: Number of episodes for evaluation

    Example:
        >>> config = TrainingConfig(
        ...     pop_size=256,
        ...     learning_rate=0.01,
        ...     noise_std=0.1,
        ... )
    """

    pop_size: int = 256
    learning_rate: float = 0.01
    noise_std: float = 0.1
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])
    max_episode_steps: int = 1000
    num_eval_episodes: int = 5


class ESTrainer:
    """
    Evolution Strategies trainer for neural networks.

    Uses OpenAI-style Evolution Strategies for training neural network
    policies on Brax environments.

    Args:
        env: BraxEnv instance
        config: TrainingConfig instance
        seed: Random seed

    Example:
        >>> env = BraxEnv("humanoid")
        >>> trainer = ESTrainer(env)
        >>> results = trainer.train(generations=100)
        >>> trainer.save("model.pkl")
    """

    def __init__(
        self,
        env: BraxEnv,
        config: Optional[TrainingConfig] = None,
        seed: int = 42,
    ):
        self.env = env
        self.config = config or TrainingConfig()
        self.seed = seed

        # Build network architecture
        self.layer_sizes = [
            env.obs_dim,
            *self.config.hidden_sizes,
            env.action_dim,
        ]

        # Initialize parameters
        self.key = jax.random.PRNGKey(seed)
        self.params = self._init_params(self.key)

        # Track training progress
        self.best_fitness = -float("inf")
        self.best_params = None
        self.history: List[Dict[str, float]] = []

    def _init_params(self, key: jax.random.PRNGKey) -> List[jnp.ndarray]:
        """Initialize network parameters."""
        params = []
        for i in range(len(self.layer_sizes) - 1):
            key, subkey = jax.random.split(key)
            # Xavier initialization
            scale = jnp.sqrt(2.0 / (self.layer_sizes[i] + self.layer_sizes[i + 1]))
            w = jax.random.normal(
                subkey, (self.layer_sizes[i], self.layer_sizes[i + 1])
            ) * scale
            params.append(w)
        return params

    def _forward(self, params: List[jnp.ndarray], obs: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the network."""
        x = obs
        for i, w in enumerate(params):
            x = x @ w
            if i < len(params) - 1:
                x = jnp.tanh(x)
        return jnp.tanh(x)  # Output in [-1, 1]

    def _evaluate(
        self,
        params: List[jnp.ndarray],
        key: jax.random.PRNGKey,
    ) -> float:
        """Evaluate parameters on environment."""
        obs, env_state = self.env.reset(key)
        total_reward = 0.0

        for _ in range(self.config.max_episode_steps):
            key, step_key = jax.random.split(key)
            action = self._forward(params, obs)
            obs, env_state, reward, done, _ = self.env.step(
                env_state, action, step_key
            )
            total_reward += float(jnp.mean(reward))

        return total_reward

    def _es_step(
        self,
        params: List[jnp.ndarray],
        key: jax.random.PRNGKey,
    ) -> Tuple[List[jnp.ndarray], Dict[str, float]]:
        """Perform one ES update step."""
        fitness_list = []
        noise_list = []

        # Evaluate population
        for _ in range(self.config.pop_size):
            key, eval_key, noise_key = jax.random.split(key, 3)

            # Generate noise
            noise = [
                jax.random.normal(noise_key, w.shape) for w in params
            ]

            # Evaluate positive perturbation
            pos_params = [
                w + self.config.noise_std * n for w, n in zip(params, noise)
            ]
            pos_fitness = self._evaluate(pos_params, eval_key)

            # Evaluate negative perturbation
            key, eval_key = jax.random.split(key)
            neg_params = [
                w - self.config.noise_std * n for w, n in zip(params, noise)
            ]
            neg_fitness = self._evaluate(neg_params, eval_key)

            fitness_list.append((pos_fitness, neg_fitness))
            noise_list.append(noise)

        # Compute gradient estimate
        fitness_array = jnp.array(fitness_list)
        fitness_diff = fitness_array[:, 0] - fitness_array[:, 1]

        # Normalize
        fitness_normalized = (fitness_diff - jnp.mean(fitness_diff)) / (
            jnp.std(fitness_diff) + 1e-8
        )

        # Update parameters
        new_params = []
        for j, w in enumerate(params):
            grad = jnp.zeros_like(w)
            for i in range(self.config.pop_size):
                grad += fitness_normalized[i] * noise_list[i][j]
            grad /= self.config.pop_size * self.config.noise_std
            new_params.append(w + self.config.learning_rate * grad)

        # Compute metrics
        all_fitness = fitness_array.flatten()
        metrics = {
            "mean_fitness": float(jnp.mean(all_fitness)),
            "max_fitness": float(jnp.max(all_fitness)),
            "min_fitness": float(jnp.min(all_fitness)),
            "std_fitness": float(jnp.std(all_fitness)),
        }

        return new_params, metrics

    def train(
        self,
        generations: int = 100,
        log_interval: int = 10,
        save_best: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the network using Evolution Strategies.

        Args:
            generations: Number of generations to train
            log_interval: How often to log progress
            save_best: Whether to save best parameters
            verbose: Whether to print progress

        Returns:
            Dictionary with training results

        Example:
            >>> trainer = ESTrainer(env)
            >>> results = trainer.train(generations=100)
            >>> print(f"Best fitness: {results['best_fitness']}")
        """
        if verbose:
            print(f"Training for {generations} generations")
            print(f"Population size: {self.config.pop_size}")
            print(f"Network: {self.layer_sizes}")
            print("-" * 50)

        start_time = time.time()

        for gen in range(generations):
            self.key, step_key = jax.random.split(self.key)

            # ES update
            self.params, metrics = self._es_step(self.params, step_key)

            # Track best
            if metrics["max_fitness"] > self.best_fitness:
                self.best_fitness = metrics["max_fitness"]
                if save_best:
                    self.best_params = [w.copy() for w in self.params]

            # Log
            self.history.append({
                "generation": gen,
                **metrics,
            })

            if verbose and (gen % log_interval == 0 or gen == generations - 1):
                elapsed = time.time() - start_time
                print(
                    f"Gen {gen:4d} [{elapsed:6.1f}s] | "
                    f"Mean: {metrics['mean_fitness']:8.2f} | "
                    f"Max: {metrics['max_fitness']:8.2f} | "
                    f"Best: {self.best_fitness:8.2f}"
                )

        total_time = time.time() - start_time

        if verbose:
            print("-" * 50)
            print(f"Training completed in {total_time:.1f}s")
            print(f"Best fitness: {self.best_fitness:.2f}")

        return {
            "best_fitness": self.best_fitness,
            "final_fitness": metrics["mean_fitness"],
            "total_time": total_time,
            "generations": generations,
            "history": self.history,
        }

    def evaluate(
        self,
        num_episodes: int = 10,
        use_best: bool = True,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate the trained policy.

        Args:
            num_episodes: Number of evaluation episodes
            use_best: Whether to use best parameters
            verbose: Whether to print results

        Returns:
            Dictionary with evaluation metrics
        """
        params = self.best_params if use_best and self.best_params else self.params

        rewards = []
        for ep in range(num_episodes):
            self.key, eval_key = jax.random.split(self.key)
            reward = self._evaluate(params, eval_key)
            rewards.append(reward)

            if verbose:
                print(f"Episode {ep + 1}: {reward:.2f}")

        rewards_array = jnp.array(rewards)
        results = {
            "mean_reward": float(jnp.mean(rewards_array)),
            "std_reward": float(jnp.std(rewards_array)),
            "max_reward": float(jnp.max(rewards_array)),
            "min_reward": float(jnp.min(rewards_array)),
        }

        if verbose:
            print("-" * 30)
            print(f"Mean: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")

        return results

    def save(self, path: str) -> None:
        """
        Save trained model to file.

        Args:
            path: File path to save to
        """
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "params": self.params,
            "best_params": self.best_params,
            "layer_sizes": self.layer_sizes,
            "config": {
                "pop_size": self.config.pop_size,
                "learning_rate": self.config.learning_rate,
                "noise_std": self.config.noise_std,
                "hidden_sizes": self.config.hidden_sizes,
                "max_episode_steps": self.config.max_episode_steps,
            },
            "best_fitness": self.best_fitness,
            "history": self.history,
            "env_name": self.env.env_name,
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, env: Optional[BraxEnv] = None) -> "ESTrainer":
        """
        Load trained model from file.

        Args:
            path: File path to load from
            env: Optional BraxEnv instance

        Returns:
            Loaded ESTrainer instance
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        # Create environment if not provided
        if env is None:
            env = BraxEnv(data["env_name"])

        # Create config
        config = TrainingConfig(**data["config"])

        # Create trainer
        trainer = cls(env, config)
        trainer.params = data["params"]
        trainer.best_params = data["best_params"]
        trainer.best_fitness = data["best_fitness"]
        trainer.history = data["history"]

        return trainer

    def get_policy(self) -> Callable:
        """
        Get the trained policy function.

        Returns:
            Policy function (obs) -> action
        """
        params = self.best_params if self.best_params else self.params

        def policy(obs: jnp.ndarray) -> jnp.ndarray:
            return self._forward(params, obs)

        return policy
