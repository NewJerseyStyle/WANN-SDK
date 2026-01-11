"""
Trainer for WANN SDK

A simple, flexible trainer that works with any Problem.
Inspired by PyTorch Lightning Trainer and HuggingFace Trainer.
"""

import jax
import jax.numpy as jnp
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import pickle
from pathlib import Path
import time

from .problem import Problem


@dataclass
class TrainerConfig:
    """
    Configuration for the Trainer.

    Args:
        pop_size: Population size for ES
        learning_rate: Learning rate for parameter updates
        noise_std: Standard deviation for parameter noise
        hidden_sizes: Hidden layer sizes [64, 64] creates 2 hidden layers
        activation: Activation function - 'tanh', 'relu', 'sigmoid'
        output_activation: Output activation - 'tanh', 'none', 'softmax'

    Example:
        >>> config = TrainerConfig(
        ...     pop_size=256,
        ...     hidden_sizes=[128, 64],
        ...     learning_rate=0.02,
        ... )
    """
    # ES hyperparameters
    pop_size: int = 256
    learning_rate: float = 0.01
    noise_std: float = 0.1

    # Network architecture
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = 'tanh'
    output_activation: str = 'tanh'

    # Training settings
    seed: int = 42
    save_best: bool = True
    verbose: bool = True


class Trainer:
    """
    Evolution Strategies trainer for any Problem.

    Simple interface inspired by PyTorch Lightning and HuggingFace.

    Example:
        >>> problem = SupervisedProblem(x_train, y_train)
        >>> trainer = Trainer(problem)
        >>> trainer.fit(generations=100)
        >>> metrics = trainer.evaluate()
        >>> trainer.save("model.pkl")
    """

    def __init__(
        self,
        problem: Problem,
        config: Optional[TrainerConfig] = None,
    ):
        """
        Initialize trainer.

        Args:
            problem: Problem instance defining the task
            config: TrainerConfig for hyperparameters
        """
        self.problem = problem
        self.config = config or TrainerConfig()

        # Build network architecture
        self.layer_sizes = [
            problem.input_dim,
            *self.config.hidden_sizes,
            problem.output_dim,
        ]

        # Initialize
        self.key = jax.random.PRNGKey(self.config.seed)
        self.params = self._init_params(self.key)

        # Get activation functions
        self._activation = self._get_activation(self.config.activation)
        self._output_activation = self._get_activation(self.config.output_activation)

        # Training state
        self.best_fitness = -float("inf")
        self.best_params: Optional[List[jnp.ndarray]] = None
        self.history: List[Dict[str, float]] = []
        self.is_fitted = False

    def _get_activation(self, name: str) -> Callable:
        """Get activation function by name."""
        activations = {
            'tanh': jnp.tanh,
            'relu': jax.nn.relu,
            'sigmoid': jax.nn.sigmoid,
            'softmax': lambda x: jax.nn.softmax(x, axis=-1),
            'none': lambda x: x,
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
        return activations[name]

    def _init_params(self, key: jax.random.PRNGKey) -> List[jnp.ndarray]:
        """Initialize network parameters with Xavier initialization."""
        params = []
        for i in range(len(self.layer_sizes) - 1):
            key, subkey = jax.random.split(key)
            scale = jnp.sqrt(2.0 / (self.layer_sizes[i] + self.layer_sizes[i + 1]))
            w = jax.random.normal(subkey, (self.layer_sizes[i], self.layer_sizes[i + 1])) * scale
            params.append(w)
        return params

    def _forward(self, params: List[jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the network."""
        for i, w in enumerate(params):
            x = x @ w
            if i < len(params) - 1:
                x = self._activation(x)
            else:
                x = self._output_activation(x)
        return x

    def _make_network(self, params: List[jnp.ndarray]) -> Callable:
        """Create a network function from parameters."""
        def network(x: jnp.ndarray) -> jnp.ndarray:
            return self._forward(params, x)
        return network

    def _es_step(
        self,
        params: List[jnp.ndarray],
        key: jax.random.PRNGKey,
    ) -> Tuple[List[jnp.ndarray], Dict[str, float]]:
        """Perform one ES update step."""
        fitness_list = []
        noise_list = []

        for _ in range(self.config.pop_size):
            key, eval_key, noise_key = jax.random.split(key, 3)

            # Generate noise for each parameter
            noise = [jax.random.normal(noise_key, w.shape) for w in params]

            # Positive perturbation
            pos_params = [w + self.config.noise_std * n for w, n in zip(params, noise)]
            pos_network = self._make_network(pos_params)
            pos_fitness = self.problem.evaluate(pos_network, eval_key)

            # Negative perturbation
            key, eval_key = jax.random.split(key)
            neg_params = [w - self.config.noise_std * n for w, n in zip(params, noise)]
            neg_network = self._make_network(neg_params)
            neg_fitness = self.problem.evaluate(neg_network, eval_key)

            fitness_list.append((pos_fitness, neg_fitness))
            noise_list.append(noise)

        # Compute gradient estimate
        fitness_array = jnp.array(fitness_list)
        fitness_diff = fitness_array[:, 0] - fitness_array[:, 1]

        # Normalize
        std = jnp.std(fitness_diff)
        fitness_normalized = (fitness_diff - jnp.mean(fitness_diff)) / (std + 1e-8)

        # Update parameters
        new_params = []
        for j, w in enumerate(params):
            grad = jnp.zeros_like(w)
            for i in range(self.config.pop_size):
                grad += fitness_normalized[i] * noise_list[i][j]
            grad /= self.config.pop_size * self.config.noise_std
            new_params.append(w + self.config.learning_rate * grad)

        # Metrics
        all_fitness = fitness_array.flatten()
        metrics = {
            "mean_fitness": float(jnp.mean(all_fitness)),
            "max_fitness": float(jnp.max(all_fitness)),
            "min_fitness": float(jnp.min(all_fitness)),
            "std_fitness": float(jnp.std(all_fitness)),
        }

        return new_params, metrics

    def fit(
        self,
        generations: int = 100,
        log_interval: int = 10,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, Any]:
        """
        Train the network using Evolution Strategies.

        Args:
            generations: Number of generations to train
            log_interval: How often to log progress
            callbacks: List of callback functions called each generation
                       callback(trainer, generation, metrics) -> bool
                       Return False to stop training early

        Returns:
            Dictionary with training results

        Example:
            >>> trainer.fit(generations=100)
            >>> trainer.fit(generations=50, log_interval=5)
        """
        # Setup
        self.problem.setup()
        callbacks = callbacks or []

        if self.config.verbose:
            print(f"Training for {generations} generations")
            print(f"Population size: {self.config.pop_size}")
            print(f"Network: {' -> '.join(map(str, self.layer_sizes))}")
            print("-" * 60)

        start_time = time.time()

        for gen in range(generations):
            self.key, step_key = jax.random.split(self.key)

            # ES update
            self.params, metrics = self._es_step(self.params, step_key)

            # Track best
            if metrics["max_fitness"] > self.best_fitness:
                self.best_fitness = metrics["max_fitness"]
                if self.config.save_best:
                    self.best_params = [w.copy() for w in self.params]

            # Log
            metrics["generation"] = gen
            metrics["elapsed_time"] = time.time() - start_time
            self.history.append(metrics)

            # Callbacks
            for callback in callbacks:
                if callback(self, gen, metrics) is False:
                    if self.config.verbose:
                        print(f"Early stopping at generation {gen}")
                    break

            # Print progress
            if self.config.verbose and (gen % log_interval == 0 or gen == generations - 1):
                elapsed = metrics["elapsed_time"]
                print(
                    f"Gen {gen:4d} [{elapsed:6.1f}s] | "
                    f"Mean: {metrics['mean_fitness']:10.4f} | "
                    f"Max: {metrics['max_fitness']:10.4f} | "
                    f"Best: {self.best_fitness:10.4f}"
                )

        total_time = time.time() - start_time
        self.is_fitted = True

        # Teardown
        self.problem.teardown()

        if self.config.verbose:
            print("-" * 60)
            print(f"Training completed in {total_time:.1f}s")
            print(f"Best fitness: {self.best_fitness:.4f}")

        return {
            "best_fitness": self.best_fitness,
            "final_fitness": metrics["mean_fitness"],
            "total_time": total_time,
            "generations": generations,
            "history": self.history,
        }

    def evaluate(self, num_evals: int = 10) -> Dict[str, float]:
        """
        Evaluate the trained network.

        Args:
            num_evals: Number of evaluations to average

        Returns:
            Dictionary with evaluation metrics
        """
        params = self.best_params if self.best_params else self.params
        network = self._make_network(params)

        fitness_list = []
        for _ in range(num_evals):
            self.key, eval_key = jax.random.split(self.key)
            fitness = self.problem.evaluate(network, eval_key)
            fitness_list.append(fitness)

        fitness_array = jnp.array(fitness_list)
        return {
            "mean_fitness": float(jnp.mean(fitness_array)),
            "std_fitness": float(jnp.std(fitness_array)),
            "max_fitness": float(jnp.max(fitness_array)),
            "min_fitness": float(jnp.min(fitness_array)),
        }

    def predict(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Make predictions with the trained network.

        Args:
            x: Input data, shape (n_samples, input_dim)

        Returns:
            Predictions, shape (n_samples, output_dim)
        """
        params = self.best_params if self.best_params else self.params
        return self._forward(params, x)

    def get_network(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """
        Get the trained network as a callable.

        Returns:
            Network function: (input) -> output
        """
        params = self.best_params if self.best_params else self.params
        return self._make_network(params)

    def save(self, path: str) -> None:
        """
        Save trained model.

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
                "activation": self.config.activation,
                "output_activation": self.config.output_activation,
                "seed": self.config.seed,
            },
            "best_fitness": self.best_fitness,
            "history": self.history,
            "input_dim": self.problem.input_dim,
            "output_dim": self.problem.output_dim,
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        if self.config.verbose:
            print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, problem: Optional[Problem] = None) -> "Trainer":
        """
        Load trained model.

        Args:
            path: File path to load from
            problem: Optional Problem instance (creates dummy if not provided)

        Returns:
            Loaded Trainer instance
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        # Create dummy problem if not provided
        if problem is None:
            problem = _DummyProblem(data["input_dim"], data["output_dim"])

        # Create config
        config = TrainerConfig(**data["config"])

        # Create trainer
        trainer = cls(problem, config)
        trainer.params = data["params"]
        trainer.best_params = data["best_params"]
        trainer.best_fitness = data["best_fitness"]
        trainer.history = data["history"]
        trainer.is_fitted = True

        return trainer

    def summary(self) -> str:
        """Get a summary of the trainer configuration."""
        lines = [
            "Trainer Summary",
            "=" * 40,
            f"Input dim:  {self.problem.input_dim}",
            f"Output dim: {self.problem.output_dim}",
            f"Network:    {' -> '.join(map(str, self.layer_sizes))}",
            f"Parameters: {sum(w.size for w in self.params):,}",
            "",
            "Config:",
            f"  pop_size:      {self.config.pop_size}",
            f"  learning_rate: {self.config.learning_rate}",
            f"  noise_std:     {self.config.noise_std}",
            f"  activation:    {self.config.activation}",
        ]
        if self.is_fitted:
            lines.extend([
                "",
                f"Trained: {len(self.history)} generations",
                f"Best fitness: {self.best_fitness:.4f}",
            ])
        return "\n".join(lines)


class _DummyProblem(Problem):
    """Dummy problem for loading saved models."""

    def evaluate(self, network, key):
        return 0.0
