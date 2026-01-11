#!/usr/bin/env python3
"""
Custom Problem Example

Demonstrates how to create your own Problem class using the two-stage WANN pipeline.

Usage:
    python custom_problem.py
"""

import jax
import jax.numpy as jnp

from wann_sdk import (
    Problem,
    ArchitectureSearch, SearchConfig,
    WeightTrainer, WeightTrainerConfig,
)


class XORProblem(Problem):
    """
    Classic XOR problem for demonstration.

    Shows how to implement a custom Problem class with both
    evaluate() for ES and loss() for gradient-based optimizers.
    """

    def __init__(self):
        super().__init__(input_dim=2, output_dim=1)

        self.x = jnp.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ], dtype=jnp.float32)

        self.y = jnp.array([
            [0],
            [1],
            [1],
            [0],
        ], dtype=jnp.float32)

    def evaluate(self, network, key):
        """For ES optimizer and architecture search - returns Python float."""
        predictions = jax.nn.sigmoid(network(self.x))
        mse = jnp.mean((predictions - self.y) ** 2)
        return -float(mse)

    def loss(self, network, key):
        """For gradient-based optimizers (SGD/Adam/AdamW) - returns JAX array."""
        predictions = jax.nn.sigmoid(network(self.x))
        return jnp.mean((predictions - self.y) ** 2)


class RegressionProblem(Problem):
    """
    Custom regression problem.

    Learn to fit a sine wave with noise.
    """

    def __init__(self, n_samples: int = 100, noise: float = 0.1):
        super().__init__(input_dim=1, output_dim=1)

        key = jax.random.PRNGKey(0)
        self.x = jnp.linspace(-jnp.pi, jnp.pi, n_samples).reshape(-1, 1)
        self.y = jnp.sin(self.x) + noise * jax.random.normal(key, self.x.shape)

    def evaluate(self, network, key):
        """For ES optimizer - returns Python float."""
        predictions = network(self.x)
        mse = jnp.mean((predictions - self.y) ** 2)
        return -float(mse)

    def loss(self, network, key):
        """For gradient-based optimizers - returns JAX array."""
        predictions = network(self.x)
        return jnp.mean((predictions - self.y) ** 2)


def train_xor():
    """Two-stage pipeline on XOR problem."""
    print("=" * 60)
    print("XOR Problem - Two Stage Pipeline")
    print("=" * 60)

    problem = XORProblem()

    # Stage 1: Architecture Search
    print("\n--- Stage 1: Architecture Search ---")
    search_config = SearchConfig(
        pop_size=30,
        max_nodes=8,
        activation_options=['tanh', 'relu', 'sigmoid', 'step'],
        weight_values=[-1.0, 1.0],
    )

    search = ArchitectureSearch(problem, search_config)
    genome = search.run(generations=50, log_interval=10)

    print(f"\nArchitecture: {int(jnp.sum(genome.nodes[:, 1] == 1))} hidden nodes")

    # Stage 2: Weight Training with Adam
    print("\n--- Stage 2: Weight Training (Adam) ---")
    trainer_config = WeightTrainerConfig(
        optimizer='adam',
        learning_rate=0.05,
    )

    trainer = WeightTrainer(
        genome, problem, trainer_config,
        activation_options=search_config.activation_options,
    )
    trainer.fit(epochs=50, log_interval=10)

    # Test predictions
    network = trainer.get_network()
    predictions = jax.nn.sigmoid(network(problem.x))

    print("\nXOR Predictions:")
    for i in range(4):
        print(f"  {int(problem.x[i][0])} XOR {int(problem.x[i][1])} = "
              f"{predictions[i][0]:.3f} (expected: {int(problem.y[i][0])})")


def train_regression():
    """Two-stage pipeline on regression problem."""
    print("\n" + "=" * 60)
    print("Sine Regression - Two Stage Pipeline")
    print("=" * 60)

    problem = RegressionProblem(n_samples=50, noise=0.1)

    # Stage 1: Architecture Search
    print("\n--- Stage 1: Architecture Search ---")
    search_config = SearchConfig(
        pop_size=30,
        max_nodes=10,
        activation_options=['tanh', 'sin', 'identity'],
        weight_values=[-1.0, 1.0],
    )

    search = ArchitectureSearch(problem, search_config)
    genome = search.run(generations=40, log_interval=10)

    # Stage 2: Weight Training with AdamW
    print("\n--- Stage 2: Weight Training (AdamW) ---")
    trainer_config = WeightTrainerConfig(
        optimizer='adamw',
        learning_rate=0.02,
        weight_decay=0.01,
    )

    trainer = WeightTrainer(
        genome, problem, trainer_config,
        activation_options=search_config.activation_options,
    )
    trainer.fit(epochs=50, log_interval=10)

    # Evaluate
    network = trainer.get_network()
    predictions = network(problem.x)
    mse = float(jnp.mean((predictions - problem.y) ** 2))
    print(f"\nFinal MSE: {mse:.4f}")


def main():
    train_xor()
    train_regression()

    print("\n" + "=" * 60)
    print("Custom problems demonstrated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
