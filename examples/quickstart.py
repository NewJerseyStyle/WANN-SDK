#!/usr/bin/env python3
"""
WANN SDK Quickstart

Minimal example of the two-stage WANN pipeline.

Usage:
    python quickstart.py
"""

import jax
import jax.numpy as jnp

from wann_sdk import (
    # Stage 1
    ArchitectureSearch,
    SearchConfig,
    # Stage 2
    WeightTrainer,
    WeightTrainerConfig,
    # Problem
    Problem,
)


class XORProblem(Problem):
    """
    Simple XOR problem for demonstration.

    Implements both evaluate() for ES and loss() for gradient-based optimizers.
    """

    def __init__(self):
        super().__init__(input_dim=2, output_dim=1)
        self.x = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.float32)
        self.y = jnp.array([[0], [1], [1], [0]], dtype=jnp.float32)

    def evaluate(self, network, key):
        """For ES optimizer - returns Python float."""
        pred = jax.nn.sigmoid(network(self.x))
        mse = jnp.mean((pred - self.y) ** 2)
        return -float(mse)  # Negative loss as fitness

    def loss(self, network, key):
        """For gradient-based optimizers - returns JAX array."""
        pred = jax.nn.sigmoid(network(self.x))
        return jnp.mean((pred - self.y) ** 2)  # No float()!


def main():
    print("WANN SDK Quickstart")
    print("=" * 50)

    problem = XORProblem()

    # ==========================================
    # Stage 1: Architecture Search
    # ==========================================
    print("\n--- Stage 1: Architecture Search ---")

    search_config = SearchConfig(
        pop_size=30,
        max_nodes=8,
        activation_options=['tanh', 'relu', 'sigmoid'],
        weight_values=[-1.0, 1.0],
    )

    search = ArchitectureSearch(problem, search_config)
    genome = search.run(generations=50, log_interval=10)

    print(f"\nFound architecture with {int(jnp.sum(genome.nodes[:, 1] == 1))} hidden nodes")

    # ==========================================
    # Stage 2: Weight Training
    # ==========================================
    print("\n--- Stage 2: Weight Training (Adam) ---")

    trainer_config = WeightTrainerConfig(
        optimizer='adam',  # Uses loss() method
        learning_rate=0.05,
    )

    trainer = WeightTrainer(
        genome, problem, trainer_config,
        activation_options=search_config.activation_options,
    )
    trainer.fit(epochs=50, log_interval=10)

    # Test
    network = trainer.get_network()
    predictions = jax.nn.sigmoid(network(problem.x))

    print("\nResults:")
    for i in range(4):
        x1, x2 = int(problem.x[i, 0]), int(problem.x[i, 1])
        pred = float(predictions[i, 0])
        expected = int(problem.y[i, 0])
        print(f"  {x1} XOR {x2} = {pred:.3f} (expected: {expected})")

    print("\n" + "=" * 50)
    print("Quickstart complete!")


if __name__ == "__main__":
    main()
