#!/usr/bin/env python3
"""
WANN Architecture Search Example

Demonstrates the full Weight Agnostic Neural Network pipeline:
1. Architecture Search - Evolve topology with shared weights
2. Evaluate found architecture across different weight values

This is the core WANN methodology from Gaier & Ha (2019).

Usage:
    python wann_search.py
    python wann_search.py --generations 200 --max_nodes 30
"""

import argparse
import jax
import jax.numpy as jnp

from wann_sdk import (
    ArchitectureSearch,
    SearchConfig,
    SupervisedProblem,
    Problem,
)


def search_xor():
    """Search for XOR architecture."""
    print("=" * 60)
    print("WANN Architecture Search: XOR Problem")
    print("=" * 60)

    class XORProblem(Problem):
        def __init__(self):
            super().__init__(input_dim=2, output_dim=1)
            self.x = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.float32)
            self.y = jnp.array([[0], [1], [1], [0]], dtype=jnp.float32)

        def evaluate(self, network, key):
            """For ES and architecture search."""
            pred = jax.nn.sigmoid(network(self.x))
            mse = jnp.mean((pred - self.y) ** 2)
            return -float(mse)

        def loss(self, network, key):
            """For gradient-based optimizers."""
            pred = jax.nn.sigmoid(network(self.x))
            return jnp.mean((pred - self.y) ** 2)

    problem = XORProblem()

    config = SearchConfig(
        pop_size=50,
        max_nodes=10,
        max_connections=30,
        activation_options=['tanh', 'relu', 'sigmoid', 'step'],
        weight_values=[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
        complexity_weight=0.01,
        add_node_rate=0.1,
        add_connection_rate=0.2,
    )

    search = ArchitectureSearch(problem, config)
    best_genome = search.run(generations=100, log_interval=20)

    # Test with different weight values
    print("\nEvaluating best architecture with different weights:")
    for weight in [-2.0, -1.0, 0.5, 1.0, 2.0]:
        network = search.genome_to_network(best_genome, weight)
        pred = jax.nn.sigmoid(network(problem.x))
        accuracy = jnp.mean((pred > 0.5) == problem.y)
        print(f"  Weight {weight:5.1f}: Accuracy = {float(accuracy) * 100:.1f}%")

    print(f"\nArchitecture found:")
    print(f"  Hidden nodes: {int(jnp.sum(best_genome.nodes[:, 1] == 1))}")
    print(f"  Connections: {int(jnp.sum(best_genome.connections[:, 2] == 1))}")


def search_classification():
    """Search for classification architecture."""
    print("\n" + "=" * 60)
    print("WANN Architecture Search: Classification Problem")
    print("=" * 60)

    # Generate synthetic classification data
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    # Two clusters
    n_samples = 100
    x1 = jax.random.normal(k1, (n_samples // 2, 4)) + jnp.array([2, 2, 0, 0])
    x2 = jax.random.normal(k2, (n_samples // 2, 4)) + jnp.array([-2, -2, 0, 0])
    x_train = jnp.concatenate([x1, x2], axis=0)
    y_train = jnp.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    problem = SupervisedProblem(
        x_train, y_train,
        loss_fn='cross_entropy',
        batch_size=50,
    )

    config = SearchConfig(
        pop_size=30,
        max_nodes=15,
        max_connections=50,
        activation_options=['tanh', 'relu', 'sigmoid'],
        weight_values=[-1.0, -0.5, 0.5, 1.0],
        complexity_weight=0.05,
    )

    search = ArchitectureSearch(problem, config)
    best_genome = search.run(generations=50, log_interval=10)

    # Evaluate accuracy
    print("\nEvaluating best architecture:")
    for weight in [-1.0, 0.5, 1.0]:
        network = search.genome_to_network(best_genome, weight)
        pred = network(x_train)
        pred_labels = jnp.argmax(pred, axis=-1)
        accuracy = jnp.mean(pred_labels == y_train)
        print(f"  Weight {weight:5.1f}: Accuracy = {float(accuracy) * 100:.1f}%")


def search_regression():
    """Search for regression architecture."""
    print("\n" + "=" * 60)
    print("WANN Architecture Search: Sine Regression")
    print("=" * 60)

    class SineProblem(Problem):
        def __init__(self):
            super().__init__(input_dim=1, output_dim=1)
            self.x = jnp.linspace(-jnp.pi, jnp.pi, 50).reshape(-1, 1)
            self.y = jnp.sin(self.x)

        def evaluate(self, network, key):
            """For ES and architecture search."""
            pred = network(self.x)
            mse = jnp.mean((pred - self.y) ** 2)
            return -float(mse)

        def loss(self, network, key):
            """For gradient-based optimizers."""
            pred = network(self.x)
            return jnp.mean((pred - self.y) ** 2)

    problem = SineProblem()

    config = SearchConfig(
        pop_size=40,
        max_nodes=20,
        activation_options=['tanh', 'sin', 'identity', 'gaussian'],
        weight_values=[-1.0, 1.0],  # Fewer weights for regression
        complexity_weight=0.02,
    )

    search = ArchitectureSearch(problem, config)
    best_genome = search.run(generations=80, log_interval=20)

    # Evaluate MSE
    print("\nEvaluating best architecture:")
    for weight in [-1.0, 0.5, 1.0, 2.0]:
        network = search.genome_to_network(best_genome, weight)
        pred = network(problem.x)
        mse = jnp.mean((pred - problem.y) ** 2)
        print(f"  Weight {weight:5.1f}: MSE = {float(mse):.4f}")


def main():
    parser = argparse.ArgumentParser(description="WANN Architecture Search")
    parser.add_argument("--task", choices=['xor', 'classification', 'regression', 'all'],
                        default='all', help="Task to run")
    args = parser.parse_args()

    if args.task == 'xor' or args.task == 'all':
        search_xor()

    if args.task == 'classification' or args.task == 'all':
        search_classification()

    if args.task == 'regression' or args.task == 'all':
        search_regression()

    print("\n" + "=" * 60)
    print("WANN Architecture Search Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
