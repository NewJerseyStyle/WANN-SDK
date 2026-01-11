#!/usr/bin/env python3
"""
WANN SDK Full Pipeline Example

Demonstrates the standard two-stage WANN workflow:
  Stage 1: Architecture Search - Find topology with shared weights
  Stage 2: Weight Training - Train individual weights with gradient-based optimizers

Usage:
    python wann_pipeline.py
    python wann_pipeline.py --task mnist --optimizer adamw
"""

import argparse
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
    SupervisedProblem,
    # Export
    export_to_pytorch,
)


def run_xor_pipeline():
    """Full pipeline on XOR problem."""
    print("=" * 70)
    print("WANN Pipeline: XOR Problem")
    print("=" * 70)

    # Define problem
    class XORProblem(Problem):
        def __init__(self):
            super().__init__(input_dim=2, output_dim=1)
            self.x = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.float32)
            self.y = jnp.array([[0], [1], [1], [0]], dtype=jnp.float32)

        def evaluate(self, network, key):
            """For ES optimizer - returns Python float."""
            pred = jax.nn.sigmoid(network(self.x))
            mse = jnp.mean((pred - self.y) ** 2)
            return -float(mse)

        def loss(self, network, key):
            """For gradient-based optimizers - returns JAX array."""
            pred = jax.nn.sigmoid(network(self.x))
            return jnp.mean((pred - self.y) ** 2)

    problem = XORProblem()

    # ==========================================
    # Stage 1: Architecture Search
    # ==========================================
    print("\n--- Stage 1: Architecture Search ---")

    search_config = SearchConfig(
        pop_size=50,
        max_nodes=10,
        max_connections=30,
        activation_options=['tanh', 'relu', 'sigmoid', 'step'],
        weight_values=[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
        complexity_weight=0.01,
    )

    search = ArchitectureSearch(problem, search_config)
    genome = search.run(generations=50, log_interval=10)

    print(f"\nArchitecture found:")
    print(f"  Hidden nodes: {int(jnp.sum(genome.nodes[:, 1] == 1))}")
    print(f"  Connections: {int(jnp.sum(genome.connections[:, 2] == 1))}")

    # Test with shared weights
    print("\nShared weight evaluation:")
    for w in [-1.0, 0.5, 1.0]:
        net = search.genome_to_network(genome, w)
        pred = jax.nn.sigmoid(net(problem.x))
        acc = jnp.mean((pred > 0.5) == problem.y)
        print(f"  Weight {w:5.1f}: Accuracy = {float(acc) * 100:.0f}%")

    # ==========================================
    # Stage 2: Weight Training
    # ==========================================
    print("\n--- Stage 2: Weight Training ---")

    # Try different optimizers
    for optimizer in ['es', 'adam']:
        print(f"\nOptimizer: {optimizer.upper()}")

        trainer_config = WeightTrainerConfig(
            optimizer=optimizer,
            learning_rate=0.1 if optimizer == 'es' else 0.01,
            pop_size=32,
            noise_std=0.1,
        )

        trainer = WeightTrainer(
            genome=genome,
            problem=problem,
            config=trainer_config,
            activation_options=search_config.activation_options,
        )

        trainer.fit(epochs=50, log_interval=25)

        # Evaluate
        network = trainer.get_network()
        pred = jax.nn.sigmoid(network(problem.x))
        print(f"\nFinal predictions:")
        for i in range(4):
            print(f"  {int(problem.x[i, 0])} XOR {int(problem.x[i, 1])} = "
                  f"{pred[i, 0]:.3f} (expected: {int(problem.y[i, 0])})")

    return genome, trainer


def run_classification_pipeline():
    """Full pipeline on classification problem."""
    print("\n" + "=" * 70)
    print("WANN Pipeline: Classification Problem")
    print("=" * 70)

    # Generate synthetic data
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    n_samples = 200
    x1 = jax.random.normal(k1, (n_samples // 2, 4)) + jnp.array([2, 2, 0, 0])
    x2 = jax.random.normal(k2, (n_samples // 2, 4)) + jnp.array([-2, -2, 0, 0])
    x_train = jnp.concatenate([x1, x2], axis=0)
    y_train = jnp.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    problem = SupervisedProblem(
        x_train, y_train,
        loss_fn='cross_entropy',
        batch_size=64,
    )

    # ==========================================
    # Stage 1: Architecture Search
    # ==========================================
    print("\n--- Stage 1: Architecture Search ---")

    search_config = SearchConfig(
        pop_size=40,
        max_nodes=15,
        max_connections=50,
        activation_options=['tanh', 'relu', 'sigmoid'],
        weight_values=[-1.0, -0.5, 0.5, 1.0],
        complexity_weight=0.05,
    )

    search = ArchitectureSearch(problem, search_config)
    genome = search.run(generations=30, log_interval=10)

    # ==========================================
    # Stage 2: Weight Training with AdamW
    # ==========================================
    print("\n--- Stage 2: Weight Training (AdamW) ---")

    trainer_config = WeightTrainerConfig(
        optimizer='adamw',
        learning_rate=0.01,
        weight_decay=0.01,
    )

    trainer = WeightTrainer(
        genome=genome,
        problem=problem,
        config=trainer_config,
        activation_options=search_config.activation_options,
    )

    trainer.fit(epochs=50, log_interval=10)

    # Evaluate
    network = trainer.get_network()
    pred = network(x_train)
    pred_labels = jnp.argmax(pred, axis=-1)
    accuracy = jnp.mean(pred_labels == y_train)
    print(f"\nFinal accuracy: {float(accuracy) * 100:.1f}%")

    return genome, trainer


def run_mnist_pipeline(args):
    """Full pipeline on MNIST (if available)."""
    print("\n" + "=" * 70)
    print("WANN Pipeline: MNIST Classification")
    print("=" * 70)

    # Try to load MNIST
    try:
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 784)[:5000] / 255.0  # Use subset for speed
        y_train = y_train[:5000]
        x_test = x_test.reshape(-1, 784)[:1000] / 255.0
        y_test = y_test[:1000]
    except ImportError:
        print("TensorFlow not available. Using synthetic data.")
        key = jax.random.PRNGKey(42)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        x_train = jax.random.normal(k1, (1000, 784))
        y_train = jax.random.randint(k2, (1000,), 0, 10)
        x_test = jax.random.normal(k3, (200, 784))
        y_test = jax.random.randint(k4, (200,), 0, 10)

    x_train = jnp.array(x_train)
    y_train = jnp.array(y_train)
    x_test = jnp.array(x_test)
    y_test = jnp.array(y_test)

    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    problem = SupervisedProblem(
        x_train, y_train,
        x_val=x_test, y_val=y_test,
        loss_fn='cross_entropy',
        batch_size=128,
    )

    # ==========================================
    # Stage 1: Architecture Search
    # ==========================================
    print("\n--- Stage 1: Architecture Search ---")

    search_config = SearchConfig(
        pop_size=args.pop_size,
        max_nodes=args.max_nodes,
        max_connections=200,
        activation_options=['tanh', 'relu', 'sigmoid'],
        weight_values=[-1.0, 0.5, 1.0],
        complexity_weight=0.02,
    )

    search = ArchitectureSearch(problem, search_config)
    genome = search.run(generations=args.search_generations, log_interval=10)

    num_hidden = int(jnp.sum(genome.nodes[:, 1] == 1))
    num_conns = int(jnp.sum(genome.connections[:, 2] == 1))
    print(f"\nArchitecture: {num_hidden} hidden nodes, {num_conns} connections")

    # ==========================================
    # Stage 2: Weight Training
    # ==========================================
    print("\n--- Stage 2: Weight Training ---")

    trainer_config = WeightTrainerConfig(
        optimizer=args.optimizer,
        learning_rate=args.lr,
        weight_decay=0.01 if args.optimizer == 'adamw' else 0,
        pop_size=32,
    )

    trainer = WeightTrainer(
        genome=genome,
        problem=problem,
        config=trainer_config,
        activation_options=search_config.activation_options,
    )

    trainer.fit(epochs=args.train_epochs, log_interval=10)

    # Evaluate
    network = trainer.get_network()
    train_pred = jnp.argmax(network(x_train), axis=-1)
    test_pred = jnp.argmax(network(x_test), axis=-1)

    train_acc = jnp.mean(train_pred == y_train)
    test_acc = jnp.mean(test_pred == y_test)

    print(f"\nResults:")
    print(f"  Train accuracy: {float(train_acc) * 100:.1f}%")
    print(f"  Test accuracy:  {float(test_acc) * 100:.1f}%")

    # ==========================================
    # Export to PyTorch
    # ==========================================
    if args.export:
        print("\n--- Exporting to PyTorch ---")
        export_to_pytorch(
            genome=genome,
            weights=trainer.get_weights(),
            activation_options=search_config.activation_options,
            output_path=args.export,
        )

    return genome, trainer


def main():
    parser = argparse.ArgumentParser(description="WANN Two-Stage Pipeline")
    parser.add_argument("--task", choices=['xor', 'classification', 'mnist', 'all'],
                        default='all', help="Task to run")
    parser.add_argument("--optimizer", choices=['es', 'sgd', 'adam', 'adamw'],
                        default='adamw', help="Optimizer for Stage 2")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--search_generations", type=int, default=30, help="Architecture search generations")
    parser.add_argument("--train_epochs", type=int, default=50, help="Weight training epochs")
    parser.add_argument("--pop_size", type=int, default=30, help="Population size")
    parser.add_argument("--max_nodes", type=int, default=20, help="Max hidden nodes")
    parser.add_argument("--export", type=str, default=None, help="Export path for PyTorch model")
    args = parser.parse_args()

    if args.task == 'xor' or args.task == 'all':
        run_xor_pipeline()

    if args.task == 'classification' or args.task == 'all':
        run_classification_pipeline()

    if args.task == 'mnist':
        run_mnist_pipeline(args)

    print("\n" + "=" * 70)
    print("WANN Pipeline Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
