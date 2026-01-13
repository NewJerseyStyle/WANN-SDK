#!/usr/bin/env python3
"""
Distributed Architecture Search Example

Demonstrates multi-node parallel evolution using Ray.

Usage:
    # Local parallel (uses all CPU cores)
    python distributed_search.py

    # With specific number of workers
    python distributed_search.py --workers 8

    # Connect to Ray cluster
    python distributed_search.py --cluster

Cluster Setup:
    # On head node
    ray start --head --port=6379

    # On worker nodes
    ray start --address='<head-node-ip>:6379'

    # Then run this script with --cluster flag
    python distributed_search.py --cluster

See Ray docs: https://docs.ray.io/en/latest/cluster/getting-started.html
"""

import argparse
import jax.numpy as jnp

from wann_sdk import (
    # Distributed
    DistributedSearch,
    init_ray,
    shutdown_ray,
    get_cluster_info,
    # Config
    SearchConfig,
    # Stage 2
    WeightTrainer,
    WeightTrainerConfig,
    # Problem
    SupervisedProblem,
)


def create_synthetic_data(n_samples: int = 1000, n_features: int = 10):
    """Create synthetic classification data."""
    import jax
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    # Two-class classification
    x1 = jax.random.normal(k1, (n_samples // 2, n_features)) + 1.0
    x2 = jax.random.normal(k2, (n_samples // 2, n_features)) - 1.0
    x = jnp.concatenate([x1, x2], axis=0)
    y = jnp.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    # Shuffle
    perm = jax.random.permutation(k3, n_samples)
    return x[perm], y[perm]


def run_distributed_search(args):
    """Run distributed architecture search."""
    print("=" * 60)
    print("Distributed WANN Architecture Search")
    print("=" * 60)

    # Initialize Ray
    if args.cluster:
        print("\nConnecting to Ray cluster...")
        resources = init_ray(address='auto')
    else:
        print(f"\nStarting local Ray with {args.workers} workers...")
        resources = init_ray(num_cpus=args.workers)

    print(f"Cluster resources: {resources}")

    # Get cluster info
    info = get_cluster_info()
    print(f"Available CPUs: {info['available_resources'].get('CPU', 0)}")
    print(f"Available GPUs: {info['available_resources'].get('GPU', 0)}")
    print(f"Nodes: {len(info['nodes'])}")

    # Create data
    print("\nCreating synthetic dataset...")
    x_train, y_train = create_synthetic_data(
        n_samples=args.samples,
        n_features=args.features,
    )
    print(f"Data shape: {x_train.shape}")

    # ==========================================
    # Stage 1: Distributed Architecture Search
    # ==========================================
    print("\n--- Stage 1: Distributed Architecture Search ---")

    search_config = SearchConfig(
        pop_size=args.pop_size,
        max_nodes=args.max_nodes,
        max_connections=100,
        activation_options=['tanh', 'relu', 'sigmoid', 'sin'],
        weight_values=[-1.0, 0.5, 1.0],
        complexity_weight=0.02,
    )

    # Use DistributedSearch instead of ArchitectureSearch
    search = DistributedSearch(
        problem_class=SupervisedProblem,
        problem_kwargs={
            'x_train': x_train,
            'y_train': y_train,
            'loss_fn': 'cross_entropy',
            'batch_size': 128,
        },
        config=search_config,
        num_workers=args.workers if not args.cluster else None,
    )

    genome = search.run(generations=args.generations, log_interval=5)

    num_hidden = int(jnp.sum(genome.nodes[:, 1] == 1))
    num_conns = int(jnp.sum(genome.connections[:, 2] == 1))
    print(f"\nArchitecture found: {num_hidden} hidden nodes, {num_conns} connections")

    # ==========================================
    # Stage 2: Weight Training (local)
    # ==========================================
    print("\n--- Stage 2: Weight Training ---")

    # Create local problem for weight training
    problem = SupervisedProblem(
        x_train, y_train,
        loss_fn='cross_entropy',
        batch_size=128,
    )

    trainer = WeightTrainer(
        genome=genome,
        problem=problem,
        config=WeightTrainerConfig(
            optimizer='adam',
            learning_rate=0.01,
        ),
        activation_options=search_config.activation_options,
    )

    trainer.fit(epochs=args.train_epochs, log_interval=10)

    # Evaluate
    network = trainer.get_network()
    pred = jnp.argmax(network(x_train), axis=-1)
    accuracy = jnp.mean(pred == y_train)
    print(f"\nFinal accuracy: {float(accuracy) * 100:.1f}%")

    # Cleanup
    shutdown_ray()

    return genome, trainer


def main():
    parser = argparse.ArgumentParser(description="Distributed WANN Search")
    parser.add_argument("--cluster", action="store_true",
                        help="Connect to existing Ray cluster")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of workers (local mode)")
    parser.add_argument("--pop_size", type=int, default=100,
                        help="Population size")
    parser.add_argument("--max_nodes", type=int, default=20,
                        help="Max hidden nodes")
    parser.add_argument("--generations", type=int, default=30,
                        help="Number of generations")
    parser.add_argument("--train_epochs", type=int, default=50,
                        help="Weight training epochs")
    parser.add_argument("--samples", type=int, default=1000,
                        help="Number of training samples")
    parser.add_argument("--features", type=int, default=10,
                        help="Number of input features")
    args = parser.parse_args()

    run_distributed_search(args)

    print("\n" + "=" * 60)
    print("Distributed Search Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
