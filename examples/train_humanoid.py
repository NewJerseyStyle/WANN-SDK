#!/usr/bin/env python3
"""
WANN SDK Example: Training a Humanoid Robot

This example demonstrates how to use the WANN SDK to train a neural network
policy for humanoid locomotion using Evolution Strategies (ES).

Usage:
    python train_humanoid.py --generations 100 --pop_size 256

Requirements:
    pip install wann-sdk[brax]
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Train a humanoid robot using Evolution Strategies"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="humanoid",
        help="Environment name (default: humanoid)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=100,
        help="Number of generations (default: 100)",
    )
    parser.add_argument(
        "--pop_size",
        type=int,
        default=256,
        help="Population size (default: 256)",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        default=[64, 64],
        help="Hidden layer sizes (default: 64 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.1,
        help="Noise standard deviation (default: 0.1)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./models/humanoid_es.pkl",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only evaluate a saved model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # Import WANN SDK
    try:
        from wann_sdk import BraxEnv, ESTrainer, TrainingConfig, list_environments
    except ImportError:
        print("Error: wann-sdk not installed.")
        print("Install with: pip install wann-sdk[brax]")
        sys.exit(1)

    # List available environments
    print("Available environments:")
    for name, info in list_environments().items():
        print(f"  - {name}: {info['description']}")
    print()

    # Create environment
    print(f"Creating environment: {args.env}")
    try:
        env = BraxEnv(args.env, batch_size=1)
    except ImportError:
        print("Error: Brax not installed.")
        print("Install with: pip install brax")
        sys.exit(1)

    print(f"  Observation dim: {env.obs_dim}")
    print(f"  Action dim: {env.action_dim}")
    print()

    # Create training config
    config = TrainingConfig(
        pop_size=args.pop_size,
        learning_rate=args.lr,
        noise_std=args.noise_std,
        hidden_sizes=args.hidden,
    )

    if args.eval_only:
        # Load and evaluate
        print(f"Loading model from {args.save_path}")
        trainer = ESTrainer.load(args.save_path, env)
        print(f"Best training fitness: {trainer.best_fitness:.2f}")
        print()
        print("Evaluating...")
        trainer.evaluate(num_episodes=10)
    else:
        # Train
        trainer = ESTrainer(env, config, seed=args.seed)

        print("Starting training...")
        print(f"  Generations: {args.generations}")
        print(f"  Population size: {args.pop_size}")
        print(f"  Network: {trainer.layer_sizes}")
        print()

        results = trainer.train(
            generations=args.generations,
            log_interval=10,
            verbose=True,
        )

        # Save model
        trainer.save(args.save_path)

        # Final evaluation
        print("\nFinal evaluation:")
        trainer.evaluate(num_episodes=10)


if __name__ == "__main__":
    main()
