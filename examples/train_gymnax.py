#!/usr/bin/env python3
"""
Gymnax RL Environment Example

Demonstrates WANN SDK with Gymnax classic control environments.
Gymnax provides JAX-native implementations of Gym environments.

Environments:
- CartPole-v1: Balance a pole on a cart (discrete actions)
- Pendulum-v1: Swing up and balance pendulum (continuous actions)
- MountainCar-v0: Drive car up a mountain (discrete actions)
- Acrobot-v1: Swing up double pendulum (discrete actions)

Installation:
    pip install gymnax

Usage:
    python train_gymnax.py
    python train_gymnax.py --env CartPole-v1 --generations 50
    python train_gymnax.py --env Pendulum-v1 --optimizer es
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
    # Environment
    GymnaxEnv,
    GymnaxProblem,
    list_gymnax_environments,
)


def train_cartpole(args):
    """Train on CartPole-v1 (discrete actions)."""
    print("=" * 60)
    print("WANN Pipeline: CartPole-v1")
    print("=" * 60)

    # Create environment
    env = GymnaxEnv("CartPole-v1")
    print(f"Environment: {env}")
    print(f"  Observation dim: {env.obs_dim}")
    print(f"  Action dim: {env.action_dim} (discrete)")

    # Create problem
    problem = GymnaxProblem(env, max_steps=500, num_rollouts=3)

    # ==========================================
    # Stage 1: Architecture Search
    # ==========================================
    print("\n--- Stage 1: Architecture Search ---")

    search_config = SearchConfig(
        pop_size=args.pop_size,
        max_nodes=args.max_nodes,
        max_connections=50,
        activation_options=['tanh', 'relu', 'sigmoid'],
        weight_values=[-1.0, 0.5, 1.0],
        complexity_weight=0.01,
    )

    search = ArchitectureSearch(problem, search_config)
    genome = search.run(generations=args.generations, log_interval=10)

    num_hidden = int(jnp.sum(genome.nodes[:, 1] == 1))
    num_conns = int(jnp.sum(genome.connections[:, 2] == 1))
    print(f"\nArchitecture: {num_hidden} hidden nodes, {num_conns} connections")

    # Test with shared weights
    print("\nShared weight evaluation (3 rollouts each):")
    for w in [-1.0, 0.5, 1.0]:
        network = search.genome_to_network(genome, w)
        key = jax.random.PRNGKey(42)
        rewards = []
        for _ in range(3):
            key, eval_key = jax.random.split(key)
            reward = problem.evaluate(network, eval_key)
            rewards.append(reward)
        mean_reward = jnp.mean(jnp.array(rewards))
        print(f"  Weight {w:5.1f}: Mean reward = {float(mean_reward):.1f}")

    # ==========================================
    # Stage 2: Weight Training
    # ==========================================
    print("\n--- Stage 2: Weight Training (ES) ---")

    trainer_config = WeightTrainerConfig(
        optimizer='es',  # ES works well for RL
        learning_rate=args.lr,
        pop_size=32,
        noise_std=0.1,
    )

    trainer = WeightTrainer(
        genome=genome,
        problem=problem,
        config=trainer_config,
        activation_options=search_config.activation_options,
    )

    trainer.fit(epochs=args.epochs, log_interval=10)

    # Final evaluation
    network = trainer.get_network()
    key = jax.random.PRNGKey(0)
    rewards = []
    for _ in range(10):
        key, eval_key = jax.random.split(key)
        reward = problem.evaluate(network, eval_key)
        rewards.append(reward)

    print(f"\nFinal performance (10 rollouts):")
    print(f"  Mean reward: {float(jnp.mean(jnp.array(rewards))):.1f}")
    print(f"  Max reward:  {float(jnp.max(jnp.array(rewards))):.1f}")

    return genome, trainer


def train_pendulum(args):
    """Train on Pendulum-v1 (continuous actions)."""
    print("\n" + "=" * 60)
    print("WANN Pipeline: Pendulum-v1")
    print("=" * 60)

    # Create environment
    env = GymnaxEnv("Pendulum-v1")
    print(f"Environment: {env}")
    print(f"  Observation dim: {env.obs_dim}")
    print(f"  Action dim: {env.action_dim} (continuous)")

    # Create problem
    problem = GymnaxProblem(env, max_steps=200, num_rollouts=3)

    # ==========================================
    # Stage 1: Architecture Search
    # ==========================================
    print("\n--- Stage 1: Architecture Search ---")

    search_config = SearchConfig(
        pop_size=args.pop_size,
        max_nodes=args.max_nodes,
        activation_options=['tanh', 'sin', 'identity'],
        weight_values=[-1.0, 1.0],
        complexity_weight=0.02,
    )

    search = ArchitectureSearch(problem, search_config)
    genome = search.run(generations=args.generations, log_interval=10)

    # ==========================================
    # Stage 2: Weight Training
    # ==========================================
    print("\n--- Stage 2: Weight Training (ES) ---")

    trainer_config = WeightTrainerConfig(
        optimizer='es',
        learning_rate=args.lr,
        pop_size=32,
    )

    trainer = WeightTrainer(
        genome=genome,
        problem=problem,
        config=trainer_config,
        activation_options=search_config.activation_options,
    )

    trainer.fit(epochs=args.epochs, log_interval=10)

    # Final evaluation
    network = trainer.get_network()
    key = jax.random.PRNGKey(0)
    rewards = []
    for _ in range(10):
        key, eval_key = jax.random.split(key)
        reward = problem.evaluate(network, eval_key)
        rewards.append(reward)

    print(f"\nFinal performance (10 rollouts):")
    print(f"  Mean reward: {float(jnp.mean(jnp.array(rewards))):.1f}")
    print(f"  Max reward:  {float(jnp.max(jnp.array(rewards))):.1f}")

    return genome, trainer


def train_acrobot(args):
    """Train on Acrobot-v1 (discrete actions)."""
    print("\n" + "=" * 60)
    print("WANN Pipeline: Acrobot-v1")
    print("=" * 60)

    # Create environment
    env = GymnaxEnv("Acrobot-v1")
    print(f"Environment: {env}")

    # Create problem
    problem = GymnaxProblem(env, max_steps=500, num_rollouts=3)

    # Stage 1
    print("\n--- Stage 1: Architecture Search ---")
    search_config = SearchConfig(
        pop_size=args.pop_size,
        max_nodes=args.max_nodes,
        activation_options=['tanh', 'relu', 'sin'],
        weight_values=[-1.0, 0.5, 1.0],
    )

    search = ArchitectureSearch(problem, search_config)
    genome = search.run(generations=args.generations, log_interval=10)

    # Stage 2
    print("\n--- Stage 2: Weight Training ---")
    trainer = WeightTrainer(
        genome=genome,
        problem=problem,
        config=WeightTrainerConfig(optimizer='es', learning_rate=args.lr),
        activation_options=search_config.activation_options,
    )
    trainer.fit(epochs=args.epochs, log_interval=10)

    return genome, trainer


def list_envs():
    """List available Gymnax environments."""
    print("Available Gymnax Environments:")
    print("-" * 50)
    for name, info in list_gymnax_environments().items():
        action_type = "continuous" if info.get('continuous', False) else "discrete"
        print(f"  {name:30s} obs={info['obs_dim']:3d}  act={info['action_dim']:2d} ({action_type})")
        print(f"    {info['description']}")


def main():
    parser = argparse.ArgumentParser(description="WANN + Gymnax Training")
    parser.add_argument("--env", type=str, default="cartpole",
                        choices=['cartpole', 'pendulum', 'acrobot', 'list'],
                        help="Environment to train on")
    parser.add_argument("--generations", type=int, default=30,
                        help="Architecture search generations")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Weight training epochs")
    parser.add_argument("--pop_size", type=int, default=30,
                        help="Population size for search")
    parser.add_argument("--max_nodes", type=int, default=15,
                        help="Maximum hidden nodes")
    parser.add_argument("--lr", type=float, default=0.05,
                        help="Learning rate for weight training")
    args = parser.parse_args()

    if args.env == 'list':
        list_envs()
        return

    if args.env == 'cartpole':
        train_cartpole(args)
    elif args.env == 'pendulum':
        train_pendulum(args)
    elif args.env == 'acrobot':
        train_acrobot(args)

    print("\n" + "=" * 60)
    print("Gymnax Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
