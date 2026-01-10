"""
WANN SDK - BipedalWalker Complete Training Example

Complete pipeline demonstrating:
1. Architecture search using WANN
2. Weight training
3. Evaluation and visualization

Supports two modes:
- V1: Ray/Gymnasium (compatible with BipedalWalker-v3)
- V2: Brax/Gymnax (uses humanoid, ~100x faster)

Usage:
    # V1 Mode (Gymnasium)
    python wann_bipedal.py --mode v1 --stage search
    python wann_bipedal.py --mode v1 --stage train
    python wann_bipedal.py --mode v1 --stage eval --render

    # V2 Mode (Brax - recommended)
    python wann_bipedal.py --mode v2 --stage search
    python wann_bipedal.py --mode v2 --stage train
    python wann_bipedal.py --mode v2 --stage eval

    # Full pipeline
    python wann_bipedal.py --mode v2 --stage full
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Optional
import argparse
import pickle
import time

# Core SDK imports
from core.wann_tensorneat import WANN, WANNGenome
from core.wann_sdk_core import (
    ArchitectureSpec,
    WANNArchitecture,
    TrainingConfig,
    WANNTrainer,
)

# TensorNEAT imports
from tensorneat.genome import BiasNode
from tensorneat.common import ACT, AGG


# ============================================================================
# V1 Mode: Gymnasium/Ray (Original)
# ============================================================================

def run_v1_search(args):
    """Architecture search using V1 (Gymnasium)."""
    from core.wann_sdk_ray_env import GymnasiumEnvWrapper, EnvFactory

    print("=" * 60)
    print("V1 Architecture Search: BipedalWalker-v3 (Gymnasium)")
    print("=" * 60)

    # Create environment
    env = GymnasiumEnvWrapper("BipedalWalker-v3")
    env_info = env.get_env_info()
    print(f"Environment: {env_info}")

    # Create WANN genome
    genome = WANNGenome(
        num_inputs=env_info['obs_dim'],
        num_outputs=env_info['action_dim'],
        max_nodes=300,
        max_conns=8000,
        init_hidden_layers=(),
        node_gene=BiasNode(
            activation_options=[
                ACT.tanh, ACT.relu, ACT.sigmoid,
                ACT.sin, ACT.gaussian,
            ],
            aggregation_options=AGG.sum,
        ),
        output_transform=ACT.tanh,
        weight_samples=jnp.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]),
    )

    # Create WANN algorithm
    algorithm = WANN(
        pop_size=args.pop_size,
        species_size=20,
        survival_threshold=0.1,
        genome=genome,
        complexity_weight=0.2,
    )

    # Gymnasium evaluation function
    def evaluate_population(alg_state, population):
        fitness_list = []
        weight_samples = genome.weight_samples
        num_episodes = 2
        max_steps = 300

        # Population is (pop_nodes, pop_conns) tuple
        pop_nodes, pop_conns = population
        pop_size = len(pop_nodes)

        for i in range(pop_size):
            nodes = pop_nodes[i]
            conns = pop_conns[i]
            # Transform network
            transformed = genome.transform(alg_state, nodes, conns)

            # Evaluate across weight samples
            rewards = []
            for weight in weight_samples:
                episode_rewards = []
                for ep in range(num_episodes):
                    obs, _ = env.reset(seed=ep)
                    total_reward = 0.0

                    for step in range(max_steps):
                        # Forward pass with shared weight
                        action = genome.forward_with_shared_weight(
                            alg_state, transformed, obs, float(weight)
                        )
                        action = np.array(action)

                        obs, reward, terminated, truncated, _ = env.step(action)
                        total_reward += reward

                        if terminated or truncated:
                            break

                    episode_rewards.append(total_reward)

                rewards.append(np.mean(episode_rewards))

            fitness_list.append(np.mean(rewards))

        return jnp.array(fitness_list)

    # Run search manually (bypass Pipeline jitable requirement)
    print(f"\nSearching ({args.pop_size} individuals, {args.generations} generations)...")
    start_time = time.time()

    # Initialize algorithm using State
    from tensorneat.common import State
    key = jax.random.PRNGKey(42)
    state = State(randkey=key)
    state = algorithm.setup(state)

    best_fitness_overall = -float('inf')
    best_individual = None

    for gen in range(args.generations):
        # Ask for population
        state, population = algorithm.ask(state)

        # Evaluate
        fitness = evaluate_population(state, population)

        # Population is (pop_nodes, pop_conns) tuple
        pop_nodes, pop_conns = population

        # Track best individual
        gen_best_idx = int(jnp.argmax(fitness))
        gen_best_fitness = float(fitness[gen_best_idx])
        if gen_best_fitness > best_fitness_overall:
            best_fitness_overall = gen_best_fitness
            best_individual = (pop_nodes[gen_best_idx].copy(), pop_conns[gen_best_idx].copy())

        # Tell fitness
        state = algorithm.tell(state, fitness)

        if gen % 5 == 0 or gen == args.generations - 1:
            mean_fitness = float(jnp.mean(fitness))
            elapsed = time.time() - start_time
            print(f"Gen {gen:3d} [{elapsed:.1f}s]: Best={best_fitness_overall:.2f}, Mean={mean_fitness:.2f}")

    # Save best architecture
    best_nodes, best_conns = best_individual

    # Calculate number of active connections
    num_conns = int(jnp.sum(best_conns[:, 3] != 0)) if best_conns.shape[1] > 3 else len(best_conns)

    spec = ArchitectureSpec(
        nodes=best_nodes,
        connections=best_conns,
        num_inputs=env_info['obs_dim'],
        num_outputs=env_info['action_dim'],
        num_hidden=max(0, int(jnp.sum(best_nodes[:, 0] >= 0)) - env_info['obs_dim'] - env_info['action_dim']),
        num_params=num_conns,
        search_fitness=best_fitness_overall,
        search_complexity=num_conns,
        metadata={'mode': 'v1', 'env': 'BipedalWalker-v3'}
    )

    save_path = Path(args.save_dir) / "bipedal_v1_arch.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    spec.save(str(save_path))

    print(f"\nArchitecture saved to {save_path}")
    print(f"Best fitness: {spec.search_fitness:.2f}")

    env.close()
    return spec


def run_v1_train(args):
    """Weight training using V1."""
    from core.wann_sdk_ray_env import GymnasiumEnvWrapper, EnvFactory
    from core.wann_sdk_rl_methods import create_policy_for_environment

    print("=" * 60)
    print("V1 Weight Training: BipedalWalker-v3")
    print("=" * 60)

    # Load architecture
    arch_path = Path(args.save_dir) / "bipedal_v1_arch.pkl"
    if not arch_path.exists():
        print(f"Architecture not found at {arch_path}")
        print("Run --stage search first")
        return None

    spec = ArchitectureSpec.load(str(arch_path))
    architecture = WANNArchitecture(spec)
    print(f"Loaded architecture with {spec.num_params} parameters")

    # Create environment
    env_factory = EnvFactory(env_name="BipedalWalker-v3", mode="local")
    env = env_factory.create()
    env_info = env.get_env_info()

    # Create policy
    config = TrainingConfig(
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=50000,
    )

    policy = create_policy_for_environment(
        architecture=architecture,
        env_info=env_info,
        method='ppo',
        config=config
    )

    # Training loop
    print(f"\nTraining for {args.train_steps} steps...")
    start_time = time.time()

    obs, _ = env.reset()
    episode_rewards = []
    current_episode_reward = 0

    for step in range(args.train_steps):
        action = policy.select_action(obs)
        if isinstance(action, jnp.ndarray):
            action = np.array(action)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        policy.store_transition(obs, action, reward, next_obs, done)
        current_episode_reward += reward

        if policy.ready_to_update():
            policy.update_step()

        if done:
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0
            obs, _ = env.reset()
        else:
            obs = next_obs

        if step % 10000 == 0 and len(episode_rewards) > 0:
            mean_reward = np.mean(episode_rewards[-100:])
            elapsed = time.time() - start_time
            print(f"Step {step} [{elapsed:.1f}s]: Mean reward = {mean_reward:.2f}")

    # Save trained weights
    save_path = Path(args.save_dir) / "bipedal_v1_trained.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump({
            'architecture_spec': spec,
            'params': policy.get_params(),
            'episode_rewards': episode_rewards,
        }, f)

    print(f"\nTrained model saved to {save_path}")
    env.close()

    return policy


def run_v1_eval(args):
    """Evaluation using V1."""
    from core.wann_sdk_ray_env import GymnasiumEnvWrapper
    from core.wann_sdk_rl_methods import create_policy_for_environment

    print("=" * 60)
    print("V1 Evaluation: BipedalWalker-v3")
    print("=" * 60)

    # Load model
    model_path = Path(args.save_dir) / "bipedal_v1_trained.pkl"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Run --stage train first")
        return

    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    spec = data['architecture_spec']
    params = data['params']

    architecture = WANNArchitecture(spec)

    # Create environment
    render_mode = "human" if args.render else None
    env = GymnasiumEnvWrapper("BipedalWalker-v3", render_mode=render_mode)
    env_info = env.get_env_info()

    # Create policy
    config = TrainingConfig()
    policy = create_policy_for_environment(
        architecture=architecture,
        env_info=env_info,
        method='ppo',
        config=config
    )
    policy.set_params(params)

    # Evaluate
    print(f"\nEvaluating for {args.eval_episodes} episodes...")

    rewards = []
    for ep in range(args.eval_episodes):
        obs, _ = env.reset(seed=ep)
        episode_reward = 0

        while True:
            action = policy.select_action(obs, deterministic=True)
            if isinstance(action, jnp.ndarray):
                action = np.array(action)

            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

        rewards.append(episode_reward)
        print(f"Episode {ep+1}: {episode_reward:.2f}")

    print(f"\nMean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    env.close()


# ============================================================================
# V2 Mode: Brax/Gymnax (High Performance)
# ============================================================================

def run_v2_search(args):
    """Architecture search using V2 (Brax) with ES."""
    try:
        from core.wann_sdk_brax_env import UnifiedEnv
    except ImportError as e:
        print(f"Brax not available: {e}")
        print("Install with: pip install brax")
        return None

    print("=" * 60)
    print("V2 Training: Humanoid (Brax + ES)")
    print("=" * 60)

    # Use humanoid as Brax equivalent
    env = UnifiedEnv("humanoid", backend="brax", batch_size=args.batch_size)
    env_info = env.get_env_info()
    print(f"Environment: {env_info}")

    # For V2/Brax, we use a simple ES approach
    # Just evolve connection weights directly using a simple ES approach
    print(f"\nSearching ({args.pop_size} individuals, {args.generations} generations)...")
    print(f"Batch size: {args.batch_size}")
    print("Note: V2 uses simplified ES-based search")
    start_time = time.time()

    # Simple ES-based architecture search
    key = jax.random.PRNGKey(42)

    # Initialize simple network parameters
    # For humanoid: 244 inputs, 17 outputs
    num_hidden = 32
    layer_sizes = [env_info['obs_dim'], num_hidden, env_info['action_dim']]

    # Initialize weights
    def init_params(key):
        params = []
        for i in range(len(layer_sizes) - 1):
            key, subkey = jax.random.split(key)
            w = jax.random.normal(subkey, (layer_sizes[i], layer_sizes[i+1])) * 0.1
            params.append(w)
        return params

    def forward(params, obs):
        x = obs
        for i, w in enumerate(params):
            x = x @ w
            if i < len(params) - 1:
                x = jnp.tanh(x)
        return jnp.tanh(x)  # Output in [-1, 1]

    def evaluate_params(params, eval_key):
        obs, env_state = env.reset(eval_key)
        total_reward = 0.0
        for step in range(200):
            eval_key, step_key = jax.random.split(eval_key)
            action = forward(params, obs)
            obs, env_state, reward, done, _ = env.step(env_state, action, step_key)
            total_reward += float(jnp.mean(reward))
        return total_reward

    # ES parameters
    sigma = 0.1
    learning_rate = 0.01

    best_params = init_params(key)
    best_fitness_overall = evaluate_params(best_params, key)

    for gen in range(args.generations):
        key, noise_key = jax.random.split(key)

        # Generate population with noise
        fitness_list = []
        noise_list = []

        for i in range(args.pop_size):
            key, eval_key, noise_key = jax.random.split(key, 3)

            # Add noise to params
            noise = [jax.random.normal(noise_key, w.shape) for w in best_params]
            noisy_params = [w + sigma * n for w, n in zip(best_params, noise)]

            # Evaluate
            fitness = evaluate_params(noisy_params, eval_key)
            fitness_list.append(fitness)
            noise_list.append(noise)

        fitness_array = jnp.array(fitness_list)

        # Update best
        gen_best_idx = int(jnp.argmax(fitness_array))
        gen_best_fitness = float(fitness_array[gen_best_idx])
        if gen_best_fitness > best_fitness_overall:
            best_fitness_overall = gen_best_fitness
            best_noise = noise_list[gen_best_idx]
            best_params = [w + sigma * n for w, n in zip(best_params, best_noise)]

        # ES update (simplified)
        fitness_normalized = (fitness_array - jnp.mean(fitness_array)) / (jnp.std(fitness_array) + 1e-8)
        for j, w in enumerate(best_params):
            grad = jnp.zeros_like(w)
            for i in range(args.pop_size):
                grad += fitness_normalized[i] * noise_list[i][j]
            grad /= (args.pop_size * sigma)
            best_params[j] = w + learning_rate * grad

        if gen % 5 == 0 or gen == args.generations - 1:
            mean_fitness = float(jnp.mean(fitness_array))
            elapsed = time.time() - start_time
            print(f"Gen {gen:3d} [{elapsed:.1f}s]: Best={best_fitness_overall:.2f}, Mean={mean_fitness:.2f}")

    # Save ES-trained model (different format from WANN)
    total_params = sum(w.size for w in best_params)

    save_path = Path(args.save_dir) / "bipedal_v2_trained.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'wb') as f:
        pickle.dump({
            'params': best_params,
            'layer_sizes': layer_sizes,
            'fitness': best_fitness_overall,
            'mode': 'v2_es',
            'env': 'humanoid',
            'num_params': total_params,
        }, f)

    total_time = time.time() - start_time
    print(f"\nModel saved to {save_path}")
    print(f"Best fitness: {best_fitness_overall:.2f}")
    print(f"Total params: {total_params}")
    print(f"Total time: {total_time:.1f}s")

    return best_params


def run_v2_train(args):
    """Weight training using V2 - V2 search already includes training."""
    print("=" * 60)
    print("V2 Weight Training: Humanoid (Brax + ES)")
    print("=" * 60)
    print("\nNote: V2 search already includes ES-based training.")
    print("The model from search stage is already trained.")
    print("Run --stage search to train, or --stage eval to evaluate.")
    return None


def run_v2_eval(args):
    """Evaluation using V2."""
    try:
        from core.wann_sdk_brax_env import UnifiedEnv
    except ImportError as e:
        print(f"Brax not available: {e}")
        return

    print("=" * 60)
    print("V2 Evaluation: Humanoid (Brax)")
    print("=" * 60)

    # Load model
    model_path = Path(args.save_dir) / "bipedal_v2_trained.pkl"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Run --stage search first")
        return

    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    params = data['params']
    layer_sizes = data['layer_sizes']
    print(f"Loaded model with {data['num_params']} parameters")
    print(f"Training fitness: {data['fitness']:.2f}")

    # Create environment
    env = UnifiedEnv("humanoid", backend="brax", batch_size=1)

    # Forward function
    def forward(params, obs):
        x = obs
        for i, w in enumerate(params):
            x = x @ w
            if i < len(params) - 1:
                x = jnp.tanh(x)
        return jnp.tanh(x)

    # Evaluate
    print(f"\nEvaluating for {args.eval_episodes} episodes...")

    rewards = []
    key = jax.random.PRNGKey(0)

    for ep in range(args.eval_episodes):
        key, eval_key = jax.random.split(key)
        obs, env_state = env.reset(eval_key)
        total_reward = 0.0

        for step in range(1000):
            key, step_key = jax.random.split(key)
            action = forward(params, obs)
            obs, env_state, reward, done, _ = env.step(env_state, action, step_key)
            total_reward += float(jnp.mean(reward))

        rewards.append(total_reward)
        print(f"Episode {ep+1}: {total_reward:.2f}")

    print(f"\nMean reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="WANN SDK - BipedalWalker Example")

    parser.add_argument('--mode', choices=['v1', 'v2'], default='v2',
                        help='V1=Gymnasium, V2=Brax (recommended)')
    parser.add_argument('--stage', choices=['search', 'train', 'eval', 'full'],
                        default='full', help='Pipeline stage')

    # Search parameters
    parser.add_argument('--pop_size', type=int, default=100,
                        help='Population size')
    parser.add_argument('--generations', type=int, default=20,
                        help='Number of generations')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for V2 mode')

    # Training parameters
    parser.add_argument('--train_steps', type=int, default=100000,
                        help='Training steps')

    # Evaluation parameters
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='Evaluation episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render during evaluation (V1 only)')

    # Paths
    parser.add_argument('--save_dir', default='./models',
                        help='Directory to save models')

    args = parser.parse_args()

    print(f"\nWANN SDK - BipedalWalker Example")
    print(f"Mode: {'Gymnasium (V1)' if args.mode == 'v1' else 'Brax (V2)'}")
    print(f"Stage: {args.stage}\n")

    if args.mode == 'v1':
        if args.stage in ['search', 'full']:
            run_v1_search(args)
        if args.stage in ['train', 'full']:
            run_v1_train(args)
        if args.stage in ['eval', 'full']:
            run_v1_eval(args)
    else:  # v2
        if args.stage in ['search', 'full']:
            run_v2_search(args)
        if args.stage in ['train', 'full']:
            run_v2_train(args)
        if args.stage in ['eval', 'full']:
            run_v2_eval(args)

    print("\nPipeline completed!")


if __name__ == "__main__":
    main()
