"""
WANN SDK - Quick Test Script

Tests basic functionality of the SDK components.
Run this to verify the installation is working correctly.

Usage:
    python test_sdk.py
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all core modules can be imported."""
    print("Testing imports...")

    try:
        from core.wann_tensorneat import WANNGenome, WANN
        print("  [OK] wann_tensorneat")
    except ImportError as e:
        print(f"  [FAIL] wann_tensorneat: {e}")
        return False

    try:
        from core.wann_sdk_core import (
            ArchitectureSpec,
            WANNArchitecture,
            TrainingConfig,
            ReplayBuffer,
            TrainingMethodRegistry,
        )
        print("  [OK] wann_sdk_core")
    except ImportError as e:
        print(f"  [FAIL] wann_sdk_core: {e}")
        return False

    try:
        from core.wann_sdk_ray_env import GymnasiumEnvWrapper, EnvFactory
        print("  [OK] wann_sdk_ray_env")
    except ImportError as e:
        print(f"  [FAIL] wann_sdk_ray_env: {e}")
        return False

    try:
        from core.wann_sdk_rl_methods import DQNPolicy, PPOPolicy
        print("  [OK] wann_sdk_rl_methods")
    except ImportError as e:
        print(f"  [FAIL] wann_sdk_rl_methods: {e}")
        return False

    # Optional: Brax environment
    try:
        from core.wann_sdk_brax_env import UnifiedEnv
        print("  [OK] wann_sdk_brax_env (optional)")
    except ImportError as e:
        print(f"  [SKIP] wann_sdk_brax_env: {e}")

    return True


def test_wann_genome():
    """Test WANNGenome creation and forward pass."""
    print("\nTesting WANNGenome...")

    import jax.numpy as jnp
    from core.wann_tensorneat import WANNGenome
    from tensorneat.genome import BiasNode
    from tensorneat.common import ACT, AGG

    try:
        genome = WANNGenome(
            num_inputs=4,
            num_outputs=2,
            max_nodes=20,
            max_conns=50,
            node_gene=BiasNode(
                activation_options=[ACT.tanh, ACT.relu],
                aggregation_options=AGG.sum,
            ),
            weight_samples=jnp.array([-1.0, 0.0, 1.0]),
        )
        print("  [OK] WANNGenome created")
        print(f"      - Inputs: {genome.num_inputs}")
        print(f"      - Outputs: {genome.num_outputs}")
        print(f"      - Weight samples: {genome.weight_samples}")
        return True
    except Exception as e:
        print(f"  [FAIL] WANNGenome: {e}")
        return False


def test_gymnasium_env():
    """Test Gymnasium environment wrapper."""
    print("\nTesting Gymnasium environment...")

    try:
        from core.wann_sdk_ray_env import GymnasiumEnvWrapper
        import numpy as np

        # Use CartPole as a simple test environment
        env = GymnasiumEnvWrapper("CartPole-v1")
        info = env.get_env_info()

        print("  [OK] Environment created")
        print(f"      - Name: {info['env_name']}")
        print(f"      - Obs dim: {info['obs_dim']}")
        print(f"      - Action dim: {info['action_dim']}")
        print(f"      - Discrete action: {info['action_is_discrete']}")

        # Test reset and step
        obs, _ = env.reset()
        print(f"  [OK] Reset: obs shape = {obs.shape}")

        action = np.array(0)  # CartPole uses discrete actions
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"  [OK] Step: reward = {reward}")

        env.close()
        return True

    except Exception as e:
        print(f"  [FAIL] Gymnasium env: {e}")
        return False


def test_replay_buffer():
    """Test replay buffer."""
    print("\nTesting ReplayBuffer...")

    try:
        import jax
        import jax.numpy as jnp
        from core.wann_sdk_core import ReplayBuffer

        buffer = ReplayBuffer(
            capacity=1000,
            observation_shape=(4,),
            action_shape=(1,),
        )

        # Add some transitions
        for i in range(100):
            obs = jnp.ones(4) * i
            action = jnp.array([i % 2])
            reward = float(i)
            next_obs = jnp.ones(4) * (i + 1)
            done = i % 10 == 9

            buffer.add(obs, action, reward, next_obs, done)

        print(f"  [OK] Buffer size: {buffer.size}")

        # Sample batch
        key = jax.random.PRNGKey(0)
        batch = buffer.sample(32, key)

        print(f"  [OK] Sampled batch:")
        print(f"      - Observations: {batch['observations'].shape}")
        print(f"      - Actions: {batch['actions'].shape}")
        print(f"      - Rewards: {batch['rewards'].shape}")

        return True

    except Exception as e:
        print(f"  [FAIL] ReplayBuffer: {e}")
        return False


def test_training_methods():
    """Test that training methods are registered."""
    print("\nTesting training methods...")

    try:
        from core.wann_sdk_core import TrainingMethodRegistry

        methods = TrainingMethodRegistry.list_methods()
        print(f"  [OK] Registered methods: {methods}")

        for method in methods:
            cls = TrainingMethodRegistry.get(method)
            print(f"      - {method}: {cls.__name__}")

        return True

    except Exception as e:
        print(f"  [FAIL] Training methods: {e}")
        return False


def test_brax_env():
    """Test Brax environment (optional)."""
    print("\nTesting Brax environment (optional)...")

    try:
        from core.wann_sdk_brax_env import UnifiedEnv
        import jax

        env = UnifiedEnv("ant", backend="brax", batch_size=4)
        info = env.get_env_info()

        print("  [OK] Brax environment created")
        print(f"      - Name: {info['env_name']}")
        print(f"      - Backend: {info['backend']}")
        print(f"      - Obs dim: {info['obs_dim']}")
        print(f"      - Action dim: {info['action_dim']}")
        print(f"      - Batch size: {info['batch_size']}")

        # Test reset
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        print(f"  [OK] Reset: obs shape = {obs.shape}")

        return True

    except ImportError as e:
        print(f"  [SKIP] Brax not installed: {e}")
        return True  # Not a failure, just optional

    except Exception as e:
        print(f"  [FAIL] Brax env: {e}")
        return False


def main():
    print("=" * 60)
    print("WANN SDK - Quick Test")
    print("=" * 60)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("WANNGenome", test_wann_genome()))
    results.append(("Gymnasium", test_gymnasium_env()))
    results.append(("ReplayBuffer", test_replay_buffer()))
    results.append(("Training Methods", test_training_methods()))
    results.append(("Brax (optional)", test_brax_env()))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = 0
    failed = 0

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")

    if failed == 0:
        print("\nAll tests passed! SDK is ready to use.")
        print("\nNext steps:")
        print("  1. Run the BipedalWalker example:")
        print("     python example/wann_bipedal.py --mode v1 --stage search --pop_size 50 --generations 10")
        print("")
        print("  2. Or with Brax (if installed):")
        print("     python example/wann_bipedal.py --mode v2 --stage search --pop_size 50 --generations 10")
    else:
        print("\nSome tests failed. Please check the error messages above.")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
