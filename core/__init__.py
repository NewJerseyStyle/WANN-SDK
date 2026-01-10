"""
WANN SDK Core Package

Provides Weight Agnostic Neural Networks implementation with:
- Architecture search using TensorNEAT
- Multiple RL training methods (DQN, PPO, etc.)
- Environment wrappers (Gymnasium, Brax)
"""

from .wann_tensorneat import WANNGenome, WANN
from .wann_sdk_core import (
    NetworkArchitecture,
    PolicyInterface,
    ArchitectureSpec,
    WANNArchitecture,
    TrainingConfig,
    ReplayBuffer,
    TrainingMethodRegistry,
    WANNTrainer,
    create_trainer_from_checkpoint,
    list_available_methods,
)
from .wann_sdk_ray_env import (
    GymnasiumEnvWrapper,
    DistributedEnvPool,
    EnvFactory,
    test_environment,
)
from .wann_sdk_rl_methods import (
    DQNPolicy,
    PPOPolicy,
    create_policy_for_environment,
)

# Brax/Gymnax environments (V2 - high performance)
try:
    from .wann_sdk_brax_env import (
        UnifiedEnv,
        ESPolicy,
        search_architecture,
        list_available_envs,
        test_unified_env,
    )
    BRAX_ENV_AVAILABLE = True
except ImportError:
    BRAX_ENV_AVAILABLE = False

__all__ = [
    # WANN core
    "WANNGenome",
    "WANN",
    # Architecture
    "NetworkArchitecture",
    "PolicyInterface",
    "ArchitectureSpec",
    "WANNArchitecture",
    "TrainingConfig",
    "ReplayBuffer",
    "TrainingMethodRegistry",
    "WANNTrainer",
    # Environments
    "GymnasiumEnvWrapper",
    "DistributedEnvPool",
    "EnvFactory",
    "test_environment",
    # Training methods
    "DQNPolicy",
    "PPOPolicy",
    "create_policy_for_environment",
    # Utilities
    "create_trainer_from_checkpoint",
    "list_available_methods",
    # Brax/Gymnax (V2)
    "UnifiedEnv",
    "ESPolicy",
    "search_architecture",
    "list_available_envs",
    "test_unified_env",
    "BRAX_ENV_AVAILABLE",
]
