"""
WANN SDK - Weight Agnostic Neural Networks

A JAX-based SDK for neuroevolution and neural network optimization.
Implements the full WANN two-stage pipeline.

=== Standard Two-Stage Pipeline ===

Stage 1: Architecture Search (WANN)
    - Evolve network topology (nodes, connections, activations)
    - Evaluate with shared weights across all connections
    - Find architectures that work regardless of weight value

Stage 2: Weight Training
    - Train individual weights on found architecture
    - Supports ES, SGD, Adam, AdamW optimizers
    - Export to PyTorch for downstream fine-tuning

=== Quick Start ===

    >>> from wann_sdk import (
    ...     ArchitectureSearch, SearchConfig,
    ...     WeightTrainer, WeightTrainerConfig,
    ...     SupervisedProblem, export_to_pytorch,
    ... )
    >>>
    >>> # Define problem
    >>> problem = SupervisedProblem(x_train, y_train, loss_fn='cross_entropy')
    >>>
    >>> # Stage 1: Architecture Search
    >>> search = ArchitectureSearch(problem, SearchConfig(max_nodes=30))
    >>> genome = search.run(generations=100)
    >>>
    >>> # Stage 2: Weight Training
    >>> trainer = WeightTrainer(
    ...     genome, problem,
    ...     WeightTrainerConfig(optimizer='adamw', learning_rate=0.001)
    ... )
    >>> trainer.fit(epochs=100)
    >>>
    >>> # Export to PyTorch
    >>> export_to_pytorch(genome, trainer.get_weights(), 'model.py')
"""

__version__ = "0.6.0"
__author__ = "WANN SDK Contributors"

# === Stage 1: Architecture Search ===
from .search import (
    ArchitectureSearch,
    SearchConfig,
    NetworkGenome,
)

# === Stage 2: Weight Training ===
from .weight_trainer import (
    WeightTrainer,
    WeightTrainerConfig,
    TrainableNetwork,
)

# === Problem Definition ===
from .problem import (
    Problem,
    SupervisedProblem,
    RLProblem,
    GymnaxProblem,
)

# === Export ===
from .export import (
    export_to_pytorch,
    export_to_onnx,
)

# === Environments ===
from .environments import (
    BraxEnv,
    GymnaxEnv,
    list_environments,
    list_gymnax_environments,
)

# === Architecture (Advanced) ===
from .architecture import (
    NetworkArchitecture,
    ArchitectureSpec,
    WANNArchitecture,
)

# === TensorNEAT Integration (Advanced) ===
from .algorithm import (
    WANNGenome,
    WANN,
)

# === Distributed (Ray) ===
from .distributed import (
    DistributedSearch,
    init_ray,
    shutdown_ray,
    get_cluster_info,
    wait_for_workers,
)

# === Activation Approximation ===
from .activation_approx import (
    ApproximatorConfig,
    set_cache_dir as set_approx_cache_dir,
    get_differentiable_activation,
    is_non_differentiable,
)

# === Zero-Cost Proxies & Trainability ===
from .zero_cost_proxies import (
    ZCPEvaluator,
    ZCPConfig,
    compute_synflow,
    compute_naswot,
    compute_snip,
    compute_trainability,
)

from .trainability_search import (
    TrainabilityAwareSearch,
    TrainabilitySearchConfig,
)

# === Optimizers ===
from .optimizers import (
    # Base classes
    BaseOptimizer,
    OptimizerState,
    # Gradient optimizers (Optax)
    Adam,
    AdamW,
    SGD,
    RMSProp,
    AdaGrad,
    LBFGS,
    Lion,
    Lamb,
    # Evolutionary optimizers
    ES,
    CMA,
    DE,
    PSO,
    NGOpt,
    # Registry functions
    list_optimizers,
    get_optimizer,
    register_optimizer,
    create_optimizer,
)

# === Deprecated: Old MLP-based API ===
# Use ArchitectureSearch + WeightTrainer instead
from .trainer import Trainer, TrainerConfig
from .training import ESTrainer, TrainingConfig

import warnings as _warnings

def _deprecation_warning():
    _warnings.warn(
        "Trainer and TrainerConfig are deprecated. "
        "Use the two-stage pipeline: ArchitectureSearch + WeightTrainer",
        DeprecationWarning,
        stacklevel=3,
    )


__all__ = [
    # Version
    "__version__",

    # Stage 1: Architecture Search
    "ArchitectureSearch",
    "SearchConfig",
    "NetworkGenome",

    # Stage 2: Weight Training
    "WeightTrainer",
    "WeightTrainerConfig",
    "TrainableNetwork",

    # Problem Definition
    "Problem",
    "SupervisedProblem",
    "RLProblem",
    "GymnaxProblem",

    # Export
    "export_to_pytorch",
    "export_to_onnx",

    # Environments
    "BraxEnv",
    "GymnaxEnv",
    "list_environments",
    "list_gymnax_environments",

    # Architecture (Advanced)
    "NetworkArchitecture",
    "ArchitectureSpec",
    "WANNArchitecture",

    # TensorNEAT (Advanced)
    "WANNGenome",
    "WANN",

    # Distributed (Ray)
    "DistributedSearch",
    "init_ray",
    "shutdown_ray",
    "get_cluster_info",
    "wait_for_workers",

    # Activation Approximation
    "ApproximatorConfig",
    "set_approx_cache_dir",
    "get_differentiable_activation",
    "is_non_differentiable",

    # Zero-Cost Proxies & Trainability
    "ZCPEvaluator",
    "ZCPConfig",
    "compute_synflow",
    "compute_naswot",
    "compute_snip",
    "compute_trainability",
    "TrainabilityAwareSearch",
    "TrainabilitySearchConfig",

    # Optimizers
    "BaseOptimizer",
    "OptimizerState",
    "Adam",
    "AdamW",
    "SGD",
    "RMSProp",
    "AdaGrad",
    "LBFGS",
    "ES",
    "CMA",
    "DE",
    "PSO",
    "NGOpt",
    "list_optimizers",
    "get_optimizer",
    "register_optimizer",
    "create_optimizer",

    # Deprecated
    "Trainer",
    "TrainerConfig",
    "ESTrainer",
    "TrainingConfig",
]
