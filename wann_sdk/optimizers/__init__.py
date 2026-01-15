"""
WANN SDK Optimizers

Unified optimizer interface supporting both gradient-based (JAXOpt)
and evolutionary (Nevergrad) optimization algorithms.

Quick Start:
    >>> from wann_sdk.optimizers import Adam, CMA, ES
    >>>
    >>> # Gradient-based
    >>> opt = Adam(learning_rate=0.001)
    >>> state = opt.init_state(params)
    >>> state = opt.update(state, grads=gradients)
    >>>
    >>> # Evolutionary
    >>> opt = CMA(population_size=32)
    >>> state = opt.init_state(params)
    >>> state = opt.update(state, loss_fn=my_loss_fn, key=key)

Available Optimizers:
    Gradient-based (use with gradients):
        - Adam, AdamW: Adaptive moment estimation
        - SGD: Stochastic gradient descent with momentum
        - RMSProp: Root mean square propagation
        - AdaGrad: Adaptive gradient
        - LBFGS: Limited-memory BFGS (second-order)
        - GradientDescent: Vanilla gradient descent

    Evolutionary (use with fitness/loss function):
        - ES: Built-in Evolution Strategies (no dependencies)
        - CMA: Covariance Matrix Adaptation ES (requires nevergrad)
        - DE: Differential Evolution (requires nevergrad)
        - PSO: Particle Swarm Optimization (requires nevergrad)
        - NGOpt: Auto-selecting optimizer (requires nevergrad)

Registry Functions:
    >>> from wann_sdk.optimizers import list_optimizers, get_optimizer
    >>> print(list_optimizers())
    >>> OptClass = get_optimizer("adam")
"""

from typing import Any, Dict, List, Optional, Type, Union

# Base classes
from .base import (
    BaseOptimizer,
    GradientOptimizer,
    EvolutionaryOptimizer,
    OptimizerState,
)

# Gradient-based optimizers (JAXOpt-style)
from .jaxopt_optimizers import (
    Adam,
    AdamW,
    SGD,
    RMSProp,
    AdaGrad,
    LBFGS,
    GradientDescent,
    JAXOPT_OPTIMIZERS,
    list_gradient_optimizers,
)

# Built-in ES optimizers
from .es import (
    ES,
    OpenES,
    PEPG,
    ES_OPTIMIZERS,
)

# Nevergrad optimizers (lazy import to avoid dependency issues)
try:
    from .nevergrad_optimizers import (
        CMA,
        DE,
        TwoPointsDE,
        PSO,
        OnePlusOne,
        NGOpt,
        DiagonalCMA,
        TBPSA,
        NEVERGRAD_OPTIMIZERS,
        list_evolutionary_optimizers,
        auto_select_optimizer,
    )
    _NEVERGRAD_AVAILABLE = True
except ImportError:
    # Nevergrad not installed - create placeholder classes
    _NEVERGRAD_AVAILABLE = False
    NEVERGRAD_OPTIMIZERS = {}

    def list_evolutionary_optimizers():
        return {"es": "Built-in Evolution Strategies"}

    def auto_select_optimizer(*args, **kwargs):
        return ES

    # Placeholder classes that raise helpful errors
    class _NevergradPlaceholder(EvolutionaryOptimizer):
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"{self.__class__.__name__} requires nevergrad. "
                "Install with: pip install nevergrad"
            )

        def init_state(self, params):
            pass

        def update(self, state, **kwargs):
            pass

    class CMA(_NevergradPlaceholder):
        """CMA-ES (requires nevergrad)."""
        name = "cma"

    class DE(_NevergradPlaceholder):
        """Differential Evolution (requires nevergrad)."""
        name = "de"

    class TwoPointsDE(_NevergradPlaceholder):
        """Two-Points DE (requires nevergrad)."""
        name = "twopoints_de"

    class PSO(_NevergradPlaceholder):
        """Particle Swarm Optimization (requires nevergrad)."""
        name = "pso"

    class OnePlusOne(_NevergradPlaceholder):
        """(1+1) ES (requires nevergrad)."""
        name = "oneplusone"

    class NGOpt(_NevergradPlaceholder):
        """Nevergrad auto-optimizer (requires nevergrad)."""
        name = "ngopt"

    class DiagonalCMA(_NevergradPlaceholder):
        """Diagonal CMA-ES (requires nevergrad)."""
        name = "diagonal_cma"

    class TBPSA(_NevergradPlaceholder):
        """TBPSA (requires nevergrad)."""
        name = "tbpsa"


# ============================================================
# Optimizer Registry
# ============================================================

_REGISTRY: Dict[str, Type[BaseOptimizer]] = {}


def _build_registry():
    """Build optimizer registry from all sources."""
    global _REGISTRY
    _REGISTRY = {}

    # Add gradient optimizers
    for name, cls in JAXOPT_OPTIMIZERS.items():
        _REGISTRY[name.lower()] = cls

    # Add built-in ES
    for name, cls in ES_OPTIMIZERS.items():
        _REGISTRY[name.lower()] = cls

    # Add nevergrad optimizers if available
    if _NEVERGRAD_AVAILABLE:
        for name, cls in NEVERGRAD_OPTIMIZERS.items():
            _REGISTRY[name.lower()] = cls


# Build registry on import
_build_registry()


def register_optimizer(name: str, cls: Type[BaseOptimizer]):
    """Register a custom optimizer.

    Args:
        name: Name for registry lookup (case-insensitive)
        cls: Optimizer class (must inherit from BaseOptimizer)

    Example:
        >>> class MyOptimizer(BaseOptimizer):
        ...     name = "my-opt"
        ...     def init_state(self, params):
        ...         return OptimizerState(params=params)
        ...     def update(self, state, **kwargs):
        ...         return state
        >>>
        >>> register_optimizer("my-opt", MyOptimizer)
        >>> opt = get_optimizer("my-opt")
    """
    if not issubclass(cls, BaseOptimizer):
        raise TypeError(f"Optimizer must inherit from BaseOptimizer, got {type(cls)}")
    _REGISTRY[name.lower()] = cls


def unregister_optimizer(name: str):
    """Remove an optimizer from the registry.

    Args:
        name: Optimizer name to remove
    """
    _REGISTRY.pop(name.lower(), None)


def get_optimizer(name: str) -> Type[BaseOptimizer]:
    """Get optimizer class by name.

    Args:
        name: Optimizer name (case-insensitive)

    Returns:
        Optimizer class

    Raises:
        KeyError: If optimizer not found

    Example:
        >>> AdamCls = get_optimizer("adam")
        >>> opt = AdamCls(learning_rate=0.001)
    """
    name_lower = name.lower()
    if name_lower not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(
            f"Unknown optimizer: '{name}'. Available: {available}"
        )
    return _REGISTRY[name_lower]


def list_optimizers(category: Optional[str] = None) -> Dict[str, str]:
    """List all available optimizers.

    Args:
        category: Filter by category ('gradient', 'evolutionary', or None for all)

    Returns:
        Dictionary mapping optimizer names to descriptions

    Example:
        >>> all_opts = list_optimizers()
        >>> gradient_opts = list_optimizers('gradient')
        >>> evo_opts = list_optimizers('evolutionary')
    """
    result = {}

    for name, cls in _REGISTRY.items():
        # Filter by category
        if category == 'gradient' and not cls.is_gradient_based:
            continue
        if category == 'evolutionary' and cls.is_gradient_based:
            continue

        # Get description from docstring
        doc = cls.__doc__ or ""
        description = doc.split('\n')[0].strip()
        result[name] = description

    return dict(sorted(result.items()))


def create_optimizer(
    name_or_instance: Union[str, BaseOptimizer],
    **kwargs,
) -> BaseOptimizer:
    """Create optimizer from name or return existing instance.

    Convenience function that handles both string names and instances.

    Args:
        name_or_instance: Optimizer name (str) or existing instance
        **kwargs: Hyperparameters (only used if name is string)

    Returns:
        Optimizer instance

    Example:
        >>> opt = create_optimizer("adam", learning_rate=0.001)
        >>> # Or pass existing instance
        >>> existing = Adam(learning_rate=0.01)
        >>> opt = create_optimizer(existing)  # Returns same instance
    """
    if isinstance(name_or_instance, BaseOptimizer):
        return name_or_instance

    cls = get_optimizer(name_or_instance)
    return cls(**kwargs)


def is_gradient_based(name_or_instance: Union[str, BaseOptimizer]) -> bool:
    """Check if optimizer is gradient-based.

    Args:
        name_or_instance: Optimizer name or instance

    Returns:
        True if optimizer uses gradients, False if evolutionary
    """
    if isinstance(name_or_instance, BaseOptimizer):
        return name_or_instance.is_gradient_based

    cls = get_optimizer(name_or_instance)
    return cls.is_gradient_based


# ============================================================
# Exports
# ============================================================

__all__ = [
    # Base classes
    "BaseOptimizer",
    "GradientOptimizer",
    "EvolutionaryOptimizer",
    "OptimizerState",

    # Gradient optimizers
    "Adam",
    "AdamW",
    "SGD",
    "RMSProp",
    "AdaGrad",
    "LBFGS",
    "GradientDescent",

    # Built-in ES
    "ES",
    "OpenES",
    "PEPG",

    # Nevergrad optimizers
    "CMA",
    "DE",
    "TwoPointsDE",
    "PSO",
    "OnePlusOne",
    "NGOpt",
    "DiagonalCMA",
    "TBPSA",

    # Registry functions
    "register_optimizer",
    "unregister_optimizer",
    "get_optimizer",
    "list_optimizers",
    "create_optimizer",
    "is_gradient_based",

    # Utility functions
    "list_gradient_optimizers",
    "list_evolutionary_optimizers",
    "auto_select_optimizer",
]
