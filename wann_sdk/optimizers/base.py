"""
Base Optimizer Interface

Defines the abstract base class for all optimizers in WANN SDK.
Both gradient-based (JAXOpt) and evolutionary (Nevergrad) optimizers
implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp


@dataclass
class OptimizerState:
    """Container for optimizer state.

    Attributes:
        step: Current optimization step
        params: Current parameters
        internal: Optimizer-specific internal state
    """
    step: int = 0
    params: Optional[jnp.ndarray] = None
    internal: Any = None


class BaseOptimizer(ABC):
    """Abstract base class for all optimizers.

    Subclasses must implement:
        - init_state(): Initialize optimizer state
        - update(): Perform one optimization step

    Attributes:
        name: Optimizer name for registry lookup
        is_gradient_based: True if optimizer uses gradients
        supports_batched: True if optimizer can handle batched updates

    Example:
        >>> class MyOptimizer(BaseOptimizer):
        ...     name = "my-opt"
        ...     is_gradient_based = True
        ...
        ...     def init_state(self, params):
        ...         return OptimizerState(params=params)
        ...
        ...     def update(self, state, grads=None, fitness=None, **kwargs):
        ...         new_params = state.params - 0.01 * grads
        ...         return OptimizerState(step=state.step + 1, params=new_params)
    """

    # Class attributes (override in subclasses)
    name: str = "base"
    is_gradient_based: bool = True
    supports_batched: bool = False

    def __init__(self, **kwargs):
        """Initialize optimizer with hyperparameters.

        Args:
            **kwargs: Optimizer-specific hyperparameters
        """
        self.config = kwargs
        self._validate_config()

    def _validate_config(self):
        """Validate configuration. Override for custom validation."""
        pass

    @abstractmethod
    def init_state(self, params: jnp.ndarray) -> OptimizerState:
        """Initialize optimizer state.

        Args:
            params: Initial parameter values

        Returns:
            Initialized OptimizerState
        """
        pass

    @abstractmethod
    def update(
        self,
        state: OptimizerState,
        grads: Optional[jnp.ndarray] = None,
        fitness: Optional[float] = None,
        loss_fn: Optional[Callable] = None,
        key: Optional[jax.random.PRNGKey] = None,
        **kwargs,
    ) -> OptimizerState:
        """Perform one optimization step.

        For gradient-based optimizers, `grads` is used.
        For evolutionary optimizers, `fitness` or `loss_fn` is used.

        Args:
            state: Current optimizer state
            grads: Gradients (for gradient-based optimizers)
            fitness: Fitness value (for evolutionary optimizers)
            loss_fn: Loss function for evaluation (evolutionary)
            key: Random key (for stochastic optimizers)
            **kwargs: Additional optimizer-specific arguments

        Returns:
            Updated OptimizerState with new parameters
        """
        pass

    def get_params(self, state: OptimizerState) -> jnp.ndarray:
        """Get current parameters from state.

        Args:
            state: Optimizer state

        Returns:
            Current parameter values
        """
        return state.params

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get default hyperparameters.

        Override in subclasses to provide defaults.

        Returns:
            Dictionary of default hyperparameters
        """
        return {}

    @classmethod
    def get_config_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Get configuration schema with types and descriptions.

        Override in subclasses for full documentation.

        Returns:
            Schema dictionary: {param_name: {type, default, description}}
        """
        return {}

    def __repr__(self) -> str:
        config_str = ", ".join(f"{k}={v}" for k, v in self.config.items())
        return f"{self.__class__.__name__}({config_str})"


class GradientOptimizer(BaseOptimizer):
    """Base class for gradient-based optimizers.

    Provides common functionality for optimizers that use gradients.
    """

    is_gradient_based = True

    def __init__(
        self,
        learning_rate: float = 0.001,
        **kwargs,
    ):
        """Initialize gradient optimizer.

        Args:
            learning_rate: Step size for parameter updates
            **kwargs: Additional hyperparameters
        """
        super().__init__(learning_rate=learning_rate, **kwargs)
        self.learning_rate = learning_rate

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {"learning_rate": 0.001}


class EvolutionaryOptimizer(BaseOptimizer):
    """Base class for evolutionary/gradient-free optimizers.

    Provides common functionality for population-based optimizers.
    """

    is_gradient_based = False

    def __init__(
        self,
        population_size: int = 64,
        **kwargs,
    ):
        """Initialize evolutionary optimizer.

        Args:
            population_size: Number of candidates per generation
            **kwargs: Additional hyperparameters
        """
        super().__init__(population_size=population_size, **kwargs)
        self.population_size = population_size

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {"population_size": 64}

    def ask(self, state: OptimizerState, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, Any]:
        """Generate candidate solutions (for ask-tell interface).

        Args:
            state: Current optimizer state
            key: Random key

        Returns:
            Tuple of (candidates array, ask_state for tell())
        """
        raise NotImplementedError("Subclass must implement ask()")

    def tell(
        self,
        state: OptimizerState,
        ask_state: Any,
        fitnesses: jnp.ndarray,
    ) -> OptimizerState:
        """Update optimizer with fitness evaluations (for ask-tell interface).

        Args:
            state: Current optimizer state
            ask_state: State returned from ask()
            fitnesses: Fitness values for each candidate

        Returns:
            Updated optimizer state
        """
        raise NotImplementedError("Subclass must implement tell()")
