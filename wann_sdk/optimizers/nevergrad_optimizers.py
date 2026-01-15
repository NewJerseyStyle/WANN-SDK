"""
Nevergrad Optimizer Wrappers

Provides evolutionary/gradient-free optimizers from Nevergrad library.
All optimizers follow the BaseOptimizer interface with ask-tell pattern.

Available optimizers:
- CMA: Covariance Matrix Adaptation Evolution Strategy
- DE: Differential Evolution
- PSO: Particle Swarm Optimization
- OnePlusOne: Simple (1+1) Evolution Strategy
- NGOpt: Nevergrad's auto-selecting optimizer
- TwoPointsDE: Two-points Differential Evolution

Usage:
    >>> from wann_sdk.optimizers import CMA, PSO
    >>> opt = CMA(population_size=32)
    >>> state = opt.init_state(params)
    >>> # Ask-tell pattern
    >>> candidates, ask_state = opt.ask(state, key)
    >>> fitnesses = evaluate_all(candidates)
    >>> state = opt.tell(state, ask_state, fitnesses)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import jax
import jax.numpy as jnp
import numpy as np

from .base import EvolutionaryOptimizer, OptimizerState

# Try to import nevergrad
try:
    import nevergrad as ng
    NEVERGRAD_AVAILABLE = True
except ImportError:
    NEVERGRAD_AVAILABLE = False
    ng = None


def _check_nevergrad():
    """Check if Nevergrad is available."""
    if not NEVERGRAD_AVAILABLE:
        raise ImportError(
            "Nevergrad is not installed. Install with: pip install nevergrad"
        )


class NevergradWrapper(EvolutionaryOptimizer):
    """Base wrapper for Nevergrad optimizers.

    Handles conversion between JAX arrays and Nevergrad's parametrization.
    """

    _ng_optimizer_class: str = None  # Override in subclasses

    def __init__(
        self,
        population_size: int = 64,
        **kwargs,
    ):
        super().__init__(population_size=population_size, **kwargs)
        self._ng_optimizer = None
        self._param_shape = None

    def _create_ng_optimizer(self, num_params: int):
        """Create Nevergrad optimizer instance."""
        _check_nevergrad()

        # Create parameter space
        param = ng.p.Array(shape=(num_params,))

        # Get optimizer class
        opt_class = getattr(ng.optimizers, self._ng_optimizer_class)
        return opt_class(parametrization=param, budget=10000, num_workers=1)

    def init_state(self, params: jnp.ndarray) -> OptimizerState:
        """Initialize optimizer with parameter shape."""
        self._param_shape = params.shape
        num_params = int(np.prod(params.shape))

        # Create Nevergrad optimizer
        self._ng_optimizer = self._create_ng_optimizer(num_params)

        # Set initial point
        self._ng_optimizer.parametrization.value = np.array(params.flatten())

        return OptimizerState(
            step=0,
            params=params,
            internal={'ng_optimizer': self._ng_optimizer},
        )

    def ask(
        self,
        state: OptimizerState,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, Any]:
        """Generate candidate solutions.

        Args:
            state: Current optimizer state
            key: Random key (unused, Nevergrad handles randomness)

        Returns:
            Tuple of (candidates array [pop_size, num_params], ask_state)
        """
        ng_opt = state.internal['ng_optimizer']

        # Ask for candidates
        candidates_ng = [ng_opt.ask() for _ in range(self.population_size)]

        # Convert to JAX array
        candidates = jnp.array([c.value for c in candidates_ng])

        # Reshape if needed
        if self._param_shape is not None and len(self._param_shape) > 1:
            candidates = candidates.reshape(
                (self.population_size,) + self._param_shape
            )

        return candidates, candidates_ng

    def tell(
        self,
        state: OptimizerState,
        ask_state: Any,
        fitnesses: jnp.ndarray,
    ) -> OptimizerState:
        """Update optimizer with fitness evaluations.

        Args:
            state: Current optimizer state
            ask_state: Candidates from ask()
            fitnesses: Fitness values (higher is better)

        Returns:
            Updated optimizer state
        """
        ng_opt = state.internal['ng_optimizer']
        candidates_ng = ask_state

        # Nevergrad minimizes, so negate fitnesses
        losses = -np.array(fitnesses)

        # Tell results
        for candidate, loss in zip(candidates_ng, losses):
            ng_opt.tell(candidate, float(loss))

        # Get current best
        recommendation = ng_opt.recommend()
        best_params = jnp.array(recommendation.value)

        if self._param_shape is not None:
            best_params = best_params.reshape(self._param_shape)

        return OptimizerState(
            step=state.step + 1,
            params=best_params,
            internal={'ng_optimizer': ng_opt},
        )

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

        For evolutionary optimizers, this does ask-evaluate-tell internally.
        """
        if key is None:
            key = jax.random.PRNGKey(state.step)

        if loss_fn is None:
            raise ValueError("Evolutionary optimizers require loss_fn")

        # Ask for candidates
        candidates, ask_state = self.ask(state, key)

        # Evaluate all candidates
        fitnesses = []
        for i in range(self.population_size):
            candidate = candidates[i]
            key, eval_key = jax.random.split(key)
            # loss_fn should return scalar loss (we negate for fitness)
            loss = loss_fn(candidate)
            fitnesses.append(-float(loss))  # Negate: lower loss = higher fitness

        fitnesses = jnp.array(fitnesses)

        # Tell results
        return self.tell(state, ask_state, fitnesses)


class CMA(NevergradWrapper):
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

    State-of-the-art evolutionary optimizer for continuous optimization.
    Adapts the search distribution based on successful mutations.

    Best for:
    - Continuous optimization with 10-1000 parameters
    - Non-convex, multi-modal landscapes
    - When gradients are unavailable or unreliable

    Args:
        population_size: Number of candidates per generation (default: 64)
        sigma: Initial step size / standard deviation (default: 0.5)

    Example:
        >>> opt = CMA(population_size=32)
        >>> state = opt.init_state(params)
        >>> state = opt.update(state, loss_fn=my_loss_fn, key=key)

    Reference:
        Hansen & Ostermeier, "Completely Derandomized Self-Adaptation" (2001)
    """

    name = "cma"
    _ng_optimizer_class = "CMA"

    def __init__(
        self,
        population_size: int = 64,
        sigma: float = 0.5,
        **kwargs,
    ):
        super().__init__(population_size=population_size, **kwargs)
        self.sigma = sigma

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {"population_size": 64, "sigma": 0.5}


class DE(NevergradWrapper):
    """Differential Evolution.

    Population-based optimizer using difference vectors for mutation.
    Robust and simple, good for many optimization problems.

    Best for:
    - Global optimization
    - Problems with many local minima
    - Moderate dimensionality (10-100 parameters)

    Args:
        population_size: Number of candidates (default: 64)
        cr: Crossover probability (default: 0.5)
        f: Differential weight (default: 0.8)

    Reference:
        Storn & Price, "Differential Evolution" (1997)
    """

    name = "de"
    _ng_optimizer_class = "DE"

    def __init__(
        self,
        population_size: int = 64,
        cr: float = 0.5,
        f: float = 0.8,
        **kwargs,
    ):
        super().__init__(population_size=population_size, **kwargs)
        self.cr = cr
        self.f = f

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {"population_size": 64, "cr": 0.5, "f": 0.8}


class TwoPointsDE(NevergradWrapper):
    """Two-Points Differential Evolution.

    Variant of DE that uses two-point crossover.
    Often more efficient than standard DE.

    Args:
        population_size: Number of candidates (default: 64)
    """

    name = "twopoints_de"
    _ng_optimizer_class = "TwoPointsDE"


class PSO(NevergradWrapper):
    """Particle Swarm Optimization.

    Population of particles that explore the search space,
    attracted to their personal best and global best positions.

    Best for:
    - Continuous optimization
    - Problems where good solutions cluster together
    - When you want fast initial progress

    Args:
        population_size: Number of particles (default: 64)
        omega: Inertia weight (default: 0.7)
        phip: Personal best attraction (default: 1.5)
        phig: Global best attraction (default: 1.5)

    Reference:
        Kennedy & Eberhart, "Particle Swarm Optimization" (1995)
    """

    name = "pso"
    _ng_optimizer_class = "PSO"

    def __init__(
        self,
        population_size: int = 64,
        omega: float = 0.7,
        phip: float = 1.5,
        phig: float = 1.5,
        **kwargs,
    ):
        super().__init__(population_size=population_size, **kwargs)
        self.omega = omega
        self.phip = phip
        self.phig = phig

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "population_size": 64,
            "omega": 0.7,
            "phip": 1.5,
            "phig": 1.5,
        }


class OnePlusOne(NevergradWrapper):
    """(1+1) Evolution Strategy.

    Simple ES that maintains single solution and mutates it.
    Uses 1/5 success rule to adapt step size.

    Best for:
    - Simple problems
    - Quick prototyping
    - When population overhead is too expensive

    Args:
        population_size: Not used (always 1), kept for API consistency
    """

    name = "oneplusone"
    _ng_optimizer_class = "OnePlusOne"

    def __init__(self, **kwargs):
        # OnePlusOne doesn't use population
        super().__init__(population_size=1, **kwargs)


class NGOpt(NevergradWrapper):
    """Nevergrad's Auto-selecting Optimizer.

    Automatically selects the best optimizer based on the problem
    characteristics (dimensionality, budget, etc.).

    Best for:
    - When you don't know which optimizer to use
    - General-purpose optimization
    - Benchmarking

    Args:
        population_size: Number of candidates (default: 64)
    """

    name = "ngopt"
    _ng_optimizer_class = "NGOpt"


class DiagonalCMA(NevergradWrapper):
    """Diagonal CMA-ES.

    Simplified CMA-ES that only adapts diagonal covariance.
    Faster than full CMA for high-dimensional problems.

    Best for:
    - High-dimensional problems (100+ parameters)
    - When full CMA is too slow
    - Separable or nearly-separable problems

    Args:
        population_size: Number of candidates (default: 64)
    """

    name = "diagonal_cma"
    _ng_optimizer_class = "DiagonalCMA"


class TBPSA(NevergradWrapper):
    """Test-Based Population Size Adaptation.

    ES variant that adapts population size during optimization.

    Args:
        population_size: Initial population size (default: 64)
    """

    name = "tbpsa"
    _ng_optimizer_class = "TBPSA"


# Dictionary of all Nevergrad optimizers
NEVERGRAD_OPTIMIZERS: Dict[str, Type[EvolutionaryOptimizer]] = {
    "cma": CMA,
    "de": DE,
    "twopoints_de": TwoPointsDE,
    "pso": PSO,
    "oneplusone": OnePlusOne,
    "ngopt": NGOpt,
    "diagonal_cma": DiagonalCMA,
    "tbpsa": TBPSA,
}


def list_evolutionary_optimizers() -> Dict[str, str]:
    """List all available evolutionary optimizers.

    Returns:
        Dictionary mapping optimizer names to descriptions
    """
    return {
        name: cls.__doc__.split('\n')[0] if cls.__doc__ else ""
        for name, cls in NEVERGRAD_OPTIMIZERS.items()
    }


def auto_select_optimizer(
    num_params: int,
    budget: int = 1000,
    has_gradients: bool = False,
) -> Type[EvolutionaryOptimizer]:
    """Auto-select best evolutionary optimizer for problem.

    Args:
        num_params: Number of parameters
        budget: Evaluation budget
        has_gradients: Whether gradients are available

    Returns:
        Recommended optimizer class
    """
    if has_gradients:
        # If gradients available, use gradient-based instead
        from .jaxopt_optimizers import Adam
        return Adam

    if num_params < 10:
        return OnePlusOne
    elif num_params < 100:
        return CMA
    elif num_params < 1000:
        return DiagonalCMA
    else:
        return NGOpt
