"""
Built-in Evolution Strategies Optimizer

Provides a pure-JAX implementation of Evolution Strategies (ES)
without external dependencies. Uses antithetic sampling for variance reduction.

This is the default evolutionary optimizer in WANN SDK.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import jax
import jax.numpy as jnp

from .base import EvolutionaryOptimizer, OptimizerState


class ES(EvolutionaryOptimizer):
    """Evolution Strategies with antithetic sampling.

    Pure JAX implementation of ES using parameter perturbations
    and fitness-weighted updates. No external dependencies required.

    Uses antithetic sampling (mirrored perturbations) to reduce
    variance in gradient estimates.

    Best for:
    - When you want no external dependencies
    - Small to medium parameter counts
    - RL-style fitness optimization

    Args:
        population_size: Number of perturbation pairs (default: 64)
        learning_rate: Step size for parameter updates (default: 0.01)
        noise_std: Standard deviation of perturbation noise (default: 0.1)
        weight_decay: L2 regularization coefficient (default: 0.0)

    Example:
        >>> opt = ES(population_size=32, learning_rate=0.01, noise_std=0.1)
        >>> state = opt.init_state(params)
        >>> # Using update() with loss_fn
        >>> state = opt.update(state, loss_fn=my_loss_fn, key=key)
        >>> # Or using ask-tell pattern
        >>> candidates, ask_state = opt.ask(state, key)
        >>> fitnesses = jnp.array([evaluate(c) for c in candidates])
        >>> state = opt.tell(state, ask_state, fitnesses)

    Reference:
        Salimans et al., "Evolution Strategies as a Scalable Alternative
        to Reinforcement Learning" (2017)
    """

    name = "es"
    is_gradient_based = False

    def __init__(
        self,
        population_size: int = 64,
        learning_rate: float = 0.01,
        noise_std: float = 0.1,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        super().__init__(population_size=population_size, **kwargs)
        self.learning_rate = learning_rate
        self.noise_std = noise_std
        self.weight_decay = weight_decay

    def init_state(self, params: jnp.ndarray) -> OptimizerState:
        """Initialize ES state."""
        return OptimizerState(
            step=0,
            params=params,
            internal=None,
        )

    def ask(
        self,
        state: OptimizerState,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Generate candidate solutions using antithetic sampling.

        Generates pairs of mirrored perturbations for variance reduction.

        Args:
            state: Current optimizer state
            key: Random key for noise generation

        Returns:
            Tuple of (candidates [2*pop_size, param_dim], ask_state)
            ask_state contains noise vectors for tell()
        """
        params = state.params

        # Generate noise vectors
        noise_vectors = []
        candidates = []

        for i in range(self.population_size):
            key, noise_key = jax.random.split(key)
            noise = jax.random.normal(noise_key, params.shape)
            noise_vectors.append(noise)

            # Positive perturbation
            pos_candidate = params + self.noise_std * noise
            candidates.append(pos_candidate)

            # Negative perturbation (antithetic)
            neg_candidate = params - self.noise_std * noise
            candidates.append(neg_candidate)

        candidates = jnp.stack(candidates)
        noise_vectors = jnp.stack(noise_vectors)

        ask_state = {
            'noise_vectors': noise_vectors,
            'base_params': params,
        }

        return candidates, ask_state

    def tell(
        self,
        state: OptimizerState,
        ask_state: Dict[str, Any],
        fitnesses: jnp.ndarray,
    ) -> OptimizerState:
        """Update parameters based on fitness evaluations.

        Uses fitness-weighted combination of noise vectors
        to estimate gradient direction.

        Args:
            state: Current optimizer state
            ask_state: State from ask() containing noise vectors
            fitnesses: Fitness values [2*pop_size] (positive/negative pairs)

        Returns:
            Updated optimizer state
        """
        noise_vectors = ask_state['noise_vectors']
        base_params = ask_state['base_params']

        # Separate positive and negative fitnesses
        pos_fitnesses = fitnesses[0::2]  # Even indices
        neg_fitnesses = fitnesses[1::2]  # Odd indices

        # Compute fitness differences
        fitness_diff = pos_fitnesses - neg_fitnesses

        # Normalize
        std = jnp.std(fitness_diff)
        fitness_normalized = (fitness_diff - jnp.mean(fitness_diff)) / (std + 1e-8)

        # Compute gradient estimate
        grad = jnp.zeros_like(base_params)
        for i in range(self.population_size):
            grad = grad + fitness_normalized[i] * noise_vectors[i]
        grad = grad / (self.population_size * self.noise_std)

        # Update parameters (gradient ascent for fitness maximization)
        new_params = base_params + self.learning_rate * grad

        # Apply weight decay
        if self.weight_decay > 0:
            new_params = new_params - self.learning_rate * self.weight_decay * base_params

        return OptimizerState(
            step=state.step + 1,
            params=new_params,
            internal=None,
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
        """Perform one ES optimization step.

        Internally performs ask-evaluate-tell cycle.

        Args:
            state: Current optimizer state
            grads: Ignored (ES doesn't use gradients)
            fitness: Ignored (uses loss_fn for evaluation)
            loss_fn: Function to evaluate candidates: params -> scalar loss
            key: Random key for perturbations

        Returns:
            Updated optimizer state with new parameters
        """
        if key is None:
            key = jax.random.PRNGKey(state.step)

        if loss_fn is None:
            raise ValueError("ES requires loss_fn for evaluation")

        # Ask for candidates
        candidates, ask_state = self.ask(state, key)

        # Evaluate all candidates
        fitnesses = []
        for candidate in candidates:
            loss = loss_fn(candidate)
            # Negate loss to get fitness (minimize loss = maximize fitness)
            fitnesses.append(-float(loss))

        fitnesses = jnp.array(fitnesses)

        # Tell results
        return self.tell(state, ask_state, fitnesses)

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "population_size": 64,
            "learning_rate": 0.01,
            "noise_std": 0.1,
            "weight_decay": 0.0,
        }

    @classmethod
    def get_config_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "population_size": {
                "type": int,
                "default": 64,
                "description": "Number of perturbation pairs per generation",
            },
            "learning_rate": {
                "type": float,
                "default": 0.01,
                "description": "Step size for parameter updates",
            },
            "noise_std": {
                "type": float,
                "default": 0.1,
                "description": "Standard deviation of perturbation noise",
            },
            "weight_decay": {
                "type": float,
                "default": 0.0,
                "description": "L2 regularization coefficient",
            },
        }


class OpenES(ES):
    """OpenAI-style Evolution Strategies.

    Variant of ES with additional features like fitness shaping
    and adaptive noise.

    Args:
        population_size: Number of perturbation pairs (default: 64)
        learning_rate: Step size (default: 0.01)
        noise_std: Perturbation noise (default: 0.1)
        fitness_shaping: Use rank-based fitness shaping (default: True)

    Reference:
        Salimans et al., "Evolution Strategies as a Scalable Alternative
        to Reinforcement Learning" (2017)
    """

    name = "openes"

    def __init__(
        self,
        population_size: int = 64,
        learning_rate: float = 0.01,
        noise_std: float = 0.1,
        fitness_shaping: bool = True,
        **kwargs,
    ):
        super().__init__(
            population_size=population_size,
            learning_rate=learning_rate,
            noise_std=noise_std,
            **kwargs,
        )
        self.fitness_shaping = fitness_shaping

    def tell(
        self,
        state: OptimizerState,
        ask_state: Dict[str, Any],
        fitnesses: jnp.ndarray,
    ) -> OptimizerState:
        """Update with optional fitness shaping."""
        if self.fitness_shaping:
            # Rank-based fitness shaping
            ranks = jnp.argsort(jnp.argsort(-fitnesses))  # Higher fitness = lower rank
            n = len(fitnesses)
            # Centered ranks in [-0.5, 0.5]
            shaped_fitnesses = (ranks / (n - 1)) - 0.5
            fitnesses = shaped_fitnesses

        return super().tell(state, ask_state, fitnesses)


class PEPG(ES):
    """Parameter-Exploring Policy Gradients.

    ES variant that also adapts the noise standard deviation.

    Args:
        population_size: Number of perturbation pairs (default: 64)
        learning_rate: Step size for means (default: 0.01)
        noise_std: Initial perturbation noise (default: 0.1)
        std_learning_rate: Learning rate for std adaptation (default: 0.001)

    Reference:
        Sehnke et al., "Parameter-exploring Policy Gradients" (2010)
    """

    name = "pepg"

    def __init__(
        self,
        population_size: int = 64,
        learning_rate: float = 0.01,
        noise_std: float = 0.1,
        std_learning_rate: float = 0.001,
        **kwargs,
    ):
        super().__init__(
            population_size=population_size,
            learning_rate=learning_rate,
            noise_std=noise_std,
            **kwargs,
        )
        self.std_learning_rate = std_learning_rate

    def init_state(self, params: jnp.ndarray) -> OptimizerState:
        """Initialize with adaptive noise std."""
        return OptimizerState(
            step=0,
            params=params,
            internal={'noise_std': self.noise_std},
        )

    def tell(
        self,
        state: OptimizerState,
        ask_state: Dict[str, Any],
        fitnesses: jnp.ndarray,
    ) -> OptimizerState:
        """Update parameters and noise std."""
        # First do normal ES update
        new_state = super().tell(state, ask_state, fitnesses)

        # Then adapt noise std based on fitness variance
        fitness_std = jnp.std(fitnesses)
        current_std = state.internal.get('noise_std', self.noise_std)

        # Increase std if fitness variance is low (stuck), decrease if high
        target_std = current_std * (1.0 + 0.1 * (0.5 - fitness_std))
        new_std = current_std + self.std_learning_rate * (target_std - current_std)
        new_std = jnp.clip(new_std, 0.001, 1.0)

        new_state.internal = {'noise_std': float(new_std)}
        return new_state


# Dictionary of built-in ES optimizers
ES_OPTIMIZERS = {
    "es": ES,
    "openes": OpenES,
    "pepg": PEPG,
}
