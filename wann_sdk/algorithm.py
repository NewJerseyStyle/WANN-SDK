"""
WANN Algorithm Implementation

Provides Weight Agnostic Neural Networks algorithm based on TensorNEAT.
For full NEAT-based architecture search, install tensorneat package.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Callable, Optional, Any
from functools import partial

# Try to import TensorNEAT components
try:
    from tensorneat.algorithm.neat import NEAT
    from tensorneat.genome import DefaultGenome, BiasNode
    from tensorneat.common import State, ACT, AGG
    TENSORNEAT_AVAILABLE = True
except ImportError:
    TENSORNEAT_AVAILABLE = False
    NEAT = object
    DefaultGenome = object
    State = None


def _register_activations():
    """Register custom activation functions if TensorNEAT is available."""
    if not TENSORNEAT_AVAILABLE:
        return

    def cos(x):
        return jnp.cos(x)

    def gaussian(x):
        return jnp.exp(-jnp.square(x) / 2.0)

    def step(x):
        return jnp.heaviside(x, 0.5)

    try:
        ACT.add_func("cos", cos)
        ACT.add_func("gaussian", gaussian)
        ACT.add_func("step", step)
    except Exception:
        pass  # Already registered


_register_activations()


class WANNGenome(DefaultGenome if TENSORNEAT_AVAILABLE else object):
    """
    WANN-specific genome that evaluates networks with shared weights.

    This extends the DefaultGenome from TensorNEAT to support weight-agnostic
    evaluation where all connections share the same weight value.

    Args:
        num_inputs: Number of input nodes
        num_outputs: Number of output nodes
        weight_range: Range for weight sampling (min, max)
        weight_samples: Fixed weight values to test during evaluation
        **kwargs: Additional arguments passed to DefaultGenome

    Example:
        >>> genome = WANNGenome(
        ...     num_inputs=24,
        ...     num_outputs=4,
        ...     weight_samples=jnp.array([-1.0, 0.0, 1.0]),
        ... )
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        weight_range: Tuple[float, float] = (-2.0, 2.0),
        weight_samples: Optional[jnp.ndarray] = None,
        **kwargs
    ):
        if not TENSORNEAT_AVAILABLE:
            raise ImportError(
                "TensorNEAT is required for WANNGenome. "
                "Install with: pip install wann-sdk[tensorneat]"
            )

        if weight_samples is None:
            weight_samples = jnp.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])

        super().__init__(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            **kwargs
        )
        self.weight_range = weight_range
        self.weight_samples = weight_samples
        self.num_weight_samples = len(weight_samples)

    def transform(self, state: Any, nodes: jnp.ndarray, conns: jnp.ndarray) -> Tuple:
        """
        Transform genome representation for forward pass.

        Args:
            state: Algorithm state
            nodes: Node array
            conns: Connection array

        Returns:
            Tuple of (seqs, nodes, conns, u_conns)
        """
        from tensorneat.genome.utils import unflatten_conns
        from tensorneat.common import topological_sort, I_INF

        u_conns = unflatten_conns(nodes, conns)
        conn_exist = u_conns != I_INF
        seqs = topological_sort(nodes, conn_exist)

        return seqs, nodes, conns, u_conns

    def forward_with_shared_weight(
        self,
        state: Any,
        transformed: Tuple,
        x: jnp.ndarray,
        shared_weight: float
    ) -> jnp.ndarray:
        """
        Forward pass with all connections using the same shared weight.

        Args:
            state: Algorithm state
            transformed: Transformed network parameters tuple
            x: Input array
            shared_weight: Single weight value applied to all connections

        Returns:
            Network output
        """
        seqs, nodes, conns, u_conns = transformed

        # Update all connection weights
        conns_updated = conns.at[:, 2].set(shared_weight)
        transformed_updated = (seqs, nodes, conns_updated, u_conns)

        return super().forward(state, transformed_updated, x)

    def evaluate_with_shared_weights(
        self,
        transformed: Tuple,
        state: Any,
        inputs: jnp.ndarray,
        fitness_fn: Callable,
        weight_samples: Optional[jnp.ndarray] = None
    ) -> Tuple[float, float, jnp.ndarray]:
        """
        Evaluate network performance across multiple shared weight values.

        Args:
            transformed: Transformed network parameters
            state: Algorithm state
            inputs: Input samples for evaluation
            fitness_fn: Function to compute fitness from predictions
            weight_samples: Weight values to test

        Returns:
            mean_fitness: Average fitness across all weight samples
            max_fitness: Best fitness across all weight samples
            fitness_per_weight: Fitness for each weight sample
        """
        if weight_samples is None:
            weight_samples = self.weight_samples

        from jax import vmap

        def evaluate_single_weight(weight):
            forward_fn = partial(
                self.forward_with_shared_weight,
                transformed=transformed,
                state=state,
                shared_weight=weight
            )
            predictions = vmap(forward_fn)(inputs)
            fitness = fitness_fn(predictions)
            return fitness

        fitness_per_weight = vmap(evaluate_single_weight)(weight_samples)

        mean_fitness = jnp.mean(fitness_per_weight)
        max_fitness = jnp.max(fitness_per_weight)

        return mean_fitness, max_fitness, fitness_per_weight


class WANN(NEAT if TENSORNEAT_AVAILABLE else object):
    """
    Weight Agnostic Neural Networks algorithm.

    Extends NEAT to search for architectures that perform well
    with random/shared weights rather than optimized weights.

    Args:
        pop_size: Population size
        species_size: Target number of species
        survival_threshold: Fraction of population to survive
        compatibility_threshold: Threshold for speciation
        genome: WANN genome instance
        complexity_weight: Weight for complexity penalty (0-1)
        use_max_fitness: If True, rank by max fitness instead of mean
        **kwargs: Additional arguments passed to NEAT

    Example:
        >>> genome = WANNGenome(num_inputs=24, num_outputs=4)
        >>> wann = WANN(pop_size=1000, genome=genome)
    """

    def __init__(
        self,
        pop_size: int,
        species_size: int = 20,
        survival_threshold: float = 0.1,
        compatibility_threshold: float = 1.0,
        genome: Optional[WANNGenome] = None,
        complexity_weight: float = 0.2,
        use_max_fitness: bool = False,
        **kwargs
    ):
        if not TENSORNEAT_AVAILABLE:
            raise ImportError(
                "TensorNEAT is required for WANN. "
                "Install with: pip install wann-sdk[tensorneat]"
            )

        if genome is None:
            genome = WANNGenome(
                num_inputs=1,
                num_outputs=1,
                node_gene=BiasNode(
                    activation_options=[
                        ACT.identity, ACT.sigmoid, ACT.tanh,
                        ACT.relu, ACT.sin, ACT.cos,
                        ACT.gaussian, ACT.step, ACT.inv, ACT.abs
                    ],
                    aggregation_options=AGG.sum,
                ),
            )

        super().__init__(
            pop_size=pop_size,
            species_size=species_size,
            survival_threshold=survival_threshold,
            compatibility_threshold=compatibility_threshold,
            genome=genome,
            **kwargs
        )

        self.complexity_weight = complexity_weight
        self.use_max_fitness = use_max_fitness
