"""
Trainability-Aware Architecture Search

Extends WANN search to find architectures that are not only weight-agnostic
but also trainable - meaning they can be further optimized through weight training.

Three integration strategies:
1. Sequential: WANN fitness first, then ZCP refinement
2. Hybrid: Combined fitness function
3. Parallel: Multi-objective optimization (Pareto)

Usage:
    >>> from wann_sdk import TrainabilityAwareSearch, SearchConfig
    >>> from wann_sdk import SupervisedProblem
    >>>
    >>> problem = SupervisedProblem(x_train, y_train, loss_fn='mse')
    >>> search = TrainabilityAwareSearch(
    ...     problem,
    ...     SearchConfig(max_nodes=30),
    ...     strategy='hybrid',
    ...     zcp_weight=0.5,  # Balance WANN and ZCP fitness
    ... )
    >>> genome = search.run(generations=100)
"""

import jax
import jax.numpy as jnp
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field

from .search import ArchitectureSearch, SearchConfig, NetworkGenome
from .zero_cost_proxies import ZCPEvaluator, ZCPConfig


@dataclass
class TrainabilitySearchConfig:
    """Configuration for trainability-aware search.

    Args:
        strategy: Integration strategy ('sequential', 'hybrid', 'parallel')
        zcp_proxies: List of ZCP proxies to use
        zcp_weight: Weight for ZCP fitness in hybrid mode (0-1)
        wann_weight: Weight for WANN fitness (1 - zcp_weight by default)
        dynamic_weight: Whether to adjust weights during evolution
        zcp_batch_size: Batch size for ZCP evaluation
        filter_ratio: Ratio for sequential filtering (keep top N%)
    """
    strategy: str = 'hybrid'
    zcp_proxies: List[str] = field(default_factory=lambda: ['synflow', 'naswot', 'trainability'])
    zcp_weight: float = 0.3
    wann_weight: Optional[float] = None  # Computed as 1 - zcp_weight if None
    dynamic_weight: bool = False
    zcp_batch_size: int = 32
    filter_ratio: float = 0.3  # For sequential strategy


class TrainabilityAwareSearch(ArchitectureSearch):
    """
    Architecture search that considers both weight-agnosticism and trainability.

    This extends standard WANN search to find architectures that:
    1. Perform well with shared weights (WANN fitness)
    2. Have good trainability metrics (can be further optimized)

    The balance between these objectives is controlled by zcp_weight.
    """

    def __init__(
        self,
        problem: Any,
        config: SearchConfig = None,
        strategy: str = 'hybrid',
        zcp_weight: float = 0.3,
        zcp_proxies: List[str] = None,
        trainability_config: TrainabilitySearchConfig = None,
    ):
        """
        Initialize trainability-aware search.

        Args:
            problem: Problem instance (SupervisedProblem, RLProblem, etc.)
            config: Search configuration
            strategy: Integration strategy
            zcp_weight: Weight for ZCP fitness (0-1)
            zcp_proxies: List of ZCP proxies to compute
            trainability_config: Full trainability configuration (overrides other params)
        """
        super().__init__(problem, config or SearchConfig())

        # Use provided config or create from parameters
        if trainability_config:
            self.trainability_config = trainability_config
        else:
            self.trainability_config = TrainabilitySearchConfig(
                strategy=strategy,
                zcp_weight=zcp_weight,
                zcp_proxies=zcp_proxies or ['synflow', 'naswot', 'trainability'],
            )

        # Compute WANN weight if not specified
        if self.trainability_config.wann_weight is None:
            self.trainability_config.wann_weight = 1.0 - self.trainability_config.zcp_weight

        # Initialize ZCP evaluator
        self.zcp_evaluator = ZCPEvaluator(
            proxies=self.trainability_config.zcp_proxies,
            aggregation='geometric',
            normalize=True,
        )

        # Cache for ZCP scores
        self._zcp_cache: Dict[int, Dict[str, float]] = {}

        # Generation counter for dynamic weights
        self._current_generation = 0

    def run(self, generations: int = 100, verbose: bool = True) -> NetworkGenome:
        """
        Run trainability-aware architecture search.

        Args:
            generations: Number of evolution generations
            verbose: Whether to print progress

        Returns:
            Best genome found
        """
        if verbose:
            print(f"Running trainability-aware search with strategy: "
                  f"{self.trainability_config.strategy}")
            print(f"  ZCP weight: {self.trainability_config.zcp_weight:.2f}")
            print(f"  WANN weight: {self.trainability_config.wann_weight:.2f}")
            print(f"  Proxies: {self.trainability_config.zcp_proxies}")

        return super().run(generations, verbose)

    def _evaluate_genome(
        self,
        genome: NetworkGenome,
        seed: int = 0,
    ) -> Tuple[float, int]:
        """
        Evaluate genome with combined WANN + ZCP fitness.

        Args:
            genome: Genome to evaluate
            seed: Random seed

        Returns:
            (combined_fitness, complexity)
        """
        # Get WANN fitness from parent class
        wann_fitness, complexity = super()._evaluate_genome(genome, seed)

        if self.trainability_config.strategy == 'sequential':
            # Sequential: only compute ZCP for top candidates
            # This is handled at the population level
            return wann_fitness, complexity

        # Compute ZCP score
        zcp_score = self._compute_zcp_score(genome, seed)

        # Combine scores
        if self.trainability_config.dynamic_weight:
            # Adjust weights based on generation
            # Start with more WANN weight, gradually increase ZCP
            progress = self._current_generation / max(1, 100)  # Assume 100 generations
            zcp_weight = self.trainability_config.zcp_weight * (0.5 + 0.5 * progress)
            wann_weight = 1.0 - zcp_weight
        else:
            zcp_weight = self.trainability_config.zcp_weight
            wann_weight = self.trainability_config.wann_weight

        combined_fitness = wann_weight * wann_fitness + zcp_weight * zcp_score

        return combined_fitness, complexity

    def _compute_zcp_score(self, genome: NetworkGenome, seed: int = 0) -> float:
        """
        Compute ZCP score for a genome.

        Args:
            genome: Genome to evaluate
            seed: Random seed

        Returns:
            Aggregated ZCP score
        """
        # Check cache
        genome_hash = hash(str(genome))
        if genome_hash in self._zcp_cache:
            return self._zcp_cache[genome_hash]['aggregated']

        try:
            # Build network from genome
            network = self._build_network(genome)
            weights = network.get_params()

            # Get sample batch from problem
            x_batch, y_batch = self._get_sample_batch()

            # Create forward function matching ZCP evaluator signature (params, x)
            def forward_fn(params, x):
                return network.forward(params, x)

            # Evaluate with ZCP
            key = jax.random.PRNGKey(seed)
            scores = self.zcp_evaluator.evaluate(
                forward_fn, weights, x_batch, y_batch,
                loss_fn=self.problem.loss_fn if hasattr(self.problem, 'loss_fn') else None,
                key=key,
            )

            # Cache result
            self._zcp_cache[genome_hash] = scores

            return scores.get('aggregated', 0.5)

        except Exception as e:
            # On error, return neutral score
            return 0.5

    def _get_sample_batch(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get a sample batch from the problem for ZCP evaluation."""
        batch_size = self.trainability_config.zcp_batch_size

        # Handle different problem types
        if hasattr(self.problem, 'x_train') and hasattr(self.problem, 'y_train'):
            # Supervised problem
            x = self.problem.x_train[:batch_size]
            y = self.problem.y_train[:batch_size]
            return x, y

        elif hasattr(self.problem, 'get_batch'):
            # Custom problem with batch method
            return self.problem.get_batch(batch_size)

        else:
            # RL problem - generate random inputs
            input_dim = getattr(self.problem, 'input_dim', 4)
            output_dim = getattr(self.problem, 'output_dim', 2)
            x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, input_dim))
            y = jax.random.normal(jax.random.PRNGKey(1), (batch_size, output_dim))
            return x, y

    def _build_network(self, genome: NetworkGenome) -> Any:
        """Build network from genome for ZCP evaluation."""
        # Use TrainableNetwork for proper weight handling
        from .weight_trainer import TrainableNetwork

        return TrainableNetwork(
            genome=genome,
            activation_options=self.config.activation_options,
            init_weight=1.0,
            key=jax.random.PRNGKey(42),
        )

    def _evaluate_population_sequential(
        self,
        population: List[NetworkGenome],
        base_seed: int,
    ) -> List[Tuple[float, int]]:
        """
        Sequential strategy: WANN filter then ZCP refinement.

        Args:
            population: Population of genomes
            base_seed: Base random seed

        Returns:
            List of (fitness, complexity) tuples
        """
        # Stage 1: Evaluate all with WANN
        wann_results = []
        for i, genome in enumerate(population):
            fitness, complexity = super()._evaluate_genome(genome, base_seed + i)
            wann_results.append((genome, fitness, complexity))

        # Sort by WANN fitness
        wann_results.sort(key=lambda x: x[1], reverse=True)

        # Stage 2: Compute ZCP for top candidates
        n_top = int(len(population) * self.trainability_config.filter_ratio)
        n_top = max(1, n_top)

        results = []
        for i, (genome, wann_fitness, complexity) in enumerate(wann_results):
            if i < n_top:
                # Compute ZCP for top candidates
                zcp_score = self._compute_zcp_score(genome, base_seed + i)
                combined = (self.trainability_config.wann_weight * wann_fitness +
                           self.trainability_config.zcp_weight * zcp_score)
            else:
                # Use WANN fitness only for others
                combined = wann_fitness

            results.append((combined, complexity))

        return results

    def get_zcp_breakdown(self, genome: NetworkGenome) -> Dict[str, float]:
        """
        Get detailed ZCP scores for a genome.

        Args:
            genome: Genome to analyze

        Returns:
            Dictionary with individual proxy scores
        """
        genome_hash = hash(str(genome))
        if genome_hash in self._zcp_cache:
            return self._zcp_cache[genome_hash]

        # Compute fresh
        self._compute_zcp_score(genome)
        return self._zcp_cache.get(genome_hash, {})


def create_hybrid_fitness(
    wann_fn: Callable,
    zcp_evaluator: ZCPEvaluator,
    wann_weight: float = 0.7,
    zcp_weight: float = 0.3,
) -> Callable:
    """
    Create a hybrid fitness function combining WANN and ZCP.

    Args:
        wann_fn: Original WANN fitness function
        zcp_evaluator: ZCP evaluator
        wann_weight: Weight for WANN fitness
        zcp_weight: Weight for ZCP fitness

    Returns:
        Hybrid fitness function
    """
    def hybrid_fitness(genome, problem, seed=0):
        # WANN fitness
        wann_score = wann_fn(genome, problem, seed)

        # ZCP fitness
        try:
            network = problem.build_network(genome)
            x_batch, y_batch = problem.get_batch(32)

            def forward_fn(params, x):
                return network.forward(x, params)

            key = jax.random.PRNGKey(seed)
            zcp_scores = zcp_evaluator.evaluate(
                forward_fn, network.get_params(),
                x_batch, y_batch, key=key
            )
            zcp_score = zcp_scores['aggregated']
        except Exception:
            zcp_score = 0.5

        return wann_weight * wann_score + zcp_weight * zcp_score

    return hybrid_fitness
