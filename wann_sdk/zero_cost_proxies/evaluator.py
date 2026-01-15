"""
Zero-Cost Proxy Evaluator

Unified interface for computing and aggregating multiple zero-cost proxies.
Supports hierarchical filtering and custom aggregation strategies.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from .gradient_based import (
    compute_synflow,
    compute_snip,
    compute_fisher,
    compute_grasp,
)
from .activation_based import (
    compute_naswot,
    compute_expressivity,
)
from .trainability import (
    compute_trainability,
    compute_progressivity,
    compute_gradient_snr,
)


class AggregationMethod(Enum):
    """Methods for aggregating multiple proxy scores."""
    MEAN = 'mean'                    # Simple average
    WEIGHTED = 'weighted'            # Weighted average
    GEOMETRIC = 'geometric'          # Geometric mean
    RANK = 'rank'                    # Rank-based (percentile)
    HARMONIC = 'harmonic'            # Harmonic mean


@dataclass
class ZCPConfig:
    """Configuration for zero-cost proxy evaluation.

    Args:
        proxies: List of proxy names to compute
        aggregation: Method for combining scores
        weights: Optional weights for weighted aggregation
        normalize: Whether to normalize scores to [0, 1]
        batch_size: Batch size for data-dependent proxies
    """
    proxies: List[str] = field(default_factory=lambda: ['synflow', 'naswot', 'trainability'])
    aggregation: str = 'geometric'
    weights: Optional[Dict[str, float]] = None
    normalize: bool = True
    batch_size: int = 32


# Available proxies categorized by type
GRADIENT_PROXIES = {'synflow', 'snip', 'fisher', 'grasp'}
ACTIVATION_PROXIES = {'naswot', 'expressivity'}
TRAINABILITY_PROXIES = {'trainability', 'progressivity', 'gradient_snr'}

# Data requirements
DATA_DEPENDENT = {'snip', 'fisher', 'grasp', 'naswot', 'expressivity',
                  'trainability', 'progressivity', 'gradient_snr'}
DATA_AGNOSTIC = {'synflow'}

ALL_PROXIES = GRADIENT_PROXIES | ACTIVATION_PROXIES | TRAINABILITY_PROXIES


class ZCPEvaluator:
    """
    Unified evaluator for zero-cost proxies.

    Computes multiple proxies and aggregates them into a single score.
    Supports both data-dependent and data-agnostic evaluation.

    Example:
        >>> evaluator = ZCPEvaluator(
        ...     proxies=['synflow', 'naswot', 'trainability'],
        ...     aggregation='geometric'
        ... )
        >>> scores = evaluator.evaluate(forward_fn, params, x_batch, y_batch)
        >>> final_score = scores['aggregated']
    """

    def __init__(
        self,
        proxies: List[str] = None,
        aggregation: str = 'geometric',
        weights: Optional[Dict[str, float]] = None,
        normalize: bool = True,
    ):
        """
        Initialize evaluator.

        Args:
            proxies: List of proxy names (default: synflow, naswot, trainability)
            aggregation: Aggregation method ('mean', 'weighted', 'geometric', 'rank', 'harmonic')
            weights: Weights for each proxy (for weighted aggregation)
            normalize: Whether to normalize scores
        """
        self.proxies = proxies or ['synflow', 'naswot', 'trainability']
        self.aggregation = aggregation
        self.weights = weights or {p: 1.0 for p in self.proxies}
        self.normalize = normalize

        # Validate proxy names
        for p in self.proxies:
            if p not in ALL_PROXIES:
                raise ValueError(f"Unknown proxy: {p}. Available: {ALL_PROXIES}")

        # Check if data is needed
        self.requires_data = any(p in DATA_DEPENDENT for p in self.proxies)
        self.requires_labels = any(p in GRADIENT_PROXIES - {'synflow'} or
                                   p in TRAINABILITY_PROXIES
                                   for p in self.proxies)

    def evaluate(
        self,
        forward_fn: Callable,
        params: Dict[str, jnp.ndarray],
        x_batch: Optional[jnp.ndarray] = None,
        y_batch: Optional[jnp.ndarray] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        loss_fn: Optional[Callable] = None,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Dict[str, float]:
        """
        Evaluate architecture using configured proxies.

        Args:
            forward_fn: Network forward function (params, x) -> y
            params: Network parameters
            x_batch: Input data batch (required for data-dependent proxies)
            y_batch: Labels (required for gradient-based proxies)
            input_shape: Input shape for synflow (if x_batch not provided)
            loss_fn: Loss function for gradient-based proxies
            key: Random key

        Returns:
            Dictionary with individual proxy scores and aggregated score
        """
        key = key if key is not None else jax.random.PRNGKey(42)

        # Validate inputs
        if self.requires_data and x_batch is None:
            raise ValueError("x_batch required for data-dependent proxies")
        if self.requires_labels and y_batch is None:
            raise ValueError("y_batch required for gradient-based proxies")

        # Infer input shape
        if input_shape is None and x_batch is not None:
            input_shape = x_batch.shape[1:]

        scores = {}

        for proxy in self.proxies:
            try:
                if proxy == 'synflow':
                    scores[proxy] = compute_synflow(
                        forward_fn, params, input_shape, key
                    )
                elif proxy == 'snip':
                    scores[proxy] = compute_snip(
                        forward_fn, params, x_batch, y_batch, loss_fn
                    )
                elif proxy == 'fisher':
                    scores[proxy] = compute_fisher(
                        forward_fn, params, x_batch, y_batch, loss_fn
                    )
                elif proxy == 'grasp':
                    scores[proxy] = compute_grasp(
                        forward_fn, params, x_batch, y_batch, loss_fn
                    )
                elif proxy == 'naswot':
                    scores[proxy] = compute_naswot(
                        forward_fn, params, x_batch, key
                    )
                elif proxy == 'expressivity':
                    scores[proxy] = compute_expressivity(
                        forward_fn, params, x_batch
                    )
                elif proxy == 'trainability':
                    scores[proxy] = compute_trainability(
                        forward_fn, params, x_batch, y_batch, loss_fn
                    )
                elif proxy == 'progressivity':
                    scores[proxy] = compute_progressivity(
                        forward_fn, params, x_batch
                    )
                elif proxy == 'gradient_snr':
                    scores[proxy] = compute_gradient_snr(
                        forward_fn, params, x_batch, y_batch, loss_fn, key=key
                    )
            except Exception as e:
                # Handle errors gracefully
                scores[proxy] = 0.0

        # Normalize if requested
        if self.normalize:
            scores = self._normalize_scores(scores)

        # Aggregate
        scores['aggregated'] = aggregate_scores(
            scores, self.aggregation, self.weights
        )

        return scores

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to [0, 1] range using sigmoid."""
        normalized = {}
        for k, v in scores.items():
            if k == 'aggregated':
                continue
            # Use sigmoid for unbounded scores
            if abs(v) > 10:
                v = 10.0 * jnp.sign(v)  # Clip extreme values
            normalized[k] = float(1 / (1 + jnp.exp(-v)))
        return normalized

    def evaluate_population(
        self,
        population: List[Tuple[Callable, Dict]],
        x_batch: Optional[jnp.ndarray] = None,
        y_batch: Optional[jnp.ndarray] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        loss_fn: Optional[Callable] = None,
    ) -> List[Dict[str, float]]:
        """
        Evaluate a population of architectures.

        Args:
            population: List of (forward_fn, params) tuples
            x_batch: Input batch
            y_batch: Labels
            input_shape: Input shape
            loss_fn: Loss function

        Returns:
            List of score dictionaries for each architecture
        """
        results = []
        for i, (forward_fn, params) in enumerate(population):
            key = jax.random.PRNGKey(i)
            scores = self.evaluate(
                forward_fn, params, x_batch, y_batch, input_shape, loss_fn, key
            )
            results.append(scores)
        return results


def aggregate_scores(
    scores: Dict[str, float],
    method: str = 'geometric',
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Aggregate multiple proxy scores into a single value.

    Args:
        scores: Dictionary of proxy scores
        method: Aggregation method
        weights: Optional weights per proxy

    Returns:
        Aggregated score
    """
    # Filter out 'aggregated' key if present
    values = {k: v for k, v in scores.items() if k != 'aggregated'}

    if not values:
        return 0.0

    weights = weights or {k: 1.0 for k in values}

    if method == 'mean':
        return float(jnp.mean(jnp.array(list(values.values()))))

    elif method == 'weighted':
        total = sum(values[k] * weights.get(k, 1.0) for k in values)
        weight_sum = sum(weights.get(k, 1.0) for k in values)
        return float(total / weight_sum) if weight_sum > 0 else 0.0

    elif method == 'geometric':
        # Geometric mean (good for combining different scales)
        log_sum = sum(jnp.log(max(v, 1e-10)) * weights.get(k, 1.0)
                      for k, v in values.items())
        weight_sum = sum(weights.get(k, 1.0) for k in values)
        return float(jnp.exp(log_sum / weight_sum)) if weight_sum > 0 else 0.0

    elif method == 'harmonic':
        # Harmonic mean (emphasizes lower values)
        inv_sum = sum(weights.get(k, 1.0) / max(v, 1e-10)
                      for k, v in values.items())
        weight_sum = sum(weights.get(k, 1.0) for k in values)
        return float(weight_sum / inv_sum) if inv_sum > 0 else 0.0

    elif method == 'rank':
        # Rank-based: each proxy votes by percentile
        # This requires population context, so just use mean here
        return float(jnp.mean(jnp.array(list(values.values()))))

    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def hierarchical_filter(
    population: List[Any],
    score_fn: Callable[[Any], float],
    levels: List[int],
) -> List[Any]:
    """
    Apply hierarchical filtering to population.

    Progressively filters candidates through multiple levels,
    keeping top-k at each level.

    Args:
        population: List of candidates to filter
        score_fn: Function to compute score for each candidate
        levels: List of k values for each filtering level
                e.g., [1000, 100, 10] keeps top 1000, then 100, then 10

    Returns:
        Filtered population
    """
    candidates = list(population)

    for k in levels:
        if len(candidates) <= k:
            continue

        # Score all candidates
        scored = [(c, score_fn(c)) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Keep top k
        candidates = [c for c, s in scored[:k]]

    return candidates
