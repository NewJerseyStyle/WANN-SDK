"""
Zero-Cost Proxies for Architecture Evaluation

This module provides zero-cost proxy metrics for evaluating neural network
architectures without training. These can be used standalone or integrated
with WANN search to find architectures that are not only weight-agnostic
but also trainable.

Proxies implemented:
- Synflow: Data-agnostic, measures parameter importance via gradient flow
- NASWOT: Activation-based, measures expressivity via Jacobian diversity
- SNIP: Gradient-based, measures parameter importance with data
- Trainability: Measures gradient flow stability (inspired by AZ-NAS)

Usage:
    >>> from wann_sdk.zero_cost_proxies import (
    ...     compute_synflow, compute_naswot, compute_snip,
    ...     compute_trainability, ZCPEvaluator
    ... )
    >>>
    >>> # Single proxy
    >>> score = compute_synflow(network, input_shape)
    >>>
    >>> # Multi-proxy evaluation
    >>> evaluator = ZCPEvaluator(proxies=['synflow', 'naswot', 'trainability'])
    >>> scores = evaluator.evaluate(network, data_batch)
"""

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
    TrainabilityMetrics,
)

from .evaluator import (
    ZCPEvaluator,
    ZCPConfig,
    aggregate_scores,
)

__all__ = [
    # Gradient-based
    'compute_synflow',
    'compute_snip',
    'compute_fisher',
    'compute_grasp',

    # Activation-based
    'compute_naswot',
    'compute_expressivity',

    # Trainability
    'compute_trainability',
    'compute_progressivity',
    'TrainabilityMetrics',

    # Evaluator
    'ZCPEvaluator',
    'ZCPConfig',
    'aggregate_scores',
]
