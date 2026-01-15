"""
Gradient-Based Zero-Cost Proxies

Implements proxies that use gradient information to evaluate architectures:
- Synflow: Parameter flow without data
- SNIP: Single-shot network pruning
- Fisher: Fisher information approximation
- GraSP: Gradient signal preservation
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from typing import Callable, Dict, Optional, Tuple, Any
from functools import partial


def compute_synflow(
    forward_fn: Callable,
    params: Dict[str, jnp.ndarray],
    input_shape: Tuple[int, ...],
    key: Optional[jax.random.PRNGKey] = None,
) -> float:
    """
    Compute Synflow score (data-agnostic).

    Synflow measures the "synaptic flow" through the network by computing
    the product of all parameters and its gradient. This approximates
    network expressivity without requiring data.

    Score = Σ |θ ⊙ ∇(Π|θ|)|

    Args:
        forward_fn: Network forward function (params, x) -> y
        params: Network parameters as pytree
        input_shape: Input tensor shape (excluding batch)
        key: Random key (optional, uses default if None)

    Returns:
        Synflow score (higher = better)

    Reference:
        Tanaka et al. "Pruning neural networks without any data" (2020)
    """
    key = key if key is not None else jax.random.PRNGKey(0)

    # Create all-ones input (data-agnostic)
    x = jnp.ones((1,) + input_shape)

    # Make all parameters positive for product
    def make_positive(p):
        return jnp.abs(p) + 1e-10

    positive_params = jax.tree_util.tree_map(make_positive, params)

    # Synflow loss: product of all outputs (approximates parameter flow)
    def synflow_loss(params):
        output = forward_fn(params, x)
        return jnp.sum(output)

    # Compute gradient
    grads = grad(synflow_loss)(positive_params)

    # Score = sum of |param * grad|
    def score_leaf(p, g):
        return jnp.sum(jnp.abs(p * g))

    scores = jax.tree_util.tree_map(score_leaf, positive_params, grads)
    total_score = sum(jax.tree_util.tree_leaves(scores))

    return float(total_score)


def compute_snip(
    forward_fn: Callable,
    params: Dict[str, jnp.ndarray],
    x_batch: jnp.ndarray,
    y_batch: jnp.ndarray,
    loss_fn: Callable = None,
) -> float:
    """
    Compute SNIP score (data-dependent).

    SNIP measures the sensitivity of the loss to parameter removal,
    indicating which parameters are most important for the task.

    Score = Σ |θ ⊙ ∇L(θ)|

    Args:
        forward_fn: Network forward function (params, x) -> y
        params: Network parameters as pytree
        x_batch: Input data batch
        y_batch: Target labels batch
        loss_fn: Loss function (pred, target) -> scalar (default: MSE)

    Returns:
        SNIP score (higher = more important parameters)

    Reference:
        Lee et al. "SNIP: Single-shot Network Pruning" (2018)
    """
    if loss_fn is None:
        loss_fn = lambda pred, target: jnp.mean((pred - target) ** 2)

    def task_loss(params):
        pred = forward_fn(params, x_batch)
        return loss_fn(pred, y_batch)

    grads = grad(task_loss)(params)

    def score_leaf(p, g):
        return jnp.sum(jnp.abs(p * g))

    scores = jax.tree_util.tree_map(score_leaf, params, grads)
    total_score = sum(jax.tree_util.tree_leaves(scores))

    return float(total_score)


def compute_fisher(
    forward_fn: Callable,
    params: Dict[str, jnp.ndarray],
    x_batch: jnp.ndarray,
    y_batch: jnp.ndarray,
    loss_fn: Callable = None,
) -> float:
    """
    Compute Fisher information score.

    Uses the diagonal of the Fisher information matrix to estimate
    parameter importance. This is related to the curvature of the
    loss landscape around the current parameters.

    Score = Σ (∇L(θ))²

    Args:
        forward_fn: Network forward function (params, x) -> y
        params: Network parameters as pytree
        x_batch: Input data batch
        y_batch: Target labels batch
        loss_fn: Loss function (pred, target) -> scalar

    Returns:
        Fisher score (higher = more informative parameters)

    Reference:
        Theis et al. "Faster gaze prediction with dense networks" (2018)
    """
    if loss_fn is None:
        loss_fn = lambda pred, target: jnp.mean((pred - target) ** 2)

    def task_loss(params):
        pred = forward_fn(params, x_batch)
        return loss_fn(pred, y_batch)

    grads = grad(task_loss)(params)

    # Fisher = squared gradients
    def score_leaf(g):
        return jnp.sum(g ** 2)

    scores = jax.tree_util.tree_map(score_leaf, grads)
    total_score = sum(jax.tree_util.tree_leaves(scores))

    return float(total_score)


def compute_grasp(
    forward_fn: Callable,
    params: Dict[str, jnp.ndarray],
    x_batch: jnp.ndarray,
    y_batch: jnp.ndarray,
    loss_fn: Callable = None,
) -> float:
    """
    Compute GraSP score (Gradient Signal Preservation).

    GraSP uses Hessian-gradient products to identify parameters that
    preserve gradient flow during training. This helps find architectures
    that train well.

    Score = -∇²L(θ)·∇L(θ)

    Note: This is an approximation using finite differences to avoid
    computing the full Hessian.

    Args:
        forward_fn: Network forward function (params, x) -> y
        params: Network parameters as pytree
        x_batch: Input data batch
        y_batch: Target labels batch
        loss_fn: Loss function

    Returns:
        GraSP score (higher = better gradient preservation)

    Reference:
        Wang et al. "Picking Winning Tickets Before Training" (2020)
    """
    if loss_fn is None:
        loss_fn = lambda pred, target: jnp.mean((pred - target) ** 2)

    def task_loss(params):
        pred = forward_fn(params, x_batch)
        return loss_fn(pred, y_batch)

    # First gradient
    grads = grad(task_loss)(params)

    # Approximate Hessian-vector product using finite differences
    eps = 1e-4

    def add_scaled(p, g, scale):
        return jax.tree_util.tree_map(lambda a, b: a + scale * b, p, g)

    params_plus = add_scaled(params, grads, eps)
    params_minus = add_scaled(params, grads, -eps)

    grads_plus = grad(task_loss)(params_plus)
    grads_minus = grad(task_loss)(params_minus)

    # Hv ≈ (∇L(θ+εg) - ∇L(θ-εg)) / (2ε)
    def hessian_grad_product(g_plus, g_minus, g):
        hv = (g_plus - g_minus) / (2 * eps)
        return -jnp.sum(hv * g)

    scores = jax.tree_util.tree_map(hessian_grad_product, grads_plus, grads_minus, grads)
    total_score = sum(jax.tree_util.tree_leaves(scores))

    return float(total_score)


def compute_gradient_norm(
    forward_fn: Callable,
    params: Dict[str, jnp.ndarray],
    x_batch: jnp.ndarray,
    y_batch: jnp.ndarray,
    loss_fn: Callable = None,
) -> float:
    """
    Compute gradient L2 norm.

    Simple proxy that measures the magnitude of gradients.
    Large gradients may indicate good learning signal.

    Args:
        forward_fn: Network forward function
        params: Network parameters
        x_batch: Input batch
        y_batch: Target batch
        loss_fn: Loss function

    Returns:
        L2 norm of gradients
    """
    if loss_fn is None:
        loss_fn = lambda pred, target: jnp.mean((pred - target) ** 2)

    def task_loss(params):
        pred = forward_fn(params, x_batch)
        return loss_fn(pred, y_batch)

    grads = grad(task_loss)(params)

    # L2 norm
    squared_norms = jax.tree_util.tree_map(lambda g: jnp.sum(g ** 2), grads)
    total_norm = jnp.sqrt(sum(jax.tree_util.tree_leaves(squared_norms)))

    return float(total_norm)
