"""
Trainability Metrics for Architecture Evaluation

Measures how well an architecture can be trained with gradient descent.
Inspired by AZ-NAS trainability and progressivity components.

Key metrics:
- Trainability: Gradient flow stability across layers
- Progressivity: Layer-wise activation progression
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from typing import Callable, Dict, Optional, Tuple, List, Any, NamedTuple
from dataclasses import dataclass


@dataclass
class TrainabilityMetrics:
    """Container for trainability-related metrics."""
    trainability: float       # Gradient flow stability
    progressivity: float      # Layer activation change
    gradient_variance: float  # Gradient variance across params
    signal_to_noise: float    # Gradient SNR


def compute_trainability(
    forward_fn: Callable,
    params: Dict[str, jnp.ndarray],
    x_batch: jnp.ndarray,
    y_batch: jnp.ndarray,
    loss_fn: Callable = None,
) -> float:
    """
    Compute trainability score (gradient flow stability).

    Measures how stable gradients are across different parts of the network.
    Architectures with stable gradient flow train better.

    The score is based on the coefficient of variation (CV) of gradient
    magnitudes across layers. Lower CV = more stable = better trainability.

    Score = 1 / (1 + CV)  where CV = std(grads) / mean(grads)

    Args:
        forward_fn: Network forward function (params, x) -> y
        params: Network parameters as pytree
        x_batch: Input data batch
        y_batch: Target labels batch
        loss_fn: Loss function (pred, target) -> scalar

    Returns:
        Trainability score (0-1, higher = better)

    Reference:
        Inspired by AZ-NAS trainability component (Lee & Ham, CVPR 2024)
    """
    if loss_fn is None:
        loss_fn = lambda pred, target: jnp.mean((pred - target) ** 2)

    def task_loss(params):
        pred = forward_fn(params, x_batch)
        return loss_fn(pred, y_batch)

    grads = grad(task_loss)(params)

    # Compute gradient magnitude for each parameter group
    def compute_magnitude(g):
        return jnp.sqrt(jnp.mean(g ** 2) + 1e-10)

    magnitudes = jax.tree_util.tree_map(compute_magnitude, grads)
    mag_list = jnp.array(jax.tree_util.tree_leaves(magnitudes))

    if len(mag_list) < 2:
        return 0.5  # Single param, neutral score

    # Filter out near-zero magnitudes
    mag_list = mag_list[mag_list > 1e-8]

    if len(mag_list) < 2:
        return 0.1  # Very low gradients, poor trainability

    # Compute coefficient of variation
    mean_mag = jnp.mean(mag_list)
    std_mag = jnp.std(mag_list)

    cv = std_mag / (mean_mag + 1e-10)

    # Transform to 0-1 score (lower CV = higher score)
    # Use sigmoid-like transformation
    trainability = 1.0 / (1.0 + cv)

    return float(trainability)


def compute_progressivity(
    forward_fn: Callable,
    params: Dict[str, jnp.ndarray],
    x_batch: jnp.ndarray,
    layer_outputs_fn: Callable = None,
) -> float:
    """
    Compute progressivity score (layer-wise activation change).

    Measures how much activations change from layer to layer.
    Good architectures should progressively transform inputs.

    Score is based on the mean absolute difference between
    consecutive layer activations.

    Args:
        forward_fn: Network forward function
        params: Network parameters
        x_batch: Input data batch
        layer_outputs_fn: Optional function to get intermediate activations
                         If None, uses forward_fn output only

    Returns:
        Progressivity score (higher = more progressive transformation)

    Reference:
        Inspired by AZ-NAS progressivity component (Lee & Ham, CVPR 2024)
    """
    # Get final output
    output = forward_fn(params, x_batch)

    # Flatten for comparison
    x_flat = x_batch.reshape(x_batch.shape[0], -1)
    y_flat = output.reshape(output.shape[0], -1) if output.ndim > 1 else output

    # Compute mean change
    # Normalize to comparable scale
    x_norm = x_flat / (jnp.linalg.norm(x_flat, axis=1, keepdims=True) + 1e-8)
    y_norm = y_flat / (jnp.linalg.norm(y_flat, axis=1, keepdims=True) + 1e-8)

    # Truncate/pad to match dimensions
    min_dim = min(x_norm.shape[1], y_norm.shape[1])
    x_trunc = x_norm[:, :min_dim]
    y_trunc = y_norm[:, :min_dim]

    # Mean change per sample
    change = jnp.mean(jnp.abs(y_trunc - x_trunc), axis=1)
    avg_change = jnp.mean(change)

    # Normalize to 0-1 (change of 1 = completely different)
    progressivity = float(jnp.clip(avg_change, 0, 1))

    return progressivity


def compute_gradient_snr(
    forward_fn: Callable,
    params: Dict[str, jnp.ndarray],
    x_batch: jnp.ndarray,
    y_batch: jnp.ndarray,
    loss_fn: Callable = None,
    num_samples: int = 4,
    key: Optional[jax.random.PRNGKey] = None,
) -> float:
    """
    Compute gradient signal-to-noise ratio (SNR).

    Measures the consistency of gradients across different mini-batches.
    Higher SNR indicates more stable training signal.

    This is related to ZiCo's gradient variation coefficient.

    Args:
        forward_fn: Network forward function
        params: Network parameters
        x_batch: Input batch (will be split into mini-batches)
        y_batch: Target batch
        loss_fn: Loss function
        num_samples: Number of mini-batches to compare
        key: Random key for shuffling

    Returns:
        Gradient SNR (higher = more consistent gradients)
    """
    if loss_fn is None:
        loss_fn = lambda pred, target: jnp.mean((pred - target) ** 2)

    key = key if key is not None else jax.random.PRNGKey(42)

    batch_size = x_batch.shape[0]
    mini_batch_size = max(1, batch_size // num_samples)

    if mini_batch_size < 1 or batch_size < num_samples:
        return 0.5  # Not enough data for comparison

    # Collect gradients from different mini-batches
    all_grads = []

    for i in range(num_samples):
        start_idx = i * mini_batch_size
        end_idx = min(start_idx + mini_batch_size, batch_size)

        if start_idx >= batch_size:
            break

        x_mini = x_batch[start_idx:end_idx]
        y_mini = y_batch[start_idx:end_idx]

        def task_loss(params):
            pred = forward_fn(params, x_mini)
            return loss_fn(pred, y_mini)

        grads = grad(task_loss)(params)

        # Flatten gradients to single vector
        grad_flat = jnp.concatenate([g.flatten() for g in jax.tree_util.tree_leaves(grads)])
        all_grads.append(grad_flat)

    if len(all_grads) < 2:
        return 0.5

    # Stack gradients
    grad_matrix = jnp.stack(all_grads, axis=0)  # (num_samples, num_params)

    # Compute mean and std across samples
    mean_grad = jnp.mean(grad_matrix, axis=0)
    std_grad = jnp.std(grad_matrix, axis=0)

    # SNR = mean / std (element-wise, then average)
    snr_elements = jnp.abs(mean_grad) / (std_grad + 1e-8)
    mean_snr = jnp.mean(snr_elements)

    # Normalize to reasonable range (cap at 10 for numerical stability)
    normalized_snr = jnp.clip(mean_snr / 10.0, 0, 1)

    return float(normalized_snr)


def compute_all_trainability_metrics(
    forward_fn: Callable,
    params: Dict[str, jnp.ndarray],
    x_batch: jnp.ndarray,
    y_batch: jnp.ndarray,
    loss_fn: Callable = None,
    key: Optional[jax.random.PRNGKey] = None,
) -> TrainabilityMetrics:
    """
    Compute all trainability-related metrics.

    Args:
        forward_fn: Network forward function
        params: Network parameters
        x_batch: Input batch
        y_batch: Target batch
        loss_fn: Loss function
        key: Random key

    Returns:
        TrainabilityMetrics with all computed scores
    """
    if loss_fn is None:
        loss_fn = lambda pred, target: jnp.mean((pred - target) ** 2)

    key = key if key is not None else jax.random.PRNGKey(42)

    # Compute individual metrics
    trainability = compute_trainability(
        forward_fn, params, x_batch, y_batch, loss_fn
    )

    progressivity = compute_progressivity(
        forward_fn, params, x_batch
    )

    snr = compute_gradient_snr(
        forward_fn, params, x_batch, y_batch, loss_fn, key=key
    )

    # Gradient variance (inverse is related to trainability)
    def task_loss(params):
        pred = forward_fn(params, x_batch)
        return loss_fn(pred, y_batch)

    grads = grad(task_loss)(params)
    grad_flat = jnp.concatenate([g.flatten() for g in jax.tree_util.tree_leaves(grads)])
    grad_variance = float(jnp.var(grad_flat))

    return TrainabilityMetrics(
        trainability=trainability,
        progressivity=progressivity,
        gradient_variance=grad_variance,
        signal_to_noise=snr,
    )
