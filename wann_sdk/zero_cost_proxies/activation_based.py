"""
Activation-Based Zero-Cost Proxies

Implements proxies that use activation patterns to evaluate architectures:
- NASWOT: Measures activation diversity via Jacobian
- Expressivity: SVD entropy of activations (from AZ-NAS)
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, jacfwd
from typing import Callable, Dict, Optional, Tuple, List, Any


def compute_naswot(
    forward_fn: Callable,
    params: Dict[str, jnp.ndarray],
    x_batch: jnp.ndarray,
    key: Optional[jax.random.PRNGKey] = None,
) -> float:
    """
    Compute NASWOT score (Neural Architecture Search Without Training).

    NASWOT measures the diversity of activation patterns by computing
    the log determinant of the Jacobian covariance matrix. Higher
    diversity indicates better expressivity.

    Score = log|J·J^T|

    Args:
        forward_fn: Network forward function (params, x) -> y
        params: Network parameters
        x_batch: Input data batch (batch_size, ...)
        key: Random key for tie-breaking

    Returns:
        NASWOT score (higher = more diverse activations)

    Reference:
        Mellor et al. "Neural Architecture Search Without Training" (2021)
    """
    batch_size = x_batch.shape[0]

    if batch_size < 2:
        # Need multiple samples to measure diversity
        return 0.0

    # Compute Jacobian of output w.r.t. input for each sample
    def single_forward(x):
        x = x.reshape((1,) + x.shape)
        return forward_fn(params, x).flatten()

    # Compute activations for all samples
    outputs = vmap(single_forward)(x_batch)  # (batch, output_dim)

    # Binarize activations (ReLU-style: positive = 1, negative = 0)
    # This captures the activation pattern
    binary_codes = (outputs > 0).astype(jnp.float32)

    # Compute covariance matrix of binary codes
    # K_ij = <code_i, code_j> / dim
    K = binary_codes @ binary_codes.T / binary_codes.shape[1]

    # Add small regularization for numerical stability
    K = K + 1e-6 * jnp.eye(batch_size)

    # Score = log determinant
    # Use slogdet for numerical stability
    sign, logdet = jnp.linalg.slogdet(K)

    # If determinant is negative (shouldn't happen with regularization),
    # return a low score
    score = jnp.where(sign > 0, logdet, -100.0)

    return float(score)


def compute_expressivity(
    forward_fn: Callable,
    params: Dict[str, jnp.ndarray],
    x_batch: jnp.ndarray,
    num_singular_values: int = 10,
) -> float:
    """
    Compute expressivity score via SVD entropy (from AZ-NAS).

    Measures how spread out the singular values are for the
    activation matrix. Higher entropy = more expressivity.

    Score = -Σ p_i * log(p_i)  where p_i = σ_i / Σσ_i

    Args:
        forward_fn: Network forward function
        params: Network parameters
        x_batch: Input batch
        num_singular_values: Number of singular values to consider

    Returns:
        Expressivity score (higher = better)

    Reference:
        Lee & Ham "AZ-NAS: Assembling Zero-Cost Proxies" (CVPR 2024)
    """
    # Get activations
    outputs = forward_fn(params, x_batch)  # (batch, output_dim)

    if outputs.ndim == 1:
        outputs = outputs.reshape(-1, 1)

    # Compute SVD
    # For large matrices, we only need top singular values
    try:
        U, S, Vh = jnp.linalg.svd(outputs, full_matrices=False)
    except Exception:
        return 0.0

    # Take top k singular values
    k = min(num_singular_values, len(S))
    S = S[:k]

    # Normalize to probability distribution
    S = jnp.maximum(S, 1e-10)  # Avoid log(0)
    p = S / jnp.sum(S)

    # Compute entropy
    entropy = -jnp.sum(p * jnp.log(p))

    # Normalize by max possible entropy
    max_entropy = jnp.log(k) if k > 1 else 1.0
    normalized_entropy = entropy / max_entropy

    return float(normalized_entropy)


def compute_activation_diversity(
    forward_fn: Callable,
    params: Dict[str, jnp.ndarray],
    x_batch: jnp.ndarray,
) -> float:
    """
    Compute activation pattern diversity.

    Alternative to NASWOT that uses Hamming distance between
    activation patterns instead of Jacobian.

    Args:
        forward_fn: Network forward function
        params: Network parameters
        x_batch: Input batch

    Returns:
        Diversity score (0-1, higher = more diverse)
    """
    batch_size = x_batch.shape[0]

    if batch_size < 2:
        return 0.0

    # Get activations
    outputs = forward_fn(params, x_batch)

    if outputs.ndim == 1:
        outputs = outputs.reshape(-1, 1)

    # Binarize
    binary = (outputs > 0).astype(jnp.float32)

    # Compute pairwise Hamming distances
    # d(i,j) = mean(|binary_i - binary_j|)
    def pairwise_distance(i, j):
        return jnp.mean(jnp.abs(binary[i] - binary[j]))

    total_distance = 0.0
    count = 0
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            total_distance += pairwise_distance(i, j)
            count += 1

    if count == 0:
        return 0.0

    avg_distance = total_distance / count

    return float(avg_distance)


def compute_jacobian_correlation(
    forward_fn: Callable,
    params: Dict[str, jnp.ndarray],
    x_batch: jnp.ndarray,
    eps: float = 1e-4,
) -> float:
    """
    Compute correlation of input-output Jacobians.

    Lower correlation indicates the network treats different inputs
    differently, which is generally good for expressivity.

    Args:
        forward_fn: Network forward function
        params: Network parameters
        x_batch: Input batch
        eps: Finite difference epsilon

    Returns:
        1 - avg_correlation (higher = less correlated = better)
    """
    batch_size = x_batch.shape[0]

    if batch_size < 2:
        return 0.0

    # Compute Jacobians using finite differences
    def compute_jacobian_fd(x):
        """Compute Jacobian using finite differences."""
        x_flat = x.flatten()
        input_dim = len(x_flat)

        def forward_flat(x_flat):
            x = x_flat.reshape(x.shape)
            x = x.reshape((1,) + x.shape)
            return forward_fn(params, x).flatten()

        jacobian_cols = []
        f_x = forward_flat(x_flat)
        output_dim = len(f_x)

        for i in range(min(input_dim, 10)):  # Limit for efficiency
            x_plus = x_flat.at[i].set(x_flat[i] + eps)
            f_plus = forward_flat(x_plus)
            jacobian_cols.append((f_plus - f_x) / eps)

        if jacobian_cols:
            return jnp.stack(jacobian_cols, axis=1)
        return jnp.zeros((output_dim, 1))

    # Compute Jacobians for a subset of samples
    jacobians = [compute_jacobian_fd(x_batch[i])
                 for i in range(min(batch_size, 8))]

    # Compute pairwise correlations
    correlations = []
    for i in range(len(jacobians)):
        for j in range(i + 1, len(jacobians)):
            J_i = jacobians[i].flatten()
            J_j = jacobians[j].flatten()

            # Pearson correlation
            J_i_centered = J_i - jnp.mean(J_i)
            J_j_centered = J_j - jnp.mean(J_j)

            num = jnp.sum(J_i_centered * J_j_centered)
            den = jnp.sqrt(jnp.sum(J_i_centered**2) * jnp.sum(J_j_centered**2) + 1e-8)
            corr = jnp.abs(num / den)
            correlations.append(float(corr))

    if not correlations:
        return 0.0

    avg_corr = sum(correlations) / len(correlations)

    # Return 1 - correlation so higher is better
    return 1.0 - avg_corr
