"""
Activation Function Approximation for Gradient-Based Training

Some WANN activation functions (step, abs) are non-differentiable or have
problematic gradients. This module provides differentiable approximations
for use during Stage 2 weight training.

Approach:
1. Identify non-differentiable activations in the network
2. Replace with learned approximations (MLP or KAN-style)
3. Train weights with smooth gradient flow
4. Swap back to original activations for deployment

Supported approximation methods:
- MLP: Simple 2-layer MLP with smooth activations
- KAN: Kolmogorov-Arnold Network using B-spline basis functions
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import pickle
import hashlib

# Non-differentiable or problematic activations
NON_DIFFERENTIABLE = {
    'step',      # Heaviside step - zero gradient everywhere except undefined at 0
    'abs',       # Absolute value - undefined gradient at 0
}

# Smooth alternatives for quick replacement (no learning needed)
SMOOTH_ALTERNATIVES = {
    'step': lambda x: jax.nn.sigmoid(10.0 * x),  # Steep sigmoid approximation
    'abs': lambda x: jnp.sqrt(x**2 + 1e-6),      # Smooth abs
}


@dataclass
class ApproximatorConfig:
    """Configuration for activation approximation.

    Args:
        method: Approximation method ('mlp', 'kan', 'smooth')
        hidden_size: Hidden layer size for MLP/KAN
        num_basis: Number of basis functions for KAN
        grid_range: Input range for fitting (-grid_range, grid_range)
        num_samples: Training samples for fitting
        learning_rate: Learning rate for fitting
        fit_steps: Number of optimization steps
        cache_dir: Directory for caching approximators
    """
    method: str = 'kan'
    hidden_size: int = 16
    num_basis: int = 8
    grid_range: float = 5.0
    num_samples: int = 1000
    learning_rate: float = 0.01
    fit_steps: int = 500
    cache_dir: Optional[str] = None


# ============================================================
# B-Spline Basis for KAN
# ============================================================

def bspline_basis(x: jnp.ndarray, grid: jnp.ndarray, degree: int = 3) -> jnp.ndarray:
    """
    Compute B-spline basis functions.

    Args:
        x: Input values (batch,)
        grid: Knot positions (num_knots,)
        degree: B-spline degree (default 3 = cubic)

    Returns:
        Basis values (batch, num_basis)
    """
    num_knots = len(grid)
    num_basis = num_knots - degree - 1

    # Cox-de Boor recursion (simplified for fixed degree 3)
    def basis_0(i):
        """Degree 0 basis."""
        return jnp.where(
            (x >= grid[i]) & (x < grid[i + 1]),
            1.0,
            0.0
        )

    def basis_p(i, p, cache):
        """Degree p basis using recursion."""
        if p == 0:
            return basis_0(i)

        left_num = x - grid[i]
        left_den = grid[i + p] - grid[i]
        left = jnp.where(left_den > 0, left_num / left_den * cache[i], 0.0)

        right_num = grid[i + p + 1] - x
        right_den = grid[i + p + 1] - grid[i + 1]
        right = jnp.where(right_den > 0, right_num / right_den * cache[i + 1], 0.0)

        return left + right

    # Build basis iteratively
    cache = [basis_0(i) for i in range(num_knots - 1)]

    for p in range(1, degree + 1):
        new_cache = []
        for i in range(num_knots - p - 1):
            new_cache.append(basis_p(i, p, cache))
        cache = new_cache

    return jnp.stack(cache[:num_basis], axis=-1)


def create_knot_grid(num_basis: int, grid_range: float, degree: int = 3) -> jnp.ndarray:
    """Create uniform knot grid for B-splines."""
    num_knots = num_basis + degree + 1
    # Extend grid beyond range for boundary conditions
    extended_range = grid_range * 1.2
    return jnp.linspace(-extended_range, extended_range, num_knots)


# ============================================================
# KAN-style Approximator
# ============================================================

@dataclass
class KANApproximator:
    """
    Kolmogorov-Arnold Network style approximator.

    Uses B-spline basis functions to approximate activation.
    Provides smooth, differentiable approximation.
    """
    coefficients: jnp.ndarray  # (num_basis,)
    grid: jnp.ndarray          # Knot positions
    degree: int = 3

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate approximation."""
        # Handle batched input
        original_shape = x.shape
        x_flat = x.flatten()

        # Compute basis
        basis = bspline_basis(x_flat, self.grid, self.degree)

        # Linear combination
        result = jnp.dot(basis, self.coefficients)

        return result.reshape(original_shape)


def fit_kan_approximator(
    target_fn: Callable,
    config: ApproximatorConfig,
    key: jax.random.PRNGKey,
) -> KANApproximator:
    """
    Fit KAN approximator to target function.

    Args:
        target_fn: Target activation function
        config: Approximator configuration
        key: Random key

    Returns:
        Fitted KANApproximator
    """
    # Create grid
    grid = create_knot_grid(config.num_basis, config.grid_range)
    num_basis = config.num_basis

    # Generate training data
    x_train = jnp.linspace(-config.grid_range, config.grid_range, config.num_samples)
    y_train = target_fn(x_train)

    # Compute basis matrix
    B = bspline_basis(x_train, grid)

    # Least squares fit: B @ c = y
    # c = (B^T B)^-1 B^T y
    BtB = B.T @ B + 1e-6 * jnp.eye(num_basis)  # Regularization
    Bty = B.T @ y_train
    coefficients = jnp.linalg.solve(BtB, Bty)

    return KANApproximator(
        coefficients=coefficients,
        grid=grid,
        degree=3,
    )


# ============================================================
# MLP Approximator
# ============================================================

@dataclass
class MLPApproximator:
    """
    MLP-based activation approximator.

    Simple 2-layer MLP with tanh activations for smooth gradients.
    """
    w1: jnp.ndarray  # (1, hidden)
    b1: jnp.ndarray  # (hidden,)
    w2: jnp.ndarray  # (hidden, 1)
    b2: jnp.ndarray  # (1,)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate MLP approximation."""
        original_shape = x.shape
        x_flat = x.flatten().reshape(-1, 1)

        # Forward pass
        h = jnp.tanh(x_flat @ self.w1 + self.b1)
        y = h @ self.w2 + self.b2

        return y.flatten().reshape(original_shape)


def fit_mlp_approximator(
    target_fn: Callable,
    config: ApproximatorConfig,
    key: jax.random.PRNGKey,
) -> MLPApproximator:
    """
    Fit MLP approximator to target function.

    Args:
        target_fn: Target activation function
        config: Approximator configuration
        key: Random key

    Returns:
        Fitted MLPApproximator
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)
    hidden = config.hidden_size

    # Initialize weights
    w1 = jax.random.normal(k1, (1, hidden)) * 0.5
    b1 = jnp.zeros(hidden)
    w2 = jax.random.normal(k2, (hidden, 1)) * 0.5
    b2 = jnp.zeros(1)

    # Training data
    x_train = jnp.linspace(-config.grid_range, config.grid_range, config.num_samples)
    y_train = target_fn(x_train)

    def forward(params, x):
        w1, b1, w2, b2 = params
        x = x.reshape(-1, 1)
        h = jnp.tanh(x @ w1 + b1)
        return (h @ w2 + b2).flatten()

    def loss_fn(params):
        pred = forward(params, x_train)
        return jnp.mean((pred - y_train) ** 2)

    # Optimize
    params = (w1, b1, w2, b2)
    lr = config.learning_rate

    @jit
    def update(params):
        grads = grad(loss_fn)(params)
        return tuple(p - lr * g for p, g in zip(params, grads))

    for _ in range(config.fit_steps):
        params = update(params)

    w1, b1, w2, b2 = params
    return MLPApproximator(w1=w1, b1=b1, w2=w2, b2=b2)


# ============================================================
# Approximation Cache
# ============================================================

class ApproximationCache:
    """
    Cache for storing fitted approximators.

    Saves approximators to disk for reuse across training runs.
    Uses configuration hash for cache key.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize cache.

        Args:
            cache_dir: Directory for cache files (None = memory only)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._memory_cache: Dict[str, Any] = {}

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, activation: str, config: ApproximatorConfig) -> str:
        """Create cache key from activation and config."""
        config_str = f"{activation}_{config.method}_{config.hidden_size}_{config.num_basis}_{config.grid_range}"
        return hashlib.md5(config_str.encode()).hexdigest()[:16]

    def get(self, activation: str, config: ApproximatorConfig) -> Optional[Any]:
        """Get cached approximator."""
        key = self._make_key(activation, config)

        # Check memory cache
        if key in self._memory_cache:
            return self._memory_cache[key]

        # Check disk cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    approx = pickle.load(f)
                    self._memory_cache[key] = approx
                    return approx

        return None

    def put(self, activation: str, config: ApproximatorConfig, approximator: Any):
        """Store approximator in cache."""
        key = self._make_key(activation, config)

        # Memory cache
        self._memory_cache[key] = approximator

        # Disk cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(approximator, f)

    def clear(self):
        """Clear all caches."""
        self._memory_cache.clear()
        if self.cache_dir:
            for f in self.cache_dir.glob("*.pkl"):
                f.unlink()


# Global cache instance
_global_cache = ApproximationCache()


def set_cache_dir(cache_dir: str):
    """Set global cache directory."""
    global _global_cache
    _global_cache = ApproximationCache(cache_dir)


def get_cache() -> ApproximationCache:
    """Get global cache instance."""
    return _global_cache


# ============================================================
# Main API
# ============================================================

def is_non_differentiable(activation: str) -> bool:
    """Check if activation is non-differentiable."""
    return activation in NON_DIFFERENTIABLE


def get_differentiable_activation(
    activation: str,
    config: Optional[ApproximatorConfig] = None,
    key: Optional[jax.random.PRNGKey] = None,
) -> Callable:
    """
    Get differentiable version of activation function.

    For differentiable activations, returns the original.
    For non-differentiable ones, returns a smooth approximation.

    Args:
        activation: Activation function name
        config: Approximator configuration (None = use defaults)
        key: Random key for fitting (None = use default)

    Returns:
        Differentiable activation function

    Example:
        >>> act_fn = get_differentiable_activation('step')
        >>> grad_fn = jax.grad(lambda x: act_fn(x).sum())
        >>> grad_fn(jnp.array([0.0, 1.0, -1.0]))  # Works!
    """
    # Standard activations - already differentiable
    standard_activations = {
        'tanh': jnp.tanh,
        'relu': jax.nn.relu,
        'sigmoid': jax.nn.sigmoid,
        'sin': jnp.sin,
        'cos': jnp.cos,
        'square': lambda x: x ** 2,
        'identity': lambda x: x,
        'gaussian': lambda x: jnp.exp(-x ** 2),
    }

    if activation in standard_activations:
        return standard_activations[activation]

    if activation not in NON_DIFFERENTIABLE:
        # Unknown activation, return identity
        return lambda x: x

    # Non-differentiable - need approximation
    config = config or ApproximatorConfig()
    key = key if key is not None else jax.random.PRNGKey(42)

    # Quick smooth alternative
    if config.method == 'smooth':
        return SMOOTH_ALTERNATIVES.get(activation, lambda x: x)

    # Check cache
    cache = get_cache()
    cached = cache.get(activation, config)
    if cached is not None:
        return cached

    # Get target function
    target_fns = {
        'step': lambda x: jnp.where(x > 0, 1.0, 0.0),
        'abs': jnp.abs,
    }
    target_fn = target_fns.get(activation, lambda x: x)

    # Fit approximator
    if config.method == 'kan':
        approximator = fit_kan_approximator(target_fn, config, key)
    else:  # mlp
        approximator = fit_mlp_approximator(target_fn, config, key)

    # Cache and return
    cache.put(activation, config, approximator)

    return approximator


def create_activation_map_differentiable(
    activation_options: List[str],
    config: Optional[ApproximatorConfig] = None,
) -> Dict[str, Callable]:
    """
    Create activation map with differentiable versions.

    Args:
        activation_options: List of activation names
        config: Approximator configuration

    Returns:
        Dictionary mapping names to differentiable functions

    Example:
        >>> activations = create_activation_map_differentiable(
        ...     ['tanh', 'relu', 'step', 'abs']
        ... )
        >>> # All activations now support gradients
    """
    key = jax.random.PRNGKey(42)
    result = {}

    for i, name in enumerate(activation_options):
        k = jax.random.fold_in(key, i)
        result[name] = get_differentiable_activation(name, config, k)

    return result


def get_original_activation(activation: str) -> Callable:
    """
    Get original (possibly non-differentiable) activation.

    Use this for deployment after training.

    Args:
        activation: Activation name

    Returns:
        Original activation function
    """
    all_activations = {
        'tanh': jnp.tanh,
        'relu': jax.nn.relu,
        'sigmoid': jax.nn.sigmoid,
        'sin': jnp.sin,
        'cos': jnp.cos,
        'abs': jnp.abs,
        'square': lambda x: x ** 2,
        'identity': lambda x: x,
        'step': lambda x: jnp.where(x > 0, 1.0, 0.0),
        'gaussian': lambda x: jnp.exp(-x ** 2),
    }
    return all_activations.get(activation, lambda x: x)
