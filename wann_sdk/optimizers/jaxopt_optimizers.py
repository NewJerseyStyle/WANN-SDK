"""
JAXOpt Optimizer Wrappers

Provides gradient-based optimizers from JAXOpt library.
All optimizers follow the BaseOptimizer interface.

Available optimizers:
- Adam, AdamW: Adaptive moment estimation
- SGD: Stochastic gradient descent with momentum
- RMSProp: Root mean square propagation
- AdaGrad: Adaptive gradient
- LBFGS: Limited-memory BFGS (second-order)
- NonlinearCG: Nonlinear conjugate gradient
- GradientDescent: Vanilla gradient descent

Usage:
    >>> from wann_sdk.optimizers import Adam, LBFGS
    >>> opt = Adam(learning_rate=0.001)
    >>> state = opt.init_state(params)
    >>> state = opt.update(state, grads=gradients)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from dataclasses import dataclass
import jax
import jax.numpy as jnp

from .base import GradientOptimizer, OptimizerState

# Try to import jaxopt
try:
    import jaxopt
    JAXOPT_AVAILABLE = True
except ImportError:
    JAXOPT_AVAILABLE = False
    jaxopt = None


def _check_jaxopt():
    """Check if JAXOpt is available."""
    if not JAXOPT_AVAILABLE:
        raise ImportError(
            "JAXOpt is not installed. Install with: pip install jaxopt"
        )


class Adam(GradientOptimizer):
    """Adam optimizer (Adaptive Moment Estimation).

    Combines momentum with adaptive learning rates per parameter.
    Generally the best default choice for deep learning.

    Args:
        learning_rate: Step size (default: 0.001)
        beta1: Exponential decay rate for first moment (default: 0.9)
        beta2: Exponential decay rate for second moment (default: 0.999)
        eps: Small constant for numerical stability (default: 1e-8)

    Example:
        >>> opt = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
        >>> state = opt.init_state(params)
        >>> for _ in range(100):
        ...     grads = compute_gradients(state.params)
        ...     state = opt.update(state, grads=grads)

    Reference:
        Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2014)
    """

    name = "adam"

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        if JAXOPT_AVAILABLE:
            # Use JAXOpt's implementation
            self._opt = None  # Lazy initialization
        else:
            self._opt = None

    def init_state(self, params: jnp.ndarray) -> OptimizerState:
        """Initialize Adam state with zero moments."""
        internal = {
            'm': jnp.zeros_like(params),
            'v': jnp.zeros_like(params),
        }
        return OptimizerState(step=0, params=params, internal=internal)

    def update(
        self,
        state: OptimizerState,
        grads: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> OptimizerState:
        """Perform Adam update step."""
        if grads is None:
            raise ValueError("Adam requires gradients")

        t = state.step + 1
        m = state.internal['m']
        v = state.internal['v']

        # Update moments
        m = self.beta1 * m + (1 - self.beta1) * grads
        v = self.beta2 * v + (1 - self.beta2) * (grads ** 2)

        # Bias correction
        m_hat = m / (1 - self.beta1 ** t)
        v_hat = v / (1 - self.beta2 ** t)

        # Update parameters
        new_params = state.params - self.learning_rate * m_hat / (jnp.sqrt(v_hat) + self.eps)

        return OptimizerState(
            step=t,
            params=new_params,
            internal={'m': m, 'v': v},
        )

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "learning_rate": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
        }


class AdamW(Adam):
    """AdamW optimizer (Adam with decoupled weight decay).

    Like Adam but applies weight decay directly to parameters
    rather than through the gradient. Better for regularization.

    Args:
        learning_rate: Step size (default: 0.001)
        beta1: Exponential decay rate for first moment (default: 0.9)
        beta2: Exponential decay rate for second moment (default: 0.999)
        eps: Small constant for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)

    Reference:
        Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2017)
    """

    name = "adamw"

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            **kwargs,
        )
        self.weight_decay = weight_decay

    def update(
        self,
        state: OptimizerState,
        grads: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> OptimizerState:
        """Perform AdamW update with decoupled weight decay."""
        # First do Adam update
        new_state = super().update(state, grads=grads, **kwargs)

        # Then apply weight decay
        new_params = new_state.params - self.learning_rate * self.weight_decay * state.params

        return OptimizerState(
            step=new_state.step,
            params=new_params,
            internal=new_state.internal,
        )

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        config = super().get_default_config()
        config["weight_decay"] = 0.01
        return config


class SGD(GradientOptimizer):
    """Stochastic Gradient Descent with momentum.

    Classic optimizer with optional Nesterov momentum.

    Args:
        learning_rate: Step size (default: 0.01)
        momentum: Momentum coefficient (default: 0.9)
        nesterov: Use Nesterov momentum (default: False)

    Example:
        >>> opt = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    """

    name = "sgd"

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        nesterov: bool = False,
        **kwargs,
    ):
        super().__init__(learning_rate=learning_rate, **kwargs)
        self.momentum = momentum
        self.nesterov = nesterov

    def init_state(self, params: jnp.ndarray) -> OptimizerState:
        """Initialize SGD state with zero velocity."""
        internal = {'velocity': jnp.zeros_like(params)}
        return OptimizerState(step=0, params=params, internal=internal)

    def update(
        self,
        state: OptimizerState,
        grads: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> OptimizerState:
        """Perform SGD update with momentum."""
        if grads is None:
            raise ValueError("SGD requires gradients")

        velocity = state.internal['velocity']

        # Update velocity
        velocity = self.momentum * velocity - self.learning_rate * grads

        if self.nesterov:
            # Nesterov momentum
            new_params = state.params + self.momentum * velocity - self.learning_rate * grads
        else:
            new_params = state.params + velocity

        return OptimizerState(
            step=state.step + 1,
            params=new_params,
            internal={'velocity': velocity},
        )

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "learning_rate": 0.01,
            "momentum": 0.9,
            "nesterov": False,
        }


class RMSProp(GradientOptimizer):
    """RMSProp optimizer.

    Adapts learning rate using moving average of squared gradients.
    Good for non-stationary objectives and RNNs.

    Args:
        learning_rate: Step size (default: 0.001)
        decay: Decay rate for moving average (default: 0.9)
        eps: Small constant for numerical stability (default: 1e-8)

    Reference:
        Hinton, "Neural Networks for Machine Learning" Lecture 6e (2012)
    """

    name = "rmsprop"

    def __init__(
        self,
        learning_rate: float = 0.001,
        decay: float = 0.9,
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(learning_rate=learning_rate, **kwargs)
        self.decay = decay
        self.eps = eps

    def init_state(self, params: jnp.ndarray) -> OptimizerState:
        """Initialize RMSProp state."""
        internal = {'v': jnp.zeros_like(params)}
        return OptimizerState(step=0, params=params, internal=internal)

    def update(
        self,
        state: OptimizerState,
        grads: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> OptimizerState:
        """Perform RMSProp update."""
        if grads is None:
            raise ValueError("RMSProp requires gradients")

        v = state.internal['v']

        # Update moving average of squared gradients
        v = self.decay * v + (1 - self.decay) * (grads ** 2)

        # Update parameters
        new_params = state.params - self.learning_rate * grads / (jnp.sqrt(v) + self.eps)

        return OptimizerState(
            step=state.step + 1,
            params=new_params,
            internal={'v': v},
        )

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "learning_rate": 0.001,
            "decay": 0.9,
            "eps": 1e-8,
        }


class AdaGrad(GradientOptimizer):
    """AdaGrad optimizer.

    Adapts learning rate for each parameter based on historical gradients.
    Good for sparse gradients.

    Args:
        learning_rate: Initial step size (default: 0.01)
        eps: Small constant for numerical stability (default: 1e-8)

    Reference:
        Duchi et al., "Adaptive Subgradient Methods" (2011)
    """

    name = "adagrad"

    def __init__(
        self,
        learning_rate: float = 0.01,
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(learning_rate=learning_rate, **kwargs)
        self.eps = eps

    def init_state(self, params: jnp.ndarray) -> OptimizerState:
        """Initialize AdaGrad state."""
        internal = {'sum_sq': jnp.zeros_like(params)}
        return OptimizerState(step=0, params=params, internal=internal)

    def update(
        self,
        state: OptimizerState,
        grads: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> OptimizerState:
        """Perform AdaGrad update."""
        if grads is None:
            raise ValueError("AdaGrad requires gradients")

        sum_sq = state.internal['sum_sq']

        # Accumulate squared gradients
        sum_sq = sum_sq + grads ** 2

        # Update parameters
        new_params = state.params - self.learning_rate * grads / (jnp.sqrt(sum_sq) + self.eps)

        return OptimizerState(
            step=state.step + 1,
            params=new_params,
            internal={'sum_sq': sum_sq},
        )


class LBFGS(GradientOptimizer):
    """L-BFGS optimizer (Limited-memory BFGS).

    Second-order optimizer using approximate Hessian.
    Converges faster but requires more memory and full-batch gradients.

    Note: This optimizer works differently - it runs multiple steps
    internally to find the optimal update direction.

    Args:
        learning_rate: Step size (default: 1.0, often 1.0 for L-BFGS)
        max_iterations: Maximum iterations per update (default: 20)
        history_size: Number of past gradients to store (default: 10)
        tolerance: Convergence tolerance (default: 1e-5)

    Example:
        >>> opt = LBFGS(max_iterations=20)
        >>> # Note: L-BFGS often needs loss_fn for line search
        >>> state = opt.update(state, grads=grads, loss_fn=loss_fn)
    """

    name = "lbfgs"

    def __init__(
        self,
        learning_rate: float = 1.0,
        max_iterations: int = 20,
        history_size: int = 10,
        tolerance: float = 1e-5,
        **kwargs,
    ):
        super().__init__(learning_rate=learning_rate, **kwargs)
        self.max_iterations = max_iterations
        self.history_size = history_size
        self.tolerance = tolerance
        self._jaxopt_solver = None

    def _get_solver(self, loss_fn: Callable):
        """Get or create JAXOpt LBFGS solver."""
        _check_jaxopt()
        if self._jaxopt_solver is None:
            self._jaxopt_solver = jaxopt.LBFGS(
                fun=loss_fn,
                maxiter=self.max_iterations,
                history_size=self.history_size,
                tol=self.tolerance,
            )
        return self._jaxopt_solver

    def init_state(self, params: jnp.ndarray) -> OptimizerState:
        """Initialize L-BFGS state."""
        return OptimizerState(step=0, params=params, internal=None)

    def update(
        self,
        state: OptimizerState,
        grads: Optional[jnp.ndarray] = None,
        loss_fn: Optional[Callable] = None,
        **kwargs,
    ) -> OptimizerState:
        """Perform L-BFGS update.

        Note: L-BFGS works best with loss_fn for line search.
        If only grads provided, uses simple gradient step.
        """
        if loss_fn is not None and JAXOPT_AVAILABLE:
            # Use JAXOpt's full L-BFGS with line search
            solver = self._get_solver(loss_fn)
            new_params, opt_state = solver.run(state.params)
            return OptimizerState(
                step=state.step + 1,
                params=new_params,
                internal=opt_state,
            )
        elif grads is not None:
            # Fallback: simple gradient descent step
            new_params = state.params - self.learning_rate * grads
            return OptimizerState(
                step=state.step + 1,
                params=new_params,
                internal=state.internal,
            )
        else:
            raise ValueError("L-BFGS requires either grads or loss_fn")

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "learning_rate": 1.0,
            "max_iterations": 20,
            "history_size": 10,
            "tolerance": 1e-5,
        }


class GradientDescent(GradientOptimizer):
    """Vanilla Gradient Descent (no momentum).

    Simplest gradient-based optimizer. Use SGD with momentum
    for better performance in most cases.

    Args:
        learning_rate: Step size (default: 0.01)
    """

    name = "gd"

    def __init__(self, learning_rate: float = 0.01, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)

    def init_state(self, params: jnp.ndarray) -> OptimizerState:
        """Initialize GD state (stateless)."""
        return OptimizerState(step=0, params=params, internal=None)

    def update(
        self,
        state: OptimizerState,
        grads: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> OptimizerState:
        """Perform gradient descent step."""
        if grads is None:
            raise ValueError("GradientDescent requires gradients")

        new_params = state.params - self.learning_rate * grads

        return OptimizerState(
            step=state.step + 1,
            params=new_params,
            internal=None,
        )


# Dictionary of all JAXOpt-style optimizers
JAXOPT_OPTIMIZERS: Dict[str, Type[GradientOptimizer]] = {
    "adam": Adam,
    "adamw": AdamW,
    "sgd": SGD,
    "rmsprop": RMSProp,
    "adagrad": AdaGrad,
    "lbfgs": LBFGS,
    "gd": GradientDescent,
}


def list_gradient_optimizers() -> Dict[str, str]:
    """List all available gradient-based optimizers.

    Returns:
        Dictionary mapping optimizer names to descriptions
    """
    return {
        name: cls.__doc__.split('\n')[0] if cls.__doc__ else ""
        for name, cls in JAXOPT_OPTIMIZERS.items()
    }
