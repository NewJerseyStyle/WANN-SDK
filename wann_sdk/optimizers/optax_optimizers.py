"""
Optax Optimizer Wrappers

Provides gradient-based optimizers using Optax (DeepMind's JAX optimizer library).
Optax is the standard choice for gradient optimization in the JAX ecosystem.

Available optimizers:
- Adam, AdamW: Adaptive moment estimation
- SGD: Stochastic gradient descent with momentum
- RMSProp: Root mean square propagation
- AdaGrad: Adaptive gradient
- Lion: Evolved sign momentum optimizer
- Lamb: Layer-wise adaptive moments for batch training

Usage:
    >>> from wann_sdk.optimizers import Adam, SGD
    >>> opt = Adam(learning_rate=0.001)
    >>> state = opt.init_state(params)
    >>> state = opt.update(state, grads=gradients)
"""

from typing import Any, Callable, Dict, Optional, Tuple, Type
import jax
import jax.numpy as jnp
import optax

from .base import GradientOptimizer, OptimizerState


class OptaxWrapper(GradientOptimizer):
    """Base wrapper for Optax optimizers."""

    _optax_fn: Callable = None  # Override in subclasses

    def __init__(self, learning_rate: float = 0.001, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)
        self._optax_opt = None
        self._build_optax_optimizer()

    def _build_optax_optimizer(self):
        """Build the Optax optimizer. Override in subclasses."""
        raise NotImplementedError

    def init_state(self, params: jnp.ndarray) -> OptimizerState:
        """Initialize optimizer state using Optax."""
        optax_state = self._optax_opt.init(params)
        return OptimizerState(
            step=0,
            params=params,
            internal={'optax_state': optax_state},
        )

    def update(
        self,
        state: OptimizerState,
        grads: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> OptimizerState:
        """Perform update using Optax."""
        if grads is None:
            raise ValueError(f"{self.__class__.__name__} requires gradients")

        optax_state = state.internal['optax_state']

        # Get updates from Optax
        updates, new_optax_state = self._optax_opt.update(grads, optax_state, state.params)

        # Apply updates
        new_params = optax.apply_updates(state.params, updates)

        return OptimizerState(
            step=state.step + 1,
            params=new_params,
            internal={'optax_state': new_optax_state},
        )


class Adam(OptaxWrapper):
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
        >>> state = opt.update(state, grads=gradients)

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
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        super().__init__(learning_rate=learning_rate, **kwargs)

    def _build_optax_optimizer(self):
        self._optax_opt = optax.adam(
            learning_rate=self.learning_rate,
            b1=self.beta1,
            b2=self.beta2,
            eps=self.eps,
        )

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "learning_rate": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
        }


class AdamW(OptaxWrapper):
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
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        super().__init__(learning_rate=learning_rate, **kwargs)

    def _build_optax_optimizer(self):
        self._optax_opt = optax.adamw(
            learning_rate=self.learning_rate,
            b1=self.beta1,
            b2=self.beta2,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "learning_rate": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
            "weight_decay": 0.01,
        }


class SGD(OptaxWrapper):
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
        self.momentum = momentum
        self.nesterov = nesterov
        super().__init__(learning_rate=learning_rate, **kwargs)

    def _build_optax_optimizer(self):
        self._optax_opt = optax.sgd(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            nesterov=self.nesterov,
        )

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "learning_rate": 0.01,
            "momentum": 0.9,
            "nesterov": False,
        }


class RMSProp(OptaxWrapper):
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
        self.decay = decay
        self.eps = eps
        super().__init__(learning_rate=learning_rate, **kwargs)

    def _build_optax_optimizer(self):
        self._optax_opt = optax.rmsprop(
            learning_rate=self.learning_rate,
            decay=self.decay,
            eps=self.eps,
        )

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "learning_rate": 0.001,
            "decay": 0.9,
            "eps": 1e-8,
        }


class AdaGrad(OptaxWrapper):
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
        self.eps = eps
        super().__init__(learning_rate=learning_rate, **kwargs)

    def _build_optax_optimizer(self):
        self._optax_opt = optax.adagrad(
            learning_rate=self.learning_rate,
            eps=self.eps,
        )


class Lion(OptaxWrapper):
    """Lion optimizer (Evolved Sign Momentum).

    Discovered through program search, uses sign of momentum.
    Often more memory efficient than Adam.

    Args:
        learning_rate: Step size (default: 0.0001)
        beta1: Exponential decay for momentum (default: 0.9)
        beta2: Exponential decay for velocity (default: 0.99)
        weight_decay: Weight decay coefficient (default: 0.0)

    Reference:
        Chen et al., "Symbolic Discovery of Optimization Algorithms" (2023)
    """

    name = "lion"

    def __init__(
        self,
        learning_rate: float = 0.0001,
        beta1: float = 0.9,
        beta2: float = 0.99,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        super().__init__(learning_rate=learning_rate, **kwargs)

    def _build_optax_optimizer(self):
        self._optax_opt = optax.lion(
            learning_rate=self.learning_rate,
            b1=self.beta1,
            b2=self.beta2,
            weight_decay=self.weight_decay,
        )


class Lamb(OptaxWrapper):
    """LAMB optimizer (Layer-wise Adaptive Moments).

    Adam-like optimizer with layer-wise learning rate adaptation.
    Good for large batch training.

    Args:
        learning_rate: Base step size (default: 0.001)
        beta1: Exponential decay for first moment (default: 0.9)
        beta2: Exponential decay for second moment (default: 0.999)
        eps: Numerical stability constant (default: 1e-6)
        weight_decay: Weight decay coefficient (default: 0.0)

    Reference:
        You et al., "Large Batch Optimization for Deep Learning" (2019)
    """

    name = "lamb"

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        super().__init__(learning_rate=learning_rate, **kwargs)

    def _build_optax_optimizer(self):
        self._optax_opt = optax.lamb(
            learning_rate=self.learning_rate,
            b1=self.beta1,
            b2=self.beta2,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )


class Noisy(OptaxWrapper):
    """Noisy SGD optimizer.

    Adds gradient noise for escaping local minima.

    Args:
        learning_rate: Step size (default: 0.01)
        eta: Noise scale (default: 0.1)
        gamma: Noise decay rate (default: 0.55)

    Reference:
        Neelakantan et al., "Adding Gradient Noise Improves Learning" (2015)
    """

    name = "noisy_sgd"

    def __init__(
        self,
        learning_rate: float = 0.01,
        eta: float = 0.1,
        gamma: float = 0.55,
        seed: int = 42,
        **kwargs,
    ):
        self.eta = eta
        self.gamma = gamma
        self.seed = seed
        super().__init__(learning_rate=learning_rate, **kwargs)

    def _build_optax_optimizer(self):
        self._optax_opt = optax.chain(
            optax.add_noise(self.eta, self.gamma, self.seed),
            optax.sgd(self.learning_rate),
        )


class LBFGS(GradientOptimizer):
    """L-BFGS optimizer (Limited-memory BFGS).

    Second-order optimizer using approximate Hessian.
    Requires JAXOpt for full implementation.

    Args:
        learning_rate: Step size for fallback (default: 1.0)
        max_iterations: Maximum iterations per update (default: 20)
        history_size: Number of past gradients to store (default: 10)
        tolerance: Convergence tolerance (default: 1e-5)

    Note:
        For full L-BFGS with line search, install jaxopt:
        pip install jaxopt
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

        # Try to use JAXOpt if available
        try:
            import jaxopt
            self._has_jaxopt = True
        except ImportError:
            self._has_jaxopt = False

    def init_state(self, params: jnp.ndarray) -> OptimizerState:
        return OptimizerState(step=0, params=params, internal=None)

    def update(
        self,
        state: OptimizerState,
        grads: Optional[jnp.ndarray] = None,
        loss_fn: Optional[Callable] = None,
        **kwargs,
    ) -> OptimizerState:
        """Perform L-BFGS update."""
        if loss_fn is not None and self._has_jaxopt:
            import jaxopt
            solver = jaxopt.LBFGS(
                fun=loss_fn,
                maxiter=self.max_iterations,
                history_size=self.history_size,
                tol=self.tolerance,
            )
            new_params, _ = solver.run(state.params)
            return OptimizerState(step=state.step + 1, params=new_params, internal=None)
        elif grads is not None:
            # Fallback to simple gradient descent
            new_params = state.params - self.learning_rate * grads
            return OptimizerState(step=state.step + 1, params=new_params, internal=None)
        else:
            raise ValueError("L-BFGS requires either grads or loss_fn")


# Dictionary of all Optax-based optimizers
OPTAX_OPTIMIZERS: Dict[str, Type[GradientOptimizer]] = {
    "adam": Adam,
    "adamw": AdamW,
    "sgd": SGD,
    "rmsprop": RMSProp,
    "adagrad": AdaGrad,
    "lion": Lion,
    "lamb": Lamb,
    "noisy_sgd": Noisy,
    "lbfgs": LBFGS,
}


def list_gradient_optimizers() -> Dict[str, str]:
    """List all available gradient-based optimizers."""
    return {
        name: cls.__doc__.split('\n')[0] if cls.__doc__ else ""
        for name, cls in OPTAX_OPTIMIZERS.items()
    }
