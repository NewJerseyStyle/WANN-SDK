"""
Problem Interface for WANN SDK

Defines the base Problem class that users extend for custom tasks.
Inspired by PyTorch Lightning's LightningModule and HuggingFace's Trainer patterns.
"""

import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass


class Problem(ABC):
    """
    Base class for optimization problems.

    Extend this class to define custom tasks for evolution strategies.
    Similar to PyTorch Lightning's LightningModule pattern.

    Example:
        >>> class MyProblem(Problem):
        ...     def __init__(self, input_dim, output_dim):
        ...         super().__init__(input_dim, output_dim)
        ...         self.data = load_my_data()
        ...
        ...     def evaluate(self, network, key):
        ...         predictions = network(self.data['x'])
        ...         loss = jnp.mean((predictions - self.data['y'])**2)
        ...         return -float(loss)  # For ES (needs Python float)
        ...
        ...     def loss(self, network, key):
        ...         predictions = network(self.data['x'])
        ...         return jnp.mean((predictions - self.data['y'])**2)  # For gradients (JAX array)
    """

    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize problem.

        Args:
            input_dim: Input dimension for the network
            output_dim: Output dimension for the network
        """
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def evaluate(
        self,
        network: Callable[[jnp.ndarray], jnp.ndarray],
        key: jax.random.PRNGKey,
    ) -> float:
        """
        Evaluate a network on this problem (for ES).

        Args:
            network: A callable that takes input and returns output
            key: JAX random key for stochastic evaluation

        Returns:
            Fitness score as Python float (higher is better)

        Note:
            For loss-based problems, return negative loss.
            This method is used by ES optimizer.
        """
        pass

    def loss(
        self,
        network: Callable[[jnp.ndarray], jnp.ndarray],
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """
        Compute differentiable loss (for gradient-based optimizers).

        Args:
            network: A callable that takes input and returns output
            key: JAX random key for stochastic evaluation

        Returns:
            Loss as JAX array (lower is better) - DO NOT use float()

        Note:
            Override this for gradient-based training (SGD, Adam, AdamW).
            Default implementation calls evaluate() which won't work with gradients.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement loss(). "
            "For gradient-based optimizers (sgd/adam/adamw), implement the loss() method "
            "that returns a JAX array (not float). Or use optimizer='es'."
        )

    def setup(self) -> None:
        """
        Called before training starts.
        Override to load data or initialize resources.
        """
        pass

    def teardown(self) -> None:
        """
        Called after training ends.
        Override to cleanup resources.
        """
        pass


class SupervisedProblem(Problem):
    """
    Problem for supervised learning tasks.

    Handles common supervised learning patterns with
    data loading and batch evaluation.

    Example:
        >>> problem = SupervisedProblem(
        ...     x_train=images,
        ...     y_train=labels,
        ...     loss_fn='cross_entropy',
        ... )
    """

    def __init__(
        self,
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        x_val: Optional[jnp.ndarray] = None,
        y_val: Optional[jnp.ndarray] = None,
        loss_fn: str = 'mse',
        batch_size: Optional[int] = None,
        output_activation: str = 'none',
    ):
        """
        Initialize supervised problem.

        Args:
            x_train: Training inputs, shape (n_samples, input_dim)
            y_train: Training targets, shape (n_samples, output_dim) or (n_samples,)
            x_val: Validation inputs (optional)
            y_val: Validation targets (optional)
            loss_fn: Loss function - 'mse', 'cross_entropy', or 'binary_cross_entropy'
            batch_size: Batch size for evaluation (None = full batch)
            output_activation: Output activation - 'none', 'softmax', 'sigmoid'
        """
        # Infer dimensions
        input_dim = x_train.shape[1] if len(x_train.shape) > 1 else x_train.shape[0]

        if len(y_train.shape) == 1:
            # Classification with integer labels
            output_dim = int(jnp.max(y_train)) + 1
            self._is_classification = True
        else:
            output_dim = y_train.shape[1]
            self._is_classification = False

        super().__init__(input_dim, output_dim)

        self.x_train = jnp.asarray(x_train)
        self.y_train = jnp.asarray(y_train)
        self.x_val = jnp.asarray(x_val) if x_val is not None else None
        self.y_val = jnp.asarray(y_val) if y_val is not None else None
        self.loss_fn_name = loss_fn
        self.batch_size = batch_size
        self.output_activation = output_activation

        self._loss_fn = self._get_loss_fn(loss_fn)
        self._output_fn = self._get_output_fn(output_activation)

        self.n_train = len(x_train)
        self.n_val = len(x_val) if x_val is not None else 0

    def _get_loss_fn(self, name: str) -> Callable:
        """Get loss function by name."""
        if name == 'mse':
            return lambda pred, target: jnp.mean((pred - target) ** 2)
        elif name == 'cross_entropy':
            def cross_entropy(pred, target):
                # pred: (batch, classes), target: (batch,) integers
                pred = jax.nn.softmax(pred, axis=-1)
                pred = jnp.clip(pred, 1e-7, 1 - 1e-7)
                if len(target.shape) == 1:
                    # Integer labels
                    return -jnp.mean(jnp.log(pred[jnp.arange(len(target)), target]))
                else:
                    # One-hot labels
                    return -jnp.mean(jnp.sum(target * jnp.log(pred), axis=-1))
            return cross_entropy
        elif name == 'binary_cross_entropy':
            def bce(pred, target):
                pred = jax.nn.sigmoid(pred)
                pred = jnp.clip(pred, 1e-7, 1 - 1e-7)
                return -jnp.mean(target * jnp.log(pred) + (1 - target) * jnp.log(1 - pred))
            return bce
        else:
            raise ValueError(f"Unknown loss function: {name}")

    def _get_output_fn(self, name: str) -> Callable:
        """Get output activation function."""
        if name == 'none':
            return lambda x: x
        elif name == 'softmax':
            return lambda x: jax.nn.softmax(x, axis=-1)
        elif name == 'sigmoid':
            return jax.nn.sigmoid
        elif name == 'tanh':
            return jnp.tanh
        else:
            raise ValueError(f"Unknown activation: {name}")

    def evaluate(
        self,
        network: Callable[[jnp.ndarray], jnp.ndarray],
        key: jax.random.PRNGKey,
    ) -> float:
        """Evaluate network on training data."""
        if self.batch_size is not None and self.batch_size < self.n_train:
            # Sample batch
            indices = jax.random.choice(key, self.n_train, shape=(self.batch_size,), replace=False)
            x_batch = self.x_train[indices]
            y_batch = self.y_train[indices]
        else:
            x_batch = self.x_train
            y_batch = self.y_train

        # Forward pass
        predictions = network(x_batch)

        # Compute loss
        loss = self._loss_fn(predictions, y_batch)

        # Return negative loss as fitness (higher is better)
        return -float(loss)

    def loss(
        self,
        network: Callable[[jnp.ndarray], jnp.ndarray],
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """
        Compute differentiable loss for gradient-based training.

        Returns JAX array (not float) for gradient computation.
        """
        if self.batch_size is not None and self.batch_size < self.n_train:
            indices = jax.random.choice(key, self.n_train, shape=(self.batch_size,), replace=False)
            x_batch = self.x_train[indices]
            y_batch = self.y_train[indices]
        else:
            x_batch = self.x_train
            y_batch = self.y_train

        predictions = network(x_batch)
        return self._loss_fn(predictions, y_batch)

    def evaluate_accuracy(
        self,
        network: Callable[[jnp.ndarray], jnp.ndarray],
        use_val: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate classification accuracy.

        Args:
            network: Network function
            use_val: Use validation set if available

        Returns:
            Dictionary with accuracy metrics
        """
        if use_val and self.x_val is not None:
            x, y = self.x_val, self.y_val
        else:
            x, y = self.x_train, self.y_train

        predictions = network(x)

        if self._is_classification:
            pred_labels = jnp.argmax(predictions, axis=-1)
            accuracy = jnp.mean(pred_labels == y)
        else:
            pred_labels = jnp.argmax(predictions, axis=-1)
            true_labels = jnp.argmax(y, axis=-1)
            accuracy = jnp.mean(pred_labels == true_labels)

        # Compute loss
        loss = self._loss_fn(predictions, y)

        return {
            'accuracy': float(accuracy),
            'loss': float(loss),
            'n_samples': len(x),
        }


class RLProblem(Problem):
    """
    Problem for reinforcement learning tasks.

    Wraps Brax environments for ES-based policy optimization.

    Example:
        >>> from wann_sdk import BraxEnv
        >>> env = BraxEnv("ant")
        >>> problem = RLProblem(env, max_steps=1000)
    """

    def __init__(
        self,
        env: Any,  # BraxEnv
        max_steps: int = 1000,
        num_rollouts: int = 1,
    ):
        """
        Initialize RL problem.

        Args:
            env: BraxEnv instance
            max_steps: Maximum steps per episode
            num_rollouts: Number of rollouts to average
        """
        super().__init__(env.obs_dim, env.action_dim)
        self.env = env
        self.max_steps = max_steps
        self.num_rollouts = num_rollouts

    def evaluate(
        self,
        network: Callable[[jnp.ndarray], jnp.ndarray],
        key: jax.random.PRNGKey,
    ) -> float:
        """Evaluate network as policy on environment."""
        total_reward = 0.0

        for _ in range(self.num_rollouts):
            key, reset_key = jax.random.split(key)
            obs, state = self.env.reset(reset_key)
            episode_reward = 0.0

            for _ in range(self.max_steps):
                key, step_key = jax.random.split(key)
                # Ensure obs is 1D for network input
                obs_flat = obs.flatten() if obs.ndim > 1 else obs
                # Get action from network
                action = network(obs_flat)
                # Ensure action is 1D (action_dim,) for Brax - network may return (1, action_dim)
                action = action.flatten()
                obs, state, reward, done, _ = self.env.step(state, action, step_key)
                episode_reward += float(jnp.mean(reward))

            total_reward += episode_reward

        return total_reward / self.num_rollouts


class GymnaxProblem(Problem):
    """
    Problem for Gymnax environments (classic control, MinAtar).

    Wraps Gymnax environments for ES-based policy optimization.
    Handles both discrete and continuous action spaces.

    Example:
        >>> from wann_sdk import GymnaxEnv
        >>> env = GymnaxEnv("CartPole-v1")
        >>> problem = GymnaxProblem(env, max_steps=500)
    """

    def __init__(
        self,
        env: Any,  # GymnaxEnv
        max_steps: int = 500,
        num_rollouts: int = 1,
    ):
        """
        Initialize Gymnax problem.

        Args:
            env: GymnaxEnv instance
            max_steps: Maximum steps per episode
            num_rollouts: Number of rollouts to average
        """
        super().__init__(env.obs_dim, env.action_dim)
        self.env = env
        self.max_steps = max_steps
        self.num_rollouts = num_rollouts
        self.continuous = getattr(env, 'continuous', False)

    def evaluate(
        self,
        network: Callable[[jnp.ndarray], jnp.ndarray],
        key: jax.random.PRNGKey,
    ) -> float:
        """Evaluate network as policy on environment."""
        total_reward = 0.0

        for _ in range(self.num_rollouts):
            key, reset_key = jax.random.split(key)
            obs, state = self.env.reset(reset_key)
            episode_reward = 0.0
            done = False

            for step in range(self.max_steps):
                key, step_key = jax.random.split(key)
                # Ensure obs is 1D for network input
                obs_flat = obs.flatten() if obs.ndim > 1 else obs
                # Get action from network
                action = network(obs_flat)
                # Flatten action output
                action = action.flatten()

                # For discrete actions, network outputs logits - take argmax
                if not self.continuous:
                    action = jnp.argmax(action)

                obs, state, reward, done, _ = self.env.step(state, action, step_key)
                episode_reward += float(reward)

                # Early termination on done
                if done:
                    break

            total_reward += episode_reward

        return total_reward / self.num_rollouts
