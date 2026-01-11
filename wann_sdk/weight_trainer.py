"""
Weight Trainer for WANN SDK (Stage 2)

Trains individual weights on architectures found by ArchitectureSearch.
Supports both Evolution Strategies and gradient-based optimizers (SGD, AdamW).
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import pickle
from pathlib import Path
import time

from .problem import Problem
from .search import NetworkGenome


class Optimizer(Enum):
    """Available optimizers for weight training."""
    ES = "es"           # Evolution Strategies
    SGD = "sgd"         # Stochastic Gradient Descent
    ADAM = "adam"       # Adam optimizer
    ADAMW = "adamw"     # AdamW (Adam with weight decay)


@dataclass
class WeightTrainerConfig:
    """
    Configuration for Stage 2 weight training.

    Args:
        optimizer: Optimizer to use ('es', 'sgd', 'adam', 'adamw')
        learning_rate: Learning rate for optimization

        # ES-specific
        pop_size: Population size (ES only)
        noise_std: Noise standard deviation (ES only)

        # Gradient-based specific
        batch_size: Batch size for gradient computation
        weight_decay: Weight decay for AdamW
        beta1: Adam beta1 parameter
        beta2: Adam beta2 parameter

        # General
        seed: Random seed
        verbose: Print progress

    Example:
        >>> config = WeightTrainerConfig(
        ...     optimizer='adamw',
        ...     learning_rate=0.001,
        ...     weight_decay=0.01,
        ... )
    """
    optimizer: str = 'es'
    learning_rate: float = 0.01

    # ES parameters
    pop_size: int = 64
    noise_std: float = 0.1

    # Gradient-based parameters
    batch_size: Optional[int] = None
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    # General
    seed: int = 42
    verbose: bool = True


class TrainableNetwork:
    """
    A network with trainable weights derived from a NetworkGenome.

    Converts genome topology to a differentiable network with
    individual trainable weights.
    """

    def __init__(
        self,
        genome: NetworkGenome,
        activation_options: List[str],
        init_weight: float = 1.0,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        """
        Initialize trainable network from genome.

        Args:
            genome: NetworkGenome from architecture search
            activation_options: List of activation function names
            init_weight: Initial weight value (or 'random')
            key: Random key for random initialization
        """
        self.genome = genome
        self.activation_options = activation_options

        # Build activation map
        self._activations = {
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

        # Extract topology
        self.node_ids = genome.nodes[:, 0].astype(int)
        self.node_types = genome.nodes[:, 1].astype(int)
        self.node_activations = genome.nodes[:, 2].astype(int)

        self.input_ids = self.node_ids[self.node_types == 0]
        self.hidden_ids = self.node_ids[self.node_types == 1]
        self.output_ids = self.node_ids[self.node_types == 2]

        # Get enabled connections
        self.enabled_conns = genome.connections[genome.connections[:, 2] == 1]
        self.num_weights = len(self.enabled_conns)

        # Initialize weights
        if key is not None:
            self.weights = jax.random.normal(key, (self.num_weights,)) * 0.1 + init_weight
        else:
            self.weights = jnp.ones(self.num_weights) * init_weight

        # Build connection index map for efficient lookup
        self._build_connection_map()

    def _build_connection_map(self):
        """Build efficient connection lookup structures."""
        # Map: target_node -> list of (source_node, weight_index)
        self.incoming_connections: Dict[int, List[Tuple[int, int]]] = {}

        for i, conn in enumerate(self.enabled_conns):
            source = int(conn[0])
            target = int(conn[1])
            if target not in self.incoming_connections:
                self.incoming_connections[target] = []
            self.incoming_connections[target].append((source, i))

    def _get_activation(self, idx: int) -> Callable:
        """Get activation function by index."""
        if idx < len(self.activation_options):
            name = self.activation_options[idx]
        else:
            name = 'tanh'
        return self._activations.get(name, jnp.tanh)

    def forward(self, weights: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass with explicit weights (for gradient computation).

        Args:
            weights: Weight vector (num_connections,)
            x: Input data (batch, input_dim)

        Returns:
            Output (batch, output_dim)
        """
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        batch_size = x.shape[0]

        # Initialize node values
        node_values = {}

        # Set input values
        for i, nid in enumerate(self.input_ids):
            nid = int(nid)
            if i < x.shape[1]:
                node_values[nid] = x[:, i]
            else:
                node_values[nid] = jnp.zeros(batch_size)

        # Process hidden nodes
        for nid in self.hidden_ids:
            nid = int(nid)
            incoming = self.incoming_connections.get(nid, [])
            if not incoming:
                node_values[nid] = jnp.zeros(batch_size)
            else:
                total = jnp.zeros(batch_size)
                for source_id, weight_idx in incoming:
                    if source_id in node_values:
                        total = total + weights[weight_idx] * node_values[source_id]
                # Apply activation
                act_idx = int(self.node_activations[self.node_ids == nid][0])
                activation = self._get_activation(act_idx)
                node_values[nid] = activation(total)

        # Process output nodes
        outputs = []
        for nid in self.output_ids:
            nid = int(nid)
            incoming = self.incoming_connections.get(nid, [])
            if not incoming:
                outputs.append(jnp.zeros(batch_size))
            else:
                total = jnp.zeros(batch_size)
                for source_id, weight_idx in incoming:
                    if source_id in node_values:
                        total = total + weights[weight_idx] * node_values[source_id]
                outputs.append(total)

        return jnp.stack(outputs, axis=-1)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass using current weights."""
        return self.forward(self.weights, x)

    def get_params(self) -> jnp.ndarray:
        """Get current weight parameters."""
        return self.weights

    def set_params(self, weights: jnp.ndarray):
        """Set weight parameters."""
        self.weights = weights

    def num_params(self) -> int:
        """Get number of trainable parameters."""
        return self.num_weights


class WeightTrainer:
    """
    Stage 2: Train weights on architecture found by ArchitectureSearch.

    Supports multiple optimizers:
    - Evolution Strategies (ES)
    - SGD with momentum
    - Adam / AdamW

    Example:
        >>> # Stage 1: Architecture Search
        >>> search = ArchitectureSearch(problem, SearchConfig())
        >>> genome = search.run(generations=100)
        >>>
        >>> # Stage 2: Weight Training
        >>> trainer = WeightTrainer(
        ...     genome=genome,
        ...     problem=problem,
        ...     config=WeightTrainerConfig(optimizer='adamw'),
        ... )
        >>> trainer.fit(epochs=100)
        >>> network = trainer.get_network()
    """

    def __init__(
        self,
        genome: NetworkGenome,
        problem: Problem,
        config: Optional[WeightTrainerConfig] = None,
        activation_options: Optional[List[str]] = None,
    ):
        """
        Initialize weight trainer.

        Args:
            genome: NetworkGenome from ArchitectureSearch
            problem: Problem instance
            config: WeightTrainerConfig
            activation_options: Activation functions used in search
        """
        self.genome = genome
        self.problem = problem
        self.config = config or WeightTrainerConfig()

        # Default activation options
        if activation_options is None:
            activation_options = ['tanh', 'relu', 'sigmoid', 'sin', 'abs', 'square']

        # Initialize random key
        self.key = jax.random.PRNGKey(self.config.seed)

        # Create trainable network
        self.key, init_key = jax.random.split(self.key)
        self.network = TrainableNetwork(
            genome=genome,
            activation_options=activation_options,
            init_weight=1.0,
            key=init_key,
        )

        # Initialize optimizer state
        self._init_optimizer()

        # Training state
        self.best_fitness = -float('inf')
        self.best_weights: Optional[jnp.ndarray] = None
        self.history: List[Dict[str, float]] = []

    def _init_optimizer(self):
        """Initialize optimizer state."""
        opt = self.config.optimizer.lower()

        if opt == 'es':
            # ES doesn't need optimizer state
            self.opt_state = None
        elif opt == 'sgd':
            # Momentum state
            self.opt_state = {'velocity': jnp.zeros_like(self.network.weights)}
        elif opt in ['adam', 'adamw']:
            # Adam state
            self.opt_state = {
                'm': jnp.zeros_like(self.network.weights),
                'v': jnp.zeros_like(self.network.weights),
                't': 0,
            }
        else:
            raise ValueError(f"Unknown optimizer: {opt}")

    def _loss_fn(self, weights: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        """Compute differentiable loss for gradient-based training."""
        def network_fn(x):
            return self.network.forward(weights, x)
        # Use problem.loss() which returns JAX array (not float)
        return self.problem.loss(network_fn, key)

    def _es_step(self, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, Dict[str, float]]:
        """Evolution Strategies update step."""
        weights = self.network.weights
        fitness_list = []
        noise_list = []

        for _ in range(self.config.pop_size):
            key, eval_key, noise_key = jax.random.split(key, 3)

            # Generate noise
            noise = jax.random.normal(noise_key, weights.shape)

            # Positive perturbation
            pos_weights = weights + self.config.noise_std * noise
            pos_fn = lambda x: self.network.forward(pos_weights, x)
            pos_fitness = self.problem.evaluate(pos_fn, eval_key)

            # Negative perturbation
            key, eval_key = jax.random.split(key)
            neg_weights = weights - self.config.noise_std * noise
            neg_fn = lambda x: self.network.forward(neg_weights, x)
            neg_fitness = self.problem.evaluate(neg_fn, eval_key)

            fitness_list.append((pos_fitness, neg_fitness))
            noise_list.append(noise)

        # Compute gradient estimate
        fitness_array = jnp.array(fitness_list)
        fitness_diff = fitness_array[:, 0] - fitness_array[:, 1]

        std = jnp.std(fitness_diff)
        fitness_normalized = (fitness_diff - jnp.mean(fitness_diff)) / (std + 1e-8)

        # Compute gradient
        grad = jnp.zeros_like(weights)
        for i in range(self.config.pop_size):
            grad += fitness_normalized[i] * noise_list[i]
        grad /= self.config.pop_size * self.config.noise_std

        # Update weights
        new_weights = weights + self.config.learning_rate * grad

        # Metrics
        all_fitness = fitness_array.flatten()
        metrics = {
            'mean_fitness': float(jnp.mean(all_fitness)),
            'max_fitness': float(jnp.max(all_fitness)),
            'min_fitness': float(jnp.min(all_fitness)),
        }

        return new_weights, metrics

    def _gradient_step(self, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, Dict[str, float]]:
        """Gradient-based update step (SGD, Adam, AdamW)."""
        weights = self.network.weights
        opt = self.config.optimizer.lower()

        # Compute gradient
        loss, grads = jax.value_and_grad(self._loss_fn)(weights, key)

        if opt == 'sgd':
            # SGD with momentum
            momentum = 0.9
            self.opt_state['velocity'] = (
                momentum * self.opt_state['velocity'] - self.config.learning_rate * grads
            )
            new_weights = weights + self.opt_state['velocity']

        elif opt in ['adam', 'adamw']:
            # Adam / AdamW
            self.opt_state['t'] += 1
            t = self.opt_state['t']

            self.opt_state['m'] = (
                self.config.beta1 * self.opt_state['m'] + (1 - self.config.beta1) * grads
            )
            self.opt_state['v'] = (
                self.config.beta2 * self.opt_state['v'] + (1 - self.config.beta2) * grads ** 2
            )

            # Bias correction
            m_hat = self.opt_state['m'] / (1 - self.config.beta1 ** t)
            v_hat = self.opt_state['v'] / (1 - self.config.beta2 ** t)

            # Update
            new_weights = weights - self.config.learning_rate * m_hat / (jnp.sqrt(v_hat) + self.config.eps)

            # Weight decay (AdamW)
            if opt == 'adamw':
                new_weights = new_weights - self.config.learning_rate * self.config.weight_decay * weights

        else:
            raise ValueError(f"Unknown optimizer: {opt}")

        # Evaluate fitness with new weights
        eval_fn = lambda x: self.network.forward(new_weights, x)
        fitness = self.problem.evaluate(eval_fn, key)

        metrics = {
            'loss': float(loss),
            'fitness': float(fitness),
        }

        return new_weights, metrics

    def fit(
        self,
        epochs: int = 100,
        log_interval: int = 10,
    ) -> Dict[str, Any]:
        """
        Train weights on the architecture.

        Args:
            epochs: Number of training epochs/generations
            log_interval: How often to log progress

        Returns:
            Training results dictionary
        """
        self.problem.setup()
        opt = self.config.optimizer.lower()

        if self.config.verbose:
            print(f"Stage 2: Weight Training")
            print(f"Optimizer: {opt.upper()}")
            print(f"Parameters: {self.network.num_params()}")
            print(f"Learning rate: {self.config.learning_rate}")
            print("-" * 60)

        start_time = time.time()

        for epoch in range(epochs):
            self.key, step_key = jax.random.split(self.key)

            # Update step
            if opt == 'es':
                new_weights, metrics = self._es_step(step_key)
                fitness = metrics.get('max_fitness', metrics.get('mean_fitness'))
            else:
                new_weights, metrics = self._gradient_step(step_key)
                fitness = metrics.get('fitness', -metrics.get('loss', 0))

            self.network.set_params(new_weights)

            # Track best
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_weights = new_weights.copy()

            metrics['epoch'] = epoch
            metrics['best_fitness'] = self.best_fitness
            metrics['elapsed'] = time.time() - start_time
            self.history.append(metrics)

            # Log
            if self.config.verbose and (epoch % log_interval == 0 or epoch == epochs - 1):
                elapsed = metrics['elapsed']
                if opt == 'es':
                    print(
                        f"Epoch {epoch:4d} [{elapsed:6.1f}s] | "
                        f"Mean: {metrics['mean_fitness']:10.4f} | "
                        f"Max: {metrics['max_fitness']:10.4f} | "
                        f"Best: {self.best_fitness:10.4f}"
                    )
                else:
                    print(
                        f"Epoch {epoch:4d} [{elapsed:6.1f}s] | "
                        f"Loss: {metrics['loss']:10.4f} | "
                        f"Fitness: {metrics['fitness']:10.4f} | "
                        f"Best: {self.best_fitness:10.4f}"
                    )

        self.problem.teardown()

        # Restore best weights
        if self.best_weights is not None:
            self.network.set_params(self.best_weights)

        if self.config.verbose:
            print("-" * 60)
            print(f"Training completed in {time.time() - start_time:.1f}s")
            print(f"Best fitness: {self.best_fitness:.4f}")

        return {
            'best_fitness': self.best_fitness,
            'epochs': epochs,
            'history': self.history,
        }

    def get_network(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Get the trained network as a callable."""
        weights = self.best_weights if self.best_weights is not None else self.network.weights

        def network_fn(x: jnp.ndarray) -> jnp.ndarray:
            return self.network.forward(weights, x)

        return network_fn

    def get_weights(self) -> jnp.ndarray:
        """Get trained weight parameters."""
        return self.best_weights if self.best_weights is not None else self.network.weights

    def predict(self, x: jnp.ndarray) -> jnp.ndarray:
        """Make predictions."""
        weights = self.best_weights if self.best_weights is not None else self.network.weights
        return self.network.forward(weights, x)

    def save(self, path: str):
        """Save trained model."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'genome': {
                'nodes': self.genome.nodes,
                'connections': self.genome.connections,
                'num_inputs': self.genome.num_inputs,
                'num_outputs': self.genome.num_outputs,
            },
            'weights': self.best_weights if self.best_weights is not None else self.network.weights,
            'config': {
                'optimizer': self.config.optimizer,
                'learning_rate': self.config.learning_rate,
            },
            'best_fitness': self.best_fitness,
            'history': self.history,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        if self.config.verbose:
            print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, problem: Problem) -> 'WeightTrainer':
        """Load trained model."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        genome = NetworkGenome(
            nodes=data['genome']['nodes'],
            connections=data['genome']['connections'],
            num_inputs=data['genome']['num_inputs'],
            num_outputs=data['genome']['num_outputs'],
        )

        config = WeightTrainerConfig(**data['config'])
        trainer = cls(genome, problem, config)
        trainer.network.set_params(data['weights'])
        trainer.best_weights = data['weights']
        trainer.best_fitness = data['best_fitness']
        trainer.history = data['history']

        return trainer
