"""
WANN SDK - Core Architecture and Interfaces

A modular SDK for Weight Agnostic Neural Networks with support for:
- Architecture search (NAS)
- Multiple RL training methods (DQN, PPO, SAC, etc.)
- Distributed environments via Ray
- Gymnasium environment support
- Flexible policy interfaces

Design Philosophy:
1. Separation of concerns: Architecture search vs. weight training
2. Training method agnostic: Compatible with any RL algorithm
3. Environment agnostic: Works with any Gymnasium-compatible env
4. Extensible: Easy to add new training methods or architectures
"""

import jax
import jax.numpy as jnp
from jax import vmap, jit
from typing import Dict, Any, Tuple, Optional, Callable, Protocol, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pickle
from pathlib import Path

# Import WANN core components
from .wann_tensorneat import WANNGenome


# ============================================================================
# Core Interfaces and Protocols
# ============================================================================

class NetworkArchitecture(Protocol):
    """
    Protocol for network architectures discovered by WANN.
    Any architecture must implement this interface to be compatible
    with the SDK's training methods.
    """
    
    def forward(self, x: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Forward pass through network.
        
        Args:
            x: Input observation
            params: Network parameters (weights, biases, etc.)
        
        Returns:
            Network output
        """
        ...
    
    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        ...
    
    def init_params(self, key: jax.random.PRNGKey) -> Dict[str, jnp.ndarray]:
        """Initialize parameters."""
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize architecture to dictionary."""
        ...
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NetworkArchitecture":
        """Deserialize architecture from dictionary."""
        ...


class PolicyInterface(ABC):
    """
    Abstract interface for RL policies.
    All training methods must implement this to be compatible with WANN.
    """
    
    @abstractmethod
    def select_action(
        self, 
        observation: jnp.ndarray, 
        deterministic: bool = False
    ) -> Union[jnp.ndarray, int]:
        """
        Select action given observation.
        
        Args:
            observation: Environment observation
            deterministic: If True, return deterministic action
        
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def compute_loss(
        self,
        batch: Dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """
        Compute loss for updating policy.
        
        Args:
            batch: Batch of experience data
                - observations: (batch_size, obs_dim)
                - actions: (batch_size, act_dim)
                - rewards: (batch_size,)
                - next_observations: (batch_size, obs_dim)
                - dones: (batch_size,)
                (+ method-specific fields)
        
        Returns:
            Loss value
        """
        pass
    
    @abstractmethod
    def update(
        self,
        batch: Dict[str, jnp.ndarray]
    ) -> Dict[str, float]:
        """
        Update policy parameters.
        
        Args:
            batch: Batch of experience data
        
        Returns:
            Dictionary of metrics (loss, etc.)
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, jnp.ndarray]:
        """Get current policy parameters."""
        pass
    
    @abstractmethod
    def set_params(self, params: Dict[str, jnp.ndarray]):
        """Set policy parameters."""
        pass


@dataclass
class ArchitectureSpec:
    """
    Specification of a discovered architecture.
    This is the output of the architecture search phase.
    """
    
    # Network topology
    nodes: jnp.ndarray
    connections: jnp.ndarray
    
    # Metadata
    num_inputs: int
    num_outputs: int
    num_hidden: int
    num_params: int
    
    # Performance metrics from search
    search_fitness: float
    search_complexity: float
    
    # Additional info
    activation_functions: Dict[int, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def save(self, path: str):
        """Save architecture specification."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> "ArchitectureSpec":
        """Load architecture specification."""
        with open(path, 'rb') as f:
            return pickle.load(f)


@dataclass
class TrainingConfig:
    """
    Configuration for training phase.
    Method-agnostic settings.
    """
    
    # Training hyperparameters
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 100000
    num_epochs: int = 1000
    
    # Evaluation
    eval_frequency: int = 10
    eval_episodes: int = 10
    
    # Checkpointing
    checkpoint_frequency: int = 50
    checkpoint_dir: str = "./checkpoints"
    
    # Logging
    log_frequency: int = 1
    
    # Method-specific settings
    method_kwargs: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# WANN Architecture Wrapper
# ============================================================================

class WANNArchitecture:
    """
    Wrapper that converts WANN genome to trainable architecture.
    Implements NetworkArchitecture protocol.
    """
    
    def __init__(
        self,
        spec: ArchitectureSpec,
        genome: Optional[WANNGenome] = None
    ):
        """
        Initialize WANN architecture.
        
        Args:
            spec: Architecture specification from search
            genome: WANN genome instance (optional)
        """
        self.spec = spec
        self.genome = genome or WANNGenome(
            num_inputs=spec.num_inputs,
            num_outputs=spec.num_outputs
        )
        
        # Compile forward pass
        self._forward_fn = jit(self._forward_impl)
        self._forward_batch_fn = jit(vmap(self._forward_impl, in_axes=(None, 0)))
    
    def _forward_impl(
        self, 
        params: Dict[str, jnp.ndarray],
        x: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Internal forward pass implementation.
        
        Args:
            params: Network parameters
            x: Input observation
        
        Returns:
            Network output
        """
        # Update connections with learned weights
        weights = params['connection_weights']
        conns = self.spec.connections.at[:, 4].set(weights)
        
        # Forward through network
        output = self.genome.forward(None, x, self.spec.nodes, conns)
        
        return output
    
    def forward(
        self, 
        x: jnp.ndarray, 
        params: Dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """
        Forward pass through network.
        
        Args:
            x: Input observation (obs_dim,) or (batch_size, obs_dim)
            params: Network parameters
        
        Returns:
            Network output
        """
        if x.ndim == 1:
            return self._forward_fn(params, x)
        else:
            return self._forward_batch_fn(params, x)
    
    def get_num_params(self) -> int:
        """Get number of trainable parameters."""
        return self.spec.num_params
    
    def init_params(
        self, 
        key: jax.random.PRNGKey,
        init_scale: float = 0.1
    ) -> Dict[str, jnp.ndarray]:
        """
        Initialize network parameters.
        
        Args:
            key: Random key
            init_scale: Scale of initialization
        
        Returns:
            Initialized parameters
        """
        num_conns = len(self.spec.connections)
        
        # Initialize with small random values
        # Can also use best shared weight from search as starting point
        weights = jax.random.normal(key, (num_conns,)) * init_scale
        
        # Zero out disabled connections
        enabled_mask = self.spec.connections[:, 3].astype(bool)
        weights = jnp.where(enabled_mask, weights, 0.0)
        
        return {
            'connection_weights': weights
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize architecture."""
        return {
            'spec': self.spec,
            'type': 'WANN'
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WANNArchitecture":
        """Deserialize architecture."""
        return cls(spec=data['spec'])
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """Get architecture information."""
        return {
            'num_inputs': self.spec.num_inputs,
            'num_outputs': self.spec.num_outputs,
            'num_hidden': self.spec.num_hidden,
            'num_connections': int(jnp.sum(self.spec.connections[:, 3])),
            'num_params': self.spec.num_params,
            'search_fitness': self.spec.search_fitness,
        }


# ============================================================================
# Replay Buffer (Shared by All Methods)
# ============================================================================

class ReplayBuffer:
    """
    Replay buffer for experience replay.
    Compatible with any RL training method.
    """
    
    def __init__(
        self,
        capacity: int,
        observation_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        extra_fields: Optional[Dict[str, Tuple]] = None
    ):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size
            observation_shape: Shape of observations
            action_shape: Shape of actions
            extra_fields: Additional fields with their shapes
                e.g., {'log_probs': (), 'values': ()}
        """
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.extra_fields = extra_fields or {}
        
        # Initialize storage
        self.observations = jnp.zeros((capacity, *observation_shape))
        self.actions = jnp.zeros((capacity, *action_shape))
        self.rewards = jnp.zeros(capacity)
        self.next_observations = jnp.zeros((capacity, *observation_shape))
        self.dones = jnp.zeros(capacity, dtype=bool)
        
        # Extra fields for method-specific data
        self.extras = {
            key: jnp.zeros((capacity, *shape))
            for key, shape in self.extra_fields.items()
        }
        
        self.ptr = 0
        self.size = 0
    
    def add(
        self,
        observation: jnp.ndarray,
        action: jnp.ndarray,
        reward: float,
        next_observation: jnp.ndarray,
        done: bool,
        **extras
    ):
        """Add experience to buffer."""
        self.observations = self.observations.at[self.ptr].set(observation)
        self.actions = self.actions.at[self.ptr].set(action)
        self.rewards = self.rewards.at[self.ptr].set(reward)
        self.next_observations = self.next_observations.at[self.ptr].set(next_observation)
        self.dones = self.dones.at[self.ptr].set(done)
        
        for key, value in extras.items():
            if key in self.extras:
                self.extras[key] = self.extras[key].at[self.ptr].set(value)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(
        self,
        batch_size: int,
        key: jax.random.PRNGKey
    ) -> Dict[str, jnp.ndarray]:
        """Sample batch from buffer."""
        indices = jax.random.choice(
            key, self.size, shape=(batch_size,), replace=False
        )
        
        batch = {
            'observations': self.observations[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_observations': self.next_observations[indices],
            'dones': self.dones[indices],
        }
        
        # Add extra fields
        for key, values in self.extras.items():
            batch[key] = values[indices]
        
        return batch
    
    def get_all(self) -> Dict[str, jnp.ndarray]:
        """Get all experiences in buffer."""
        batch = {
            'observations': self.observations[:self.size],
            'actions': self.actions[:self.size],
            'rewards': self.rewards[:self.size],
            'next_observations': self.next_observations[:self.size],
            'dones': self.dones[:self.size],
        }
        
        for key, values in self.extras.items():
            batch[key] = values[:self.size]
        
        return batch
    
    def clear(self):
        """Clear buffer."""
        self.ptr = 0
        self.size = 0


# ============================================================================
# Training Method Registry
# ============================================================================

class TrainingMethodRegistry:
    """
    Registry for RL training methods.
    Allows easy addition of new methods without modifying core code.
    """
    
    _methods: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a training method."""
        def decorator(method_cls):
            cls._methods[name] = method_cls
            return method_cls
        return decorator
    
    @classmethod
    def get(cls, name: str) -> type:
        """Get training method by name."""
        if name not in cls._methods:
            raise ValueError(
                f"Training method '{name}' not found. "
                f"Available: {list(cls._methods.keys())}"
            )
        return cls._methods[name]
    
    @classmethod
    def list_methods(cls) -> list:
        """List all registered methods."""
        return list(cls._methods.keys())


# ============================================================================
# Base Trainer Class
# ============================================================================

class WANNTrainer:
    """
    Base trainer class that coordinates architecture and training method.
    This is the main entry point for training WANN policies.
    """
    
    def __init__(
        self,
        architecture: WANNArchitecture,
        training_method: str,
        config: TrainingConfig,
        env_factory: Callable,
        **method_kwargs
    ):
        """
        Initialize WANN trainer.
        
        Args:
            architecture: WANN architecture from search
            training_method: Name of training method ('dqn', 'ppo', etc.)
            config: Training configuration
            env_factory: Factory function to create environment
            **method_kwargs: Method-specific arguments
        """
        self.architecture = architecture
        self.config = config
        self.env_factory = env_factory
        
        # Create policy using registered training method
        method_cls = TrainingMethodRegistry.get(training_method)
        self.policy = method_cls(
            architecture=architecture,
            config=config,
            **method_kwargs
        )
        
        # Create environment
        self.env = env_factory()
        
        # Initialize parameters
        key = jax.random.PRNGKey(42)
        self.params = architecture.init_params(key)
        self.policy.set_params(self.params)
        
        # Metrics
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'training_losses': [],
        }
    
    def train(self, num_steps: int):
        """
        Main training loop.
        
        Args:
            num_steps: Total number of environment steps
        """
        observation, info = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(num_steps):
            # Select action
            action = self.policy.select_action(observation)
            
            # Step environment
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store transition (method-specific)
            self.policy.store_transition(
                observation, action, reward, next_observation, done
            )
            
            episode_reward += reward
            episode_length += 1
            
            # Update if ready
            if self.policy.ready_to_update():
                metrics = self.policy.update_step()
                self.metrics['training_losses'].append(metrics.get('loss', 0))
            
            # Episode end
            if done:
                self.metrics['episode_rewards'].append(episode_reward)
                self.metrics['episode_lengths'].append(episode_length)
                
                observation, info = self.env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                observation = next_observation
            
            # Logging
            if step % self.config.log_frequency == 0:
                self._log_metrics(step)
            
            # Evaluation
            if step % self.config.eval_frequency == 0:
                self._evaluate()
            
            # Checkpointing
            if step % self.config.checkpoint_frequency == 0:
                self._save_checkpoint(step)
    
    def _log_metrics(self, step: int):
        """Log training metrics."""
        if len(self.metrics['episode_rewards']) > 0:
            mean_reward = jnp.mean(jnp.array(self.metrics['episode_rewards'][-100:]))
            print(f"Step {step}: Mean Reward = {mean_reward:.2f}")
    
    def _evaluate(self):
        """Evaluate current policy."""
        # Evaluation logic
        pass
    
    def _save_checkpoint(self, step: int):
        """Save training checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_{step}.pkl"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'params': self.policy.get_params(),
                'step': step,
                'metrics': self.metrics,
            }, f)


# ============================================================================
# Example: Minimal Policy Template
# ============================================================================

class MinimalPolicy(PolicyInterface):
    """
    Minimal example showing what a training method needs to implement.
    This serves as a template for adding new methods.
    """
    
    def __init__(
        self,
        architecture: WANNArchitecture,
        config: TrainingConfig,
        **kwargs
    ):
        self.architecture = architecture
        self.config = config
        self.params = None
    
    def select_action(self, observation, deterministic=False):
        # Implement action selection
        raise NotImplementedError
    
    def compute_loss(self, batch):
        # Implement loss computation
        raise NotImplementedError
    
    def update(self, batch):
        # Implement parameter update
        raise NotImplementedError
    
    def get_params(self):
        return self.params
    
    def set_params(self, params):
        self.params = params
    
    def store_transition(self, obs, action, reward, next_obs, done):
        # Store transition for later training
        pass
    
    def ready_to_update(self) -> bool:
        # Check if ready to perform update
        return False
    
    def update_step(self) -> Dict[str, float]:
        # Perform one update step
        return {}


# ============================================================================
# Utility Functions
# ============================================================================

def create_trainer_from_checkpoint(
    checkpoint_path: str,
    training_method: str,
    env_factory: Callable,
    config: Optional[TrainingConfig] = None
) -> WANNTrainer:
    """
    Create trainer from saved architecture.
    
    Args:
        checkpoint_path: Path to architecture checkpoint
        training_method: Training method to use
        env_factory: Environment factory
        config: Training config (uses default if None)
    
    Returns:
        Initialized trainer
    """
    # Load architecture
    spec = ArchitectureSpec.load(checkpoint_path)
    architecture = WANNArchitecture(spec)
    
    # Create trainer
    config = config or TrainingConfig()
    trainer = WANNTrainer(
        architecture=architecture,
        training_method=training_method,
        config=config,
        env_factory=env_factory
    )
    
    return trainer


def list_available_methods() -> list:
    """List all available training methods."""
    return TrainingMethodRegistry.list_methods()
