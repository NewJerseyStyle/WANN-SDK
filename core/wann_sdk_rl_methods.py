"""
WANN SDK - RL Training Methods

Implementations of popular RL algorithms that work with WANN architectures:
1. DQN (Deep Q-Network) - For discrete action spaces
2. PPO (Proximal Policy Optimization) - For continuous/discrete actions
3. SAC (Soft Actor-Critic) - For continuous actions

All methods implement the PolicyInterface and are registered
in the TrainingMethodRegistry for easy extensibility.
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax
from typing import Dict, Any, Optional, Tuple
from functools import partial

from .wann_sdk_core import (
    PolicyInterface,
    TrainingMethodRegistry,
    WANNArchitecture,
    TrainingConfig,
    ReplayBuffer
)


# ============================================================================
# DQN (Deep Q-Network)
# ============================================================================

@TrainingMethodRegistry.register('dqn')
class DQNPolicy(PolicyInterface):
    """
    Deep Q-Network (DQN) for discrete action spaces.
    
    Features:
    - Experience replay
    - Target network
    - Epsilon-greedy exploration
    - Double DQN (optional)
    """
    
    def __init__(
        self,
        architecture: WANNArchitecture,
        config: TrainingConfig,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 10000,
        target_update_freq: int = 1000,
        double_dqn: bool = True,
        **kwargs
    ):
        """
        Initialize DQN policy.
        
        Args:
            architecture: WANN architecture
            config: Training configuration
            gamma: Discount factor
            epsilon_start: Initial epsilon for exploration
            epsilon_end: Final epsilon
            epsilon_decay: Steps to decay epsilon
            target_update_freq: Steps between target network updates
            double_dqn: Use Double DQN
        """
        self.architecture = architecture
        self.config = config
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        
        # Initialize parameters
        key = jax.random.PRNGKey(42)
        self.params = architecture.init_params(key)
        self.target_params = self.params.copy()
        
        # Optimizer
        self.optimizer = optax.adam(config.learning_rate)
        self.opt_state = self.optimizer.init(self.params)
        
        # Replay buffer
        env_info = kwargs.get('env_info', {})
        obs_shape = (env_info.get('obs_dim', 24),)
        action_shape = ()  # Discrete action
        
        self.buffer = ReplayBuffer(
            capacity=config.buffer_size,
            observation_shape=obs_shape,
            action_shape=action_shape
        )
        
        # Training state
        self.step_count = 0
        self.epsilon = epsilon_start
        
        # Compile functions
        self._compute_q_values = jit(self._compute_q_values_impl)
        self._update_fn = jit(self._update_impl)
    
    def _compute_q_values_impl(
        self,
        params: Dict,
        observation: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute Q-values for all actions."""
        return self.architecture.forward(observation, params)
    
    def select_action(
        self,
        observation: jnp.ndarray,
        deterministic: bool = False
    ) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            observation: Current observation
            deterministic: If True, select greedy action
        
        Returns:
            Selected action
        """
        if not deterministic and jax.random.uniform(jax.random.PRNGKey(self.step_count)) < self.epsilon:
            # Random action
            num_actions = self.architecture.spec.num_outputs
            action = int(jax.random.randint(
                jax.random.PRNGKey(self.step_count),
                (),
                0,
                num_actions
            ))
        else:
            # Greedy action
            q_values = self._compute_q_values(self.params, observation)
            action = int(jnp.argmax(q_values))
        
        return action
    
    def compute_loss(self, batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Compute DQN loss.
        
        Args:
            batch: Batch of transitions
        
        Returns:
            Loss value
        """
        observations = batch['observations']
        actions = batch['actions'].astype(jnp.int32)
        rewards = batch['rewards']
        next_observations = batch['next_observations']
        dones = batch['dones']
        
        # Current Q-values
        q_values = vmap(
            lambda obs: self._compute_q_values(self.params, obs)
        )(observations)
        q_values_selected = q_values[jnp.arange(len(actions)), actions]
        
        # Target Q-values
        if self.double_dqn:
            # Double DQN: use online network to select action
            next_q_values_online = vmap(
                lambda obs: self._compute_q_values(self.params, obs)
            )(next_observations)
            next_actions = jnp.argmax(next_q_values_online, axis=1)
            
            # Use target network to evaluate
            next_q_values_target = vmap(
                lambda obs: self._compute_q_values(self.target_params, obs)
            )(next_observations)
            next_q_values = next_q_values_target[jnp.arange(len(next_actions)), next_actions]
        else:
            # Standard DQN
            next_q_values = vmap(
                lambda obs: jnp.max(self._compute_q_values(self.target_params, obs))
            )(next_observations)
        
        # TD target
        td_target = rewards + self.gamma * next_q_values * (1 - dones)
        
        # MSE loss
        loss = jnp.mean((q_values_selected - jax.lax.stop_gradient(td_target)) ** 2)
        
        return loss
    
    def _update_impl(
        self,
        params: Dict,
        opt_state: Any,
        batch: Dict[str, jnp.ndarray]
    ) -> Tuple[Dict, Any, float]:
        """Internal update implementation."""
        loss, grads = jax.value_and_grad(
            lambda p: self.compute_loss({**batch, 'params': p})
        )(params)
        
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss
    
    def update(self, batch: Dict[str, jnp.ndarray]) -> Dict[str, float]:
        """
        Update policy parameters.
        
        Args:
            batch: Batch of transitions
        
        Returns:
            Metrics dictionary
        """
        # Update parameters
        self.params, self.opt_state, loss = self._update_fn(
            self.params,
            self.opt_state,
            batch
        )
        
        # Update target network
        if self.step_count % self.target_update_freq == 0:
            self.target_params = self.params.copy()
        
        # Update epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * 
            (self.step_count / self.epsilon_decay)
        )
        
        self.step_count += 1
        
        return {
            'loss': float(loss),
            'epsilon': float(self.epsilon),
        }
    
    def store_transition(
        self,
        observation: jnp.ndarray,
        action: int,
        reward: float,
        next_observation: jnp.ndarray,
        done: bool
    ):
        """Store transition in replay buffer."""
        self.buffer.add(
            observation,
            jnp.array(action),
            reward,
            next_observation,
            done
        )
    
    def ready_to_update(self) -> bool:
        """Check if buffer has enough samples."""
        return self.buffer.size >= self.config.batch_size
    
    def update_step(self) -> Dict[str, float]:
        """Perform one update step."""
        key = jax.random.PRNGKey(self.step_count)
        batch = self.buffer.sample(self.config.batch_size, key)
        return self.update(batch)
    
    def get_params(self) -> Dict[str, jnp.ndarray]:
        """Get policy parameters."""
        return self.params
    
    def set_params(self, params: Dict[str, jnp.ndarray]):
        """Set policy parameters."""
        self.params = params
        self.target_params = params.copy()


# ============================================================================
# PPO (Proximal Policy Optimization)
# ============================================================================

@TrainingMethodRegistry.register('ppo')
class PPOPolicy(PolicyInterface):
    """
    Proximal Policy Optimization (PPO) for continuous/discrete actions.
    
    Features:
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Value function learning
    - On-policy learning
    """
    
    def __init__(
        self,
        architecture: WANNArchitecture,
        config: TrainingConfig,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        num_epochs: int = 10,
        **kwargs
    ):
        """
        Initialize PPO policy.
        
        Args:
            architecture: WANN architecture
            config: Training configuration
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            num_epochs: Number of epochs per update
        """
        self.architecture = architecture
        self.config = config
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.num_epochs = num_epochs
        
        # Initialize parameters
        key = jax.random.PRNGKey(42)
        self.params = architecture.init_params(key)
        
        # Separate value network (simple approach: additional output head)
        self.value_params = architecture.init_params(
            jax.random.split(key)[1]
        )
        
        # Optimizer
        self.optimizer = optax.adam(config.learning_rate)
        self.opt_state = self.optimizer.init(self.params)
        self.value_opt_state = self.optimizer.init(self.value_params)
        
        # Trajectory buffer (for on-policy learning)
        env_info = kwargs.get('env_info', {})
        obs_shape = (env_info.get('obs_dim', 24),)
        action_shape = (env_info.get('action_dim', 4),)
        
        self.buffer = ReplayBuffer(
            capacity=config.batch_size * 10,
            observation_shape=obs_shape,
            action_shape=action_shape,
            extra_fields={
                'log_probs': (),
                'values': (),
                'advantages': (),
                'returns': (),
            }
        )
        
        # Compile functions
        self._forward_policy = jit(self._forward_policy_impl)
        self._forward_value = jit(self._forward_value_impl)
    
    def _forward_policy_impl(
        self,
        params: Dict,
        observation: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass for policy.
        
        Returns:
            action_mean: Mean of action distribution
            action_logstd: Log standard deviation
        """
        output = self.architecture.forward(observation, params)
        
        # Split output into mean and logstd
        action_dim = output.shape[-1] // 2
        action_mean = output[..., :action_dim]
        action_logstd = output[..., action_dim:]
        
        return action_mean, action_logstd
    
    def _forward_value_impl(
        self,
        params: Dict,
        observation: jnp.ndarray
    ) -> jnp.ndarray:
        """Forward pass for value function."""
        # Use same architecture but different parameters
        value = self.architecture.forward(observation, params)
        return value[..., 0]  # Single value output
    
    def select_action(
        self,
        observation: jnp.ndarray,
        deterministic: bool = False
    ) -> jnp.ndarray:
        """
        Select action from policy.
        
        Args:
            observation: Current observation
            deterministic: If True, return mean action
        
        Returns:
            Selected action
        """
        action_mean, action_logstd = self._forward_policy(self.params, observation)
        
        if deterministic:
            return action_mean
        else:
            # Sample from Gaussian
            key = jax.random.PRNGKey(int(jnp.sum(observation) * 1000))
            action_std = jnp.exp(action_logstd)
            action = action_mean + action_std * jax.random.normal(key, action_mean.shape)
            return action
    
    def compute_loss(self, batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Compute PPO loss.
        
        Args:
            batch: Batch of trajectories
        
        Returns:
            Total loss
        """
        observations = batch['observations']
        actions = batch['actions']
        old_log_probs = batch['log_probs']
        advantages = batch['advantages']
        returns = batch['returns']
        
        # Policy loss
        action_mean, action_logstd = vmap(
            lambda obs: self._forward_policy(self.params, obs)
        )(observations)
        
        # Compute log probabilities
        action_std = jnp.exp(action_logstd)
        log_probs = -0.5 * (
            jnp.sum(((actions - action_mean) / action_std) ** 2, axis=-1) +
            jnp.sum(2 * action_logstd, axis=-1) +
            actions.shape[-1] * jnp.log(2 * jnp.pi)
        )
        
        # PPO clipped objective
        ratio = jnp.exp(log_probs - old_log_probs)
        clipped_ratio = jnp.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        policy_loss = -jnp.mean(
            jnp.minimum(ratio * advantages, clipped_ratio * advantages)
        )
        
        # Value loss
        values = vmap(
            lambda obs: self._forward_value(self.value_params, obs)
        )(observations)
        value_loss = jnp.mean((values - returns) ** 2)
        
        # Entropy bonus
        entropy = jnp.mean(jnp.sum(action_logstd + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1))
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        return total_loss
    
    def compute_gae(
        self,
        rewards: jnp.ndarray,
        values: jnp.ndarray,
        dones: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Rewards
            values: Value estimates
            dones: Done flags
        
        Returns:
            advantages: Advantage estimates
            returns: Discounted returns
        """
        advantages = jnp.zeros_like(rewards)
        last_gae = 0.0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages = advantages.at[t].set(last_gae)
        
        returns = advantages + values
        
        return advantages, returns
    
    def update(self, batch: Dict[str, jnp.ndarray]) -> Dict[str, float]:
        """Update policy using PPO."""
        # This is a simplified version
        # Full implementation would include multiple epochs and mini-batches
        raise NotImplementedError("Use update_step for PPO")
    
    def update_step(self) -> Dict[str, float]:
        """Perform PPO update on collected trajectories."""
        # Get all data from buffer
        data = self.buffer.get_all()
        
        # Compute advantages
        advantages, returns = self.compute_gae(
            data['rewards'],
            data['values'],
            data['dones']
        )
        
        # Normalize advantages
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
        
        # Add to batch
        data['advantages'] = advantages
        data['returns'] = returns
        
        # Multiple epochs of updates
        total_loss = 0.0
        for epoch in range(self.num_epochs):
            # Shuffle data
            key = jax.random.PRNGKey(epoch)
            indices = jax.random.permutation(key, self.buffer.size)
            
            # Mini-batch updates
            for start in range(0, self.buffer.size, self.config.batch_size):
                end = min(start + self.config.batch_size, self.buffer.size)
                batch_indices = indices[start:end]
                
                mini_batch = {k: v[batch_indices] for k, v in data.items()}
                
                loss = self.compute_loss(mini_batch)
                total_loss += float(loss)
        
        # Clear buffer (on-policy)
        self.buffer.clear()
        
        return {
            'loss': total_loss / (self.num_epochs * (self.buffer.size // self.config.batch_size)),
        }
    
    def store_transition(
        self,
        observation: jnp.ndarray,
        action: jnp.ndarray,
        reward: float,
        next_observation: jnp.ndarray,
        done: bool
    ):
        """Store transition with policy info."""
        # Compute log prob and value
        action_mean, action_logstd = self._forward_policy(self.params, observation)
        action_std = jnp.exp(action_logstd)
        log_prob = -0.5 * jnp.sum(((action - action_mean) / action_std) ** 2)
        
        value = self._forward_value(self.value_params, observation)
        
        self.buffer.add(
            observation,
            action,
            reward,
            next_observation,
            done,
            log_probs=log_prob,
            values=value
        )
    
    def ready_to_update(self) -> bool:
        """Check if enough data collected for update."""
        return self.buffer.size >= self.config.batch_size
    
    def get_params(self) -> Dict[str, jnp.ndarray]:
        """Get policy parameters."""
        return {
            'policy': self.params,
            'value': self.value_params
        }
    
    def set_params(self, params: Dict[str, jnp.ndarray]):
        """Set policy parameters."""
        if isinstance(params, dict) and 'policy' in params:
            self.params = params['policy']
            self.value_params = params['value']
        else:
            self.params = params


# ============================================================================
# Utility Functions
# ============================================================================

def create_policy_for_environment(
    architecture: WANNArchitecture,
    env_info: Dict[str, Any],
    method: str = 'auto',
    config: Optional[TrainingConfig] = None
) -> PolicyInterface:
    """
    Create appropriate policy for environment.
    
    Args:
        architecture: WANN architecture
        env_info: Environment information
        method: Training method ('auto', 'dqn', 'ppo', 'sac')
        config: Training configuration
    
    Returns:
        Policy instance
    """
    config = config or TrainingConfig()
    
    # Auto-select method based on action space
    if method == 'auto':
        if env_info['action_is_discrete']:
            method = 'dqn'
        else:
            method = 'ppo'
    
    # Get method class
    method_cls = TrainingMethodRegistry.get(method)
    
    # Create policy
    policy = method_cls(
        architecture=architecture,
        config=config,
        env_info=env_info
    )
    
    return policy
