"""
WANN SDK - Ray Environment Service and Gymnasium Integration

Provides distributed environment execution via Ray:
1. Ray Remote Functions - For distributed rollout collection
2. Ray Serve FastAPI - For environment as a service
3. Gymnasium wrapper - Standardized interface

This allows WANN architecture search and training to scale
across multiple workers while maintaining compatibility with
all Gymnasium environments.
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Callable
from dataclasses import dataclass
import gymnasium as gym
from abc import ABC, abstractmethod

# Ray imports (optional)
try:
    import ray
    from ray import serve
    from ray.serve import deployment
    from fastapi import FastAPI
    from pydantic import BaseModel
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("Ray not available. Install with: pip install ray[serve]")


# ============================================================================
# Gymnasium Environment Wrapper
# ============================================================================

class GymnasiumEnvWrapper:
    """
    Wrapper for Gymnasium environments to work with WANN SDK.
    Handles observation/action space conversions and JAX compatibility.
    """
    
    def __init__(
        self,
        env_name: str,
        render_mode: Optional[str] = None,
        **env_kwargs
    ):
        """
        Initialize Gymnasium environment.
        
        Args:
            env_name: Environment name (e.g., 'BipedalWalker-v3')
            render_mode: Render mode ('human', 'rgb_array', None)
            **env_kwargs: Additional environment arguments
        """
        self.env_name = env_name
        self.env = gym.make(env_name, render_mode=render_mode, **env_kwargs)
        
        # Get space information
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # Determine space types
        self.obs_is_discrete = isinstance(
            self.observation_space, 
            gym.spaces.Discrete
        )
        self.action_is_discrete = isinstance(
            self.action_space,
            gym.spaces.Discrete
        )
        
        # Get dimensions
        self.obs_dim = self._get_space_dim(self.observation_space)
        self.action_dim = self._get_space_dim(self.action_space)
    
    def _get_space_dim(self, space: gym.Space) -> int:
        """Get dimension of space."""
        if isinstance(space, gym.spaces.Discrete):
            return space.n
        elif isinstance(space, gym.spaces.Box):
            return int(np.prod(space.shape))
        elif isinstance(space, gym.spaces.MultiDiscrete):
            return int(np.sum(space.nvec))
        else:
            raise ValueError(f"Unsupported space type: {type(space)}")
    
    def reset(self, seed: Optional[int] = None) -> Tuple[jnp.ndarray, Dict]:
        """Reset environment."""
        obs, info = self.env.reset(seed=seed)
        obs = self._process_observation(obs)
        return jnp.array(obs), info
    
    def step(self, action: jnp.ndarray) -> Tuple[jnp.ndarray, float, bool, bool, Dict]:
        """Step environment."""
        action = self._process_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._process_observation(obs)
        return jnp.array(obs), float(reward), terminated, truncated, info
    
    def _process_observation(self, obs: np.ndarray) -> np.ndarray:
        """Process observation to standard format."""
        if isinstance(obs, (int, np.integer)):
            # Discrete observation to one-hot
            obs_array = np.zeros(self.obs_dim)
            obs_array[obs] = 1.0
            return obs_array
        else:
            # Flatten if needed
            return obs.flatten()
    
    def _process_action(self, action: jnp.ndarray) -> Any:
        """Process action from network output."""
        action = np.array(action)
        
        if self.action_is_discrete:
            # Convert to discrete action
            if action.shape == ():
                return int(action)
            else:
                return int(np.argmax(action))
        else:
            # Continuous action
            # Clip to action space bounds
            if isinstance(self.action_space, gym.spaces.Box):
                action = np.clip(
                    action,
                    self.action_space.low,
                    self.action_space.high
                )
            return action
    
    def close(self):
        """Close environment."""
        self.env.close()
    
    def get_env_info(self) -> Dict[str, Any]:
        """Get environment information."""
        return {
            'env_name': self.env_name,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'obs_is_discrete': self.obs_is_discrete,
            'action_is_discrete': self.action_is_discrete,
            'obs_shape': self.observation_space.shape,
            'action_shape': self.action_space.shape,
        }


# ============================================================================
# Ray Remote Environment (Distributed Rollouts)
# ============================================================================

if RAY_AVAILABLE:
    @ray.remote
    class RayRemoteEnv:
        """
        Ray remote environment for distributed rollout collection.
        Each worker runs in its own process with its own environment instance.
        """
        
        def __init__(
            self,
            env_name: str,
            worker_id: int,
            **env_kwargs
        ):
            """
            Initialize remote environment.
            
            Args:
                env_name: Gymnasium environment name
                worker_id: Unique worker identifier
                **env_kwargs: Environment arguments
            """
            self.env = GymnasiumEnvWrapper(env_name, **env_kwargs)
            self.worker_id = worker_id
        
        def reset(self, seed: Optional[int] = None):
            """Reset environment."""
            return self.env.reset(seed)
        
        def step(self, action):
            """Step environment."""
            return self.env.step(action)
        
        def rollout(
            self,
            policy_fn: Callable,
            policy_params: Dict,
            max_steps: int = 1000,
            render: bool = False
        ) -> Dict[str, Any]:
            """
            Collect a full episode rollout.
            
            Args:
                policy_fn: Function that maps (params, obs) -> action
                policy_params: Policy parameters
                max_steps: Maximum steps per episode
                render: Whether to render
            
            Returns:
                Dictionary containing episode data
            """
            observations = []
            actions = []
            rewards = []
            dones = []
            
            obs, info = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(max_steps):
                # Select action using policy
                action = policy_fn(policy_params, obs)
                
                # Store data
                observations.append(np.array(obs))
                actions.append(np.array(action))
                
                # Step environment
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                rewards.append(reward)
                dones.append(done)
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
                
                obs = next_obs
            
            return {
                'observations': np.array(observations),
                'actions': np.array(actions),
                'rewards': np.array(rewards),
                'dones': np.array(dones),
                'episode_reward': episode_reward,
                'episode_length': episode_length,
                'worker_id': self.worker_id,
            }
        
        def get_env_info(self):
            """Get environment information."""
            return self.env.get_env_info()
        
        def close(self):
            """Close environment."""
            self.env.close()


# ============================================================================
# Ray Serve API (Environment as a Service)
# ============================================================================

if RAY_AVAILABLE:
    # Pydantic models for API
    class ResetRequest(BaseModel):
        seed: Optional[int] = None
    
    class StepRequest(BaseModel):
        action: List[float]
    
    class RolloutRequest(BaseModel):
        policy_params: Dict[str, List[float]]
        max_steps: int = 1000
    
    
    @serve.deployment(num_replicas=2)
    class EnvironmentService:
        """
        FastAPI service for Gymnasium environment.
        Allows remote interaction via HTTP API.
        """
        
        def __init__(self, env_name: str = "BipedalWalker-v3"):
            """Initialize environment service."""
            self.env = GymnasiumEnvWrapper(env_name)
            self.current_obs = None
            
            # Create FastAPI app
            self.app = FastAPI(title="WANN Environment Service")
            
            # Register routes
            self.app.add_api_route("/reset", self.reset, methods=["POST"])
            self.app.add_api_route("/step", self.step, methods=["POST"])
            self.app.add_api_route("/info", self.info, methods=["GET"])
            self.app.add_api_route("/close", self.close, methods=["POST"])
        
        async def reset(self, request: ResetRequest):
            """Reset environment."""
            obs, info = self.env.reset(seed=request.seed)
            self.current_obs = obs
            return {
                "observation": obs.tolist(),
                "info": info
            }
        
        async def step(self, request: StepRequest):
            """Step environment."""
            action = jnp.array(request.action)
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.current_obs = obs
            
            return {
                "observation": obs.tolist(),
                "reward": float(reward),
                "terminated": terminated,
                "truncated": truncated,
                "info": info
            }
        
        async def info(self):
            """Get environment info."""
            return self.env.get_env_info()
        
        async def close(self):
            """Close environment."""
            self.env.close()
            return {"status": "closed"}


# ============================================================================
# Distributed Environment Pool
# ============================================================================

class DistributedEnvPool:
    """
    Pool of distributed environments for parallel rollout collection.
    Uses Ray for distribution across workers.
    """
    
    def __init__(
        self,
        env_name: str,
        num_workers: int = 4,
        **env_kwargs
    ):
        """
        Initialize distributed environment pool.
        
        Args:
            env_name: Gymnasium environment name
            num_workers: Number of parallel workers
            **env_kwargs: Environment arguments
        """
        if not RAY_AVAILABLE:
            raise ImportError("Ray is required for distributed environments")
        
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        self.env_name = env_name
        self.num_workers = num_workers
        
        # Create remote environments
        self.envs = [
            RayRemoteEnv.remote(env_name, worker_id=i, **env_kwargs)
            for i in range(num_workers)
        ]
        
        # Get environment info from first worker
        self.env_info = ray.get(self.envs[0].get_env_info.remote())
    
    def collect_rollouts(
        self,
        policy_fn: Callable,
        policy_params: Dict,
        num_rollouts: int,
        max_steps: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Collect multiple rollouts in parallel.
        
        Args:
            policy_fn: Policy function
            policy_params: Policy parameters
            num_rollouts: Number of rollouts to collect
            max_steps: Maximum steps per rollout
        
        Returns:
            List of rollout data dictionaries
        """
        # Distribute rollouts across workers
        rollouts_per_worker = num_rollouts // self.num_workers
        
        # Launch parallel rollouts
        futures = []
        for i, env in enumerate(self.envs):
            num_worker_rollouts = rollouts_per_worker
            if i < num_rollouts % self.num_workers:
                num_worker_rollouts += 1
            
            for _ in range(num_worker_rollouts):
                future = env.rollout.remote(
                    policy_fn, policy_params, max_steps
                )
                futures.append(future)
        
        # Collect results
        rollouts = ray.get(futures)
        return rollouts
    
    def reset_all(self):
        """Reset all environments."""
        futures = [env.reset.remote() for env in self.envs]
        return ray.get(futures)
    
    def close_all(self):
        """Close all environments."""
        futures = [env.close.remote() for env in self.envs]
        ray.get(futures)
    
    def get_env_info(self) -> Dict[str, Any]:
        """Get environment information."""
        return self.env_info


# ============================================================================
# Environment Factory
# ============================================================================

class EnvFactory:
    """
    Factory for creating environments with consistent configuration.
    Supports local and distributed modes.
    """
    
    def __init__(
        self,
        env_name: str,
        mode: str = "local",
        num_workers: int = 4,
        **env_kwargs
    ):
        """
        Initialize environment factory.
        
        Args:
            env_name: Gymnasium environment name
            mode: 'local', 'distributed', or 'service'
            num_workers: Number of workers (for distributed mode)
            **env_kwargs: Environment arguments
        """
        self.env_name = env_name
        self.mode = mode
        self.num_workers = num_workers
        self.env_kwargs = env_kwargs
    
    def create(self):
        """Create environment based on mode."""
        if self.mode == "local":
            return GymnasiumEnvWrapper(self.env_name, **self.env_kwargs)
        
        elif self.mode == "distributed":
            if not RAY_AVAILABLE:
                raise ImportError("Ray required for distributed mode")
            return DistributedEnvPool(
                self.env_name,
                self.num_workers,
                **self.env_kwargs
            )
        
        elif self.mode == "service":
            if not RAY_AVAILABLE:
                raise ImportError("Ray required for service mode")
            # This would start a Ray Serve deployment
            raise NotImplementedError("Service mode not yet implemented")
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def get_env_info(self) -> Dict[str, Any]:
        """Get environment information."""
        # Create temporary environment to get info
        env = GymnasiumEnvWrapper(self.env_name, **self.env_kwargs)
        info = env.get_env_info()
        env.close()
        return info


# ============================================================================
# Utilities
# ============================================================================

def test_environment(env_name: str, num_episodes: int = 5):
    """
    Test environment with random actions.
    
    Args:
        env_name: Environment name
        num_episodes: Number of test episodes
    """
    print(f"Testing environment: {env_name}")
    
    env = GymnasiumEnvWrapper(env_name)
    info = env.get_env_info()
    
    print(f"Environment Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print(f"\nRunning {num_episodes} random episodes...")
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Random action
            if info['action_is_discrete']:
                action = jnp.array(np.random.randint(0, info['action_dim']))
            else:
                action = jnp.array(
                    np.random.uniform(-1, 1, size=(info['action_dim'],))
                )
            
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        print(f"  Episode {ep+1}: Reward={episode_reward:.2f}, Length={episode_length}")
    
    env.close()
    print("Test completed!")


def start_environment_service(
    env_name: str = "BipedalWalker-v3",
    port: int = 8000
):
    """
    Start Ray Serve environment service.

    Args:
        env_name: Environment name
        port: Service port
    """
    if not RAY_AVAILABLE:
        raise ImportError("Ray required for environment service")

    # Initialize Ray Serve
    serve.start(http_options={"port": port})

    # Deploy environment service
    app = EnvironmentService.bind(env_name=env_name)
    serve.run(app, route_prefix="/env")

    print(f"Environment service started at http://localhost:{port}/env")
    print("Endpoints:")
    print("  POST /env/reset - Reset environment")
    print("  POST /env/step - Step environment")
    print("  GET  /env/info - Get environment info")
    print("  POST /env/close - Close environment")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="WANN Environment Service")
    parser.add_argument(
        "--mode",
        choices=["test", "distributed", "service"],
        default="test",
        help="Mode to run"
    )
    parser.add_argument(
        "--env",
        default="BipedalWalker-v3",
        help="Environment name"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of workers (for distributed mode)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port (for service mode)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "test":
        # Test environment
        test_environment(args.env, num_episodes=3)
    
    elif args.mode == "distributed":
        # Test distributed rollouts
        print(f"Testing distributed rollouts with {args.workers} workers...")
        
        if not RAY_AVAILABLE:
            print("Ray not available. Install with: pip install ray")
        else:
            pool = DistributedEnvPool(args.env, num_workers=args.workers)
            
            # Dummy policy
            def random_policy(params, obs):
                info = pool.get_env_info()
                if info['action_is_discrete']:
                    return jnp.array(np.random.randint(0, info['action_dim']))
                else:
                    return jnp.array(
                        np.random.uniform(-1, 1, size=(info['action_dim'],))
                    )
            
            # Collect rollouts
            rollouts = pool.collect_rollouts(
                random_policy,
                {},
                num_rollouts=10,
                max_steps=500
            )
            
            print(f"Collected {len(rollouts)} rollouts")
            for i, rollout in enumerate(rollouts):
                print(f"  Rollout {i}: "
                      f"Reward={rollout['episode_reward']:.2f}, "
                      f"Length={rollout['episode_length']}")
            
            pool.close_all()
    
    elif args.mode == "service":
        # Start environment service
        if not RAY_AVAILABLE:
            print("Ray not available. Install with: pip install ray[serve]")
        else:
            start_environment_service(args.env, args.port)
            
            print("\nPress Ctrl+C to stop service")
            try:
                import time
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down...")
                serve.shutdown()
