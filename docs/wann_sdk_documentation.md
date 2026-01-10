# WANN SDK - Developer Documentation

完整的 Weight Agnostic Neural Networks SDK,支持架構搜索和多種 RL 訓練方法。

## 📚 目錄

1. [概述](#概述)
2. [核心架構](#核心架構)
3. [安裝指南](#安裝指南)
4. [快速開始](#快速開始)
5. [核心組件](#核心組件)
6. [添加新訓練方法](#添加新訓練方法)
7. [分佈式訓練](#分佈式訓練)
8. [完整示例](#完整示例)
9. [API 參考](#api-參考)

## 🎯 概述

### 設計理念

WANN SDK 遵循以下設計原則:

1. **關注點分離**: 架構搜索與權重訓練解耦
2. **訓練方法無關**: 支持任意 RL 算法 (DQN, PPO, SAC, ...)
3. **環境無關**: 兼容所有 Gymnasium 環境
4. **可擴展性**: 易於添加新方法和組件
5. **分佈式優先**: 內建 Ray 支持

### 架構圖

```
┌─────────────────────────────────────────────────────────┐
│                    WANN SDK Architecture                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────┐         ┌───────────────────┐   │
│  │ Architecture     │         │ Training Methods  │   │
│  │ Search (WANN)    │────────▶│ - DQN             │   │
│  │                  │         │ - PPO             │   │
│  │ - TensorNEAT     │         │ - SAC             │   │
│  │ - EvoX           │         │ - Custom...       │   │
│  └──────────────────┘         └───────────────────┘   │
│           │                            │               │
│           │                            │               │
│           ▼                            ▼               │
│  ┌──────────────────┐         ┌───────────────────┐   │
│  │ Architecture     │         │ Policy Interface  │   │
│  │ Specification    │◀────────│                   │   │
│  └──────────────────┘         └───────────────────┘   │
│           │                            │               │
│           └────────────┬───────────────┘               │
│                        ▼                               │
│           ┌────────────────────────┐                   │
│           │  Environment Service   │                   │
│           │  - Gymnasium Wrapper   │                   │
│           │  - Ray Remote          │                   │
│           │  - Ray Serve API       │                   │
│           └────────────────────────┘                   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## 📦 安裝指南

### 基礎安裝

```bash
# 1. JAX
pip install -U "jax[cuda12]"  # GPU
# 或
pip install -U jax             # CPU

# 2. TensorNEAT
pip install git+https://github.com/EMI-Group/tensorneat.git

# 3. Gymnasium
pip install gymnasium[box2d]

# 4. 其他依賴
pip install optax
```

### 分佈式訓練 (可選)

```bash
# Ray for distributed computing
pip install "ray[serve]"
```

### EvoX 集成 (可選)

```bash
# JAX-based EvoX
pip install "git+https://github.com/EMI-Group/evox@v0.9.1-dev"
```

### 驗證安裝

```bash
python -c "
import jax
import tensorneat
import gymnasium
import ray
print('All dependencies installed!')
"
```

## 🚀 快速開始

### 最小示例 - CartPole

```python
from wann_sdk_core import (
    ArchitectureSpec,
    WANNArchitecture,
    TrainingConfig,
    create_trainer_from_checkpoint
)
from wann_sdk_ray_env import EnvFactory

# 1. 假設已有架構 (來自搜索階段)
spec = ArchitectureSpec.load('./models/cartpole_arch.pkl')
architecture = WANNArchitecture(spec)

# 2. 創建環境工廠
env_factory = EnvFactory(env_name="CartPole-v1", mode="local")

# 3. 創建訓練器
config = TrainingConfig(
    learning_rate=1e-3,
    batch_size=128,
    num_epochs=100
)

trainer = create_trainer_from_checkpoint(
    checkpoint_path='./models/cartpole_arch.pkl',
    training_method='dqn',
    env_factory=env_factory.create,
    config=config
)

# 4. 訓練
trainer.train(num_steps=50000)

print("Training completed!")
```

### BipedalWalker 完整流程

```bash
# 1. 架構搜索
python wann_bipedal.py --mode search \
    --pop_size 1000 \
    --generations 100 \
    --workers 4

# 2. 權重訓練
python wann_bipedal.py --mode train \
    --method ppo \
    --steps 1000000 \
    --distributed

# 3. 評估
python wann_bipedal.py --mode eval \
    --render \
    --eval_episodes 10
```

## 🧩 核心組件

### 1. 架構規範 (ArchitectureSpec)

保存從架構搜索階段得到的網絡結構:

```python
from wann_sdk_core import ArchitectureSpec

spec = ArchitectureSpec(
    nodes=nodes_array,           # 節點配置
    connections=conns_array,     # 連接配置
    num_inputs=24,
    num_outputs=4,
    num_hidden=15,
    num_params=87,
    search_fitness=250.0,
    search_complexity=87,
    metadata={'env': 'BipedalWalker-v3'}
)

# 保存
spec.save('./my_architecture.pkl')

# 加載
spec = ArchitectureSpec.load('./my_architecture.pkl')
```

### 2. WANN 架構 (WANNArchitecture)

將架構規範轉換為可訓練的網絡:

```python
from wann_sdk_core import WANNArchitecture

architecture = WANNArchitecture(spec)

# 初始化參數
key = jax.random.PRNGKey(42)
params = architecture.init_params(key)

# 前向傳播
import jax.numpy as jnp
observation = jnp.zeros(24)
output = architecture.forward(observation, params)

# 獲取架構信息
info = architecture.get_architecture_info()
print(info)
# {
#     'num_inputs': 24,
#     'num_outputs': 4,
#     'num_hidden': 15,
#     'num_connections': 87,
#     'num_params': 87,
#     'search_fitness': 250.0
# }
```

### 3. 策略接口 (PolicyInterface)

所有訓練方法必須實現的接口:

```python
from wann_sdk_core import PolicyInterface

class MyCustomPolicy(PolicyInterface):
    def select_action(self, observation, deterministic=False):
        """選擇動作"""
        pass
    
    def compute_loss(self, batch):
        """計算損失"""
        pass
    
    def update(self, batch):
        """更新參數"""
        pass
    
    def get_params(self):
        """獲取參數"""
        pass
    
    def set_params(self, params):
        """設置參數"""
        pass
```

### 4. 環境包裝器 (GymnasiumEnvWrapper)

標準化 Gymnasium 環境接口:

```python
from wann_sdk_ray_env import GymnasiumEnvWrapper

env = GymnasiumEnvWrapper("BipedalWalker-v3")

# 獲取環境信息
info = env.get_env_info()
print(info)
# {
#     'env_name': 'BipedalWalker-v3',
#     'obs_dim': 24,
#     'action_dim': 4,
#     'obs_is_discrete': False,
#     'action_is_discrete': False,
#     ...
# }

# 使用環境
obs, info = env.reset()
action = jnp.array([0.5, -0.3, 0.2, 0.1])
obs, reward, terminated, truncated, info = env.step(action)
```

## 🔧 添加新訓練方法

### 步驟 1: 實現策略接口

```python
from wann_sdk_core import PolicyInterface, TrainingMethodRegistry
import optax
import jax.numpy as jnp

@TrainingMethodRegistry.register('my_method')
class MyMethodPolicy(PolicyInterface):
    """
    自定義訓練方法。
    """
    
    def __init__(
        self,
        architecture,
        config,
        **kwargs
    ):
        self.architecture = architecture
        self.config = config
        
        # 初始化參數
        key = jax.random.PRNGKey(42)
        self.params = architecture.init_params(key)
        
        # 優化器
        self.optimizer = optax.adam(config.learning_rate)
        self.opt_state = self.optimizer.init(self.params)
        
        # 其他組件 (replay buffer, etc.)
        # ...
    
    def select_action(self, observation, deterministic=False):
        """實現動作選擇邏輯"""
        output = self.architecture.forward(observation, self.params)
        
        if deterministic:
            return output  # 或其他處理
        else:
            # 添加探索噪聲
            noise = jax.random.normal(key, output.shape) * 0.1
            return output + noise
    
    def compute_loss(self, batch):
        """實現損失計算"""
        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        
        # 你的損失函數
        predictions = self.architecture.forward(observations, self.params)
        loss = jnp.mean((predictions - actions) ** 2)
        
        return loss
    
    def update(self, batch):
        """實現參數更新"""
        loss, grads = jax.value_and_grad(self.compute_loss)(batch)
        
        updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state
        )
        self.params = optax.apply_updates(self.params, updates)
        
        return {'loss': float(loss)}
    
    def store_transition(self, obs, action, reward, next_obs, done):
        """存儲轉換 (如果使用 replay buffer)"""
        # 你的實現
        pass
    
    def ready_to_update(self):
        """檢查是否準備好更新"""
        return True  # 或基於 buffer 大小等條件
    
    def update_step(self):
        """執行一步更新"""
        # 從 buffer 採樣並更新
        return self.update(batch)
    
    def get_params(self):
        return self.params
    
    def set_params(self, params):
        self.params = params
```

### 步驟 2: 使用新方法

```python
from wann_sdk_core import create_trainer_from_checkpoint

# 現在可以使用你的新方法
trainer = create_trainer_from_checkpoint(
    checkpoint_path='./arch.pkl',
    training_method='my_method',  # 你註冊的名字
    env_factory=env_factory,
    config=config
)

trainer.train(num_steps=100000)
```

### 步驟 3: 添加方法特定功能

```python
# 例如: 添加優先級經驗回放
class MyMethodWithPER(MyMethodPolicy):
    def __init__(self, architecture, config, **kwargs):
        super().__init__(architecture, config, **kwargs)
        
        # 優先級經驗回放
        self.per_buffer = PrioritizedReplayBuffer(
            capacity=config.buffer_size,
            alpha=0.6,
            beta=0.4
        )
    
    def store_transition(self, obs, action, reward, next_obs, done):
        # 計算 TD error 作為優先級
        td_error = self._compute_td_error(obs, action, reward, next_obs, done)
        self.per_buffer.add(obs, action, reward, next_obs, done, td_error)
```

## 🌐 分佈式訓練

### Ray Remote 環境

使用多個 worker 並行收集經驗:

```python
from wann_sdk_ray_env import DistributedEnvPool

# 創建分佈式環境池
pool = DistributedEnvPool(
    env_name="BipedalWalker-v3",
    num_workers=8  # 8 個並行 worker
)

# 定義策略函數
def policy_fn(params, obs):
    return architecture.forward(obs, params)

# 並行收集 rollouts
rollouts = pool.collect_rollouts(
    policy_fn=policy_fn,
    policy_params=params,
    num_rollouts=100,
    max_steps=1000
)

# 處理結果
for rollout in rollouts:
    print(f"Worker {rollout['worker_id']}: "
          f"Reward={rollout['episode_reward']:.2f}")

# 清理
pool.close_all()
```

### Ray Serve API

將環境作為服務運行:

```python
from wann_sdk_ray_env import start_environment_service

# 啟動環境服務
start_environment_service(
    env_name="BipedalWalker-v3",
    port=8000
)

# 服務端點:
# POST /env/reset - 重置環境
# POST /env/step - 執行動作
# GET  /env/info - 獲取環境信息
# POST /env/close - 關閉環境
```

客戶端使用:

```python
import requests

# 重置
response = requests.post(
    "http://localhost:8000/env/reset",
    json={"seed": 42}
)
obs = response.json()['observation']

# 步進
response = requests.post(
    "http://localhost:8000/env/step",
    json={"action": [0.5, -0.3, 0.2, 0.1]}
)
result = response.json()
next_obs = result['observation']
reward = result['reward']
```

### 分佈式架構搜索

使用 EvoX 進行多設備架構搜索:

```python
from wann_evox_adapter import EvoXWANNAlgorithm

# 創建 WANN 算法
wann = WANN(pop_size=1024, genome=genome)

# 包裝為 EvoX 算法
distributed_config = {
    'num_devices': 4,
    'device_type': 'gpu'
}

algorithm = EvoXWANNAlgorithm(
    wann_algorithm=wann,
    distributed_config=distributed_config
)

# 自動分佈式訓練
for gen in range(num_generations):
    state, distributed_pop = algorithm.distributed_ask(state)
    fitness = problem.distributed_evaluate(state, distributed_pop)
    state = algorithm.distributed_tell(state, fitness)
```

## 📖 完整示例

### 示例 1: CartPole with DQN

```python
"""
CartPole 環境使用 DQN 訓練
"""

import jax
import jax.numpy as jnp
from wann_sdk_core import *
from wann_sdk_ray_env import *
from wann_sdk_rl_methods import *

# 1. 架構搜索 (或加載已有架構)
# ... (architecture search code) ...

# 2. 加載架構
spec = ArchitectureSpec.load('./cartpole_arch.pkl')
architecture = WANNArchitecture(spec)

# 3. 創建環境
env = GymnasiumEnvWrapper("CartPole-v1")
env_info = env.get_env_info()

# 4. 創建 DQN 策略
config = TrainingConfig(
    learning_rate=1e-3,
    batch_size=128,
    buffer_size=10000
)

policy = create_policy_for_environment(
    architecture=architecture,
    env_info=env_info,
    method='dqn',
    config=config
)

# 5. 訓練循環
num_steps = 50000
obs, _ = env.reset()

for step in range(num_steps):
    # 選擇動作
    action = policy.select_action(obs)
    
    # 執行
    next_obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    
    # 存儲
    policy.store_transition(obs, action, reward, next_obs, done)
    
    # 更新
    if policy.ready_to_update():
        metrics = policy.update_step()
        
        if step % 1000 == 0:
            print(f"Step {step}: Loss={metrics['loss']:.4f}")
    
    # 重置
    if done:
        obs, _ = env.reset()
    else:
        obs = next_obs

# 6. 保存
final_params = policy.get_params()
# ... save params ...

print("Training completed!")
```

### 示例 2: BipedalWalker with PPO (分佈式)

```python
"""
BipedalWalker 使用 PPO 和分佈式訓練
"""

# 1. 創建分佈式環境池
pool = DistributedEnvPool(
    env_name="BipedalWalker-v3",
    num_workers=8
)

# 2. 加載架構
spec = ArchitectureSpec.load('./bipedal_arch.pkl')
architecture = WANNArchitecture(spec)

# 3. 創建 PPO 策略
env_info = pool.get_env_info()
config = TrainingConfig(
    learning_rate=3e-4,
    batch_size=256
)

policy = create_policy_for_environment(
    architecture=architecture,
    env_info=env_info,
    method='ppo',
    config=config
)

# 4. 訓練
num_iterations = 1000

for iteration in range(num_iterations):
    # 並行收集 trajectories
    def policy_fn(params, obs):
        return policy.select_action(obs, deterministic=False)
    
    rollouts = pool.collect_rollouts(
        policy_fn=policy_fn,
        policy_params=policy.get_params(),
        num_rollouts=32,
        max_steps=1000
    )
    
    # 存儲 trajectories
    for rollout in rollouts:
        for t in range(len(rollout['observations'])):
            policy.store_transition(
                rollout['observations'][t],
                rollout['actions'][t],
                rollout['rewards'][t],
                rollout['observations'][t+1] if t < len(rollout['observations'])-1 
                    else rollout['observations'][t],
                rollout['dones'][t]
            )
    
    # PPO 更新
    metrics = policy.update_step()
    
    # 記錄
    mean_reward = np.mean([r['episode_reward'] for r in rollouts])
    print(f"Iteration {iteration}: "
          f"Mean Reward={mean_reward:.2f}, "
          f"Loss={metrics['loss']:.4f}")

pool.close_all()
```

### 示例 3: 自定義環境和訓練方法

```python
"""
使用自定義環境和自定義訓練方法
"""

# 1. 創建自定義環境
import gymnasium as gym

class MyCustomEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(10,)
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(2,)
        )
    
    def reset(self, seed=None):
        # 你的實現
        obs = np.zeros(10)
        return obs, {}
    
    def step(self, action):
        # 你的實現
        obs = np.zeros(10)
        reward = 0.0
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {}

# 註冊環境
gym.register(
    id='MyCustomEnv-v0',
    entry_point=MyCustomEnv
)

# 2. 使用環境
env = GymnasiumEnvWrapper('MyCustomEnv-v0')

# 3. 使用你的自定義訓練方法
policy = create_policy_for_environment(
    architecture=architecture,
    env_info=env.get_env_info(),
    method='my_method',  # 你之前註冊的方法
    config=config
)

# 4. 訓練
# ... (training loop) ...
```

## 📚 API 參考

### ArchitectureSpec

```python
ArchitectureSpec(
    nodes: jnp.ndarray,
    connections: jnp.ndarray,
    num_inputs: int,
    num_outputs: int,
    num_hidden: int,
    num_params: int,
    search_fitness: float,
    search_complexity: float,
    activation_functions: Dict[int, str] = {},
    metadata: Dict[str, Any] = {}
)

# 方法
spec.save(path: str)
spec = ArchitectureSpec.load(path: str)
```

### WANNArchitecture

```python
WANNArchitecture(
    spec: ArchitectureSpec,
    genome: Optional[WANNGenome] = None
)

# 方法
params = architecture.init_params(key: PRNGKey)
output = architecture.forward(x: Array, params: Dict)
num_params = architecture.get_num_params()
info = architecture.get_architecture_info()
dict_data = architecture.to_dict()
architecture = WANNArchitecture.from_dict(dict_data)
```

### TrainingConfig

```python
TrainingConfig(
    learning_rate: float = 3e-4,
    batch_size: int = 256,
    buffer_size: int = 100000,
    num_epochs: int = 1000,
    eval_frequency: int = 10,
    eval_episodes: int = 10,
    checkpoint_frequency: int = 50,
    checkpoint_dir: str = "./checkpoints",
    log_frequency: int = 1,
    method_kwargs: Dict[str, Any] = {}
)
```

### GymnasiumEnvWrapper

```python
GymnasiumEnvWrapper(
    env_name: str,
    render_mode: Optional[str] = None,
    **env_kwargs
)

# 方法
obs, info = env.reset(seed: Optional[int])
obs, reward, terminated, truncated, info = env.step(action: Array)
info = env.get_env_info()
env.close()
```

### DistributedEnvPool

```python
DistributedEnvPool(
    env_name: str,
    num_workers: int = 4,
    **env_kwargs
)

# 方法
rollouts = pool.collect_rollouts(
    policy_fn: Callable,
    policy_params: Dict,
    num_rollouts: int,
    max_steps: int = 1000
)
pool.reset_all()
pool.close_all()
info = pool.get_env_info()
```

### TrainingMethodRegistry

```python
# 註冊新方法
@TrainingMethodRegistry.register('method_name')
class MyMethod(PolicyInterface):
    ...

# 獲取方法
method_cls = TrainingMethodRegistry.get('method_name')

# 列出所有方法
methods = TrainingMethodRegistry.list_methods()
```

## 🔍 故障排除

### 常見問題

#### 1. JAX OOM

```python
# 減少批次大小
config = TrainingConfig(batch_size=64)

# 或使用 CPU
import jax
jax.config.update('jax_platform_name', 'cpu')
```

#### 2. Ray 初始化錯誤

```python
# 確保 Ray 未初始化
import ray
if ray.is_initialized():
    ray.shutdown()

# 重新初始化
ray.init(ignore_reinit_error=True)
```

#### 3. 環境兼容性

```python
# 測試環境
from wann_sdk_ray_env import test_environment
test_environment("YourEnv-v0", num_episodes=3)
```

## 📝 最佳實踐

### 1. 架構搜索參數調優

```python
# 較小的種群用於快速實驗
WANN(pop_size=500, generations=50)

# 較大的種群用於最終搜索
WANN(pop_size=2000, generations=200)

# 調整複雜度權重
WANN(complexity_weight=0.3)  # 更偏好簡單網絡
```

### 2. 訓練穩定性

```python
# 使用學習率調度
import optax
schedule = optax.exponential_decay(
    init_value=3e-4,
    transition_steps=10000,
    decay_rate=0.99
)
optimizer = optax.adam(schedule)
```

### 3. 檢查點管理

```python
# 定期保存
config = TrainingConfig(
    checkpoint_frequency=10000,
    checkpoint_dir="./checkpoints"
)

# 從檢查點恢復
# ... (load checkpoint and resume) ...
```

## 🤝 貢獻指南

歡迎貢獻新的訓練方法、環境包裝器或改進！

### 添加新訓練方法

1. 實現 `PolicyInterface`
2. 使用 `@TrainingMethodRegistry.register` 註冊
3. 添加測試
4. 更新文檔

### 添加新環境

1. 繼承 `GymnasiumEnvWrapper`
2. 實現特定環境的預處理
3. 添加測試
4. 更新文檔

## 📄 許可證

本 SDK 遵循以下許可證:
- WANN: MIT License
- TensorNEAT: GPL-3.0
- EvoX: GPL-3.0

## 🙏 致謝

- Weight Agnostic Neural Networks (Gaier & Ha)
- TensorNEAT (EMI-Group)
- EvoX (EMI-Group)
- Gymnasium (Farama Foundation)
- Ray (Anyscale)

---

**Happy Training! 🚀**

如有問題,請參考示例代碼或提交 Issue。
