# SDK API Reference

**Weight Agnostic Neural Networks SDK** - A JAX-based toolkit implementing the full WANN two-stage pipeline for architecture search and weight training.

## Two-Stage Pipeline

WANN SDK implements the standard Weight Agnostic Neural Network methodology:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 1: Architecture Search                                       │
│  ────────────────────────────────────────────────────────────────── │
│  • Evolve network topology (nodes, connections, activations)        │
│  • Evaluate with SHARED weights across all connections              │
│  • Find architectures that work regardless of weight value          │
│  • Uses NEAT-like neuroevolution                                    │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 2: Weight Training                                           │
│  ────────────────────────────────────────────────────────────────── │
│  • Train INDIVIDUAL weights on found architecture                   │
│  • Supports: ES, SGD, Adam, AdamW optimizers                        │
│  • Export to PyTorch for downstream fine-tuning                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Stage 1: Architecture Search

### SearchConfig

Configure the architecture search space:

```python
SearchConfig(
    pop_size=100,              # Population size
    max_nodes=50,              # Max hidden nodes to evolve
    max_connections=200,       # Max connections to evolve
    activation_options=[       # Activations to search over
        'tanh', 'relu', 'sigmoid', 'sin', 'abs', 'square'
    ],
    weight_values=[            # Shared weights for evaluation
        -2.0, -1.0, -0.5, 0.5, 1.0, 2.0
    ],
    complexity_weight=0.1,     # Penalty for network size
    add_node_rate=0.03,        # Mutation: add node
    add_connection_rate=0.05,  # Mutation: add connection
)
```

### ArchitectureSearch

```python
search = ArchitectureSearch(problem, config)

# Run search
genome = search.run(generations=100, log_interval=10)

# Get network with specific weight
network = search.get_best_network(weight=1.0)

# Convert any genome to network
network = search.genome_to_network(genome, weight=0.5)
```

## Stage 2: Weight Training

### WeightTrainerConfig

Configure weight training:

```python
WeightTrainerConfig(
    optimizer='adamw',        # 'es', 'sgd', 'adam', 'adamw'
    learning_rate=0.001,

    # ES-specific
    pop_size=64,
    noise_std=0.1,

    # Adam/AdamW-specific
    weight_decay=0.01,
    beta1=0.9,
    beta2=0.999,
)
```

### WeightTrainer

```python
trainer = WeightTrainer(genome, problem, config)

# Train
trainer.fit(epochs=100, log_interval=10)

# Get trained network
network = trainer.get_network()
weights = trainer.get_weights()

# Save/Load
trainer.save('model.pkl')
trainer = WeightTrainer.load('model.pkl', problem)
```

## Problem Definition

### Custom Problem

```python
from wann_sdk import Problem
import jax.numpy as jnp

class MyProblem(Problem):
    def __init__(self):
        super().__init__(input_dim=10, output_dim=2)
        self.x, self.y = load_data()

    def evaluate(self, network, key) -> float:
        """For ES optimizer - returns Python float."""
        output = network(self.x)
        loss = jnp.mean((output - self.y) ** 2)
        return -float(loss)  # Negative loss = fitness

    def loss(self, network, key) -> jnp.ndarray:
        """For gradient-based optimizers (SGD/Adam/AdamW) - returns JAX array."""
        output = network(self.x)
        return jnp.mean((output - self.y) ** 2)  # NO float()!
```

> **Important**:
> - `evaluate()` returns `float` - used by ES and architecture search
> - `loss()` returns `jnp.ndarray` - used by gradient-based optimizers (SGD, Adam, AdamW)
> - `SupervisedProblem` implements both automatically

### SupervisedProblem

```python
from wann_sdk import SupervisedProblem

problem = SupervisedProblem(
    x_train, y_train,
    x_val=x_test, y_val=y_test,
    loss_fn='cross_entropy',  # 'mse', 'cross_entropy', 'binary_cross_entropy'
    batch_size=256,
)
```

### RLProblem (Brax)

```python
from wann_sdk import RLProblem, BraxEnv

env = BraxEnv("ant")
problem = RLProblem(env, max_steps=1000)
```

### GymnaxProblem (Gymnax Classic Control)

```python
from wann_sdk import GymnaxProblem, GymnaxEnv

env = GymnaxEnv("CartPole-v1")
problem = GymnaxProblem(env, max_steps=500)
```

## Export

### Export to PyTorch

```python
from wann_sdk import export_to_pytorch

export_to_pytorch(
    genome=genome,
    weights=trainer.get_weights(),
    activation_options=['tanh', 'relu', 'sigmoid'],
    output_path='wann_model.py',
)
```

Generated PyTorch model can be fine-tuned:

```python
from wann_model import WANNModel
import torch

model = WANNModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    output = model(x_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
```

### Export to ONNX

```python
from wann_sdk import export_to_onnx

export_to_onnx(genome, weights, activation_options, 'model.onnx')
```

# Examples

## XOR Problem

```python
from wann_sdk import Problem, ArchitectureSearch, SearchConfig, WeightTrainer, WeightTrainerConfig
import jax
import jax.numpy as jnp

class XORProblem(Problem):
    def __init__(self):
        super().__init__(input_dim=2, output_dim=1)
        self.x = jnp.array([[0,0], [0,1], [1,0], [1,1]], dtype=jnp.float32)
        self.y = jnp.array([[0], [1], [1], [0]], dtype=jnp.float32)

    def evaluate(self, network, key):
        """For ES and architecture search - returns float."""
        pred = jax.nn.sigmoid(network(self.x))
        return -float(jnp.mean((pred - self.y)**2))

    def loss(self, network, key):
        """For gradient-based optimizers - returns JAX array."""
        pred = jax.nn.sigmoid(network(self.x))
        return jnp.mean((pred - self.y)**2)  # NO float()!

problem = XORProblem()

# Stage 1: Architecture Search
search = ArchitectureSearch(problem, SearchConfig(max_nodes=10))
genome = search.run(generations=100)

# Stage 2: Weight Training
# Option A: Use Adam (requires loss() method)
trainer = WeightTrainer(genome, problem, WeightTrainerConfig(optimizer='adam'))

# Option B: Use ES (only requires evaluate() method)
# trainer = WeightTrainer(genome, problem, WeightTrainerConfig(optimizer='es'))

trainer.fit(epochs=100)
```

> **Note**: If you only implement `evaluate()`, use `optimizer='es'` for Stage 2.
> Gradient-based optimizers (sgd/adam/adamw) require the `loss()` method.

## MNIST Classification

```bash
python -m examples.wann_pipeline --task mnist --optimizer adamw
```

```python
from wann_sdk import ArchitectureSearch, SearchConfig, WeightTrainer, WeightTrainerConfig, SupervisedProblem

# Load data
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784) / 255.0

problem = SupervisedProblem(x_train, y_train, loss_fn='cross_entropy', batch_size=256)

# Stage 1: Find architecture
search = ArchitectureSearch(problem, SearchConfig(
    max_nodes=1000,
    activation_options=['tanh', 'relu'],
))
genome = search.run(generations=50)

# Stage 2: Train weights with AdamW
trainer = WeightTrainer(genome, problem, WeightTrainerConfig(
    optimizer='adamw',
    learning_rate=0.001,
))
trainer.fit(epochs=100)
```

## RL: Ant Locomotion (Brax)

```python
from wann_sdk import ArchitectureSearch, SearchConfig, WeightTrainer, WeightTrainerConfig
from wann_sdk import RLProblem, BraxEnv

env = BraxEnv("ant")
problem = RLProblem(env, max_steps=1000)

# Stage 1
search = ArchitectureSearch(problem, SearchConfig(max_nodes=20))
genome = search.run(generations=100)

# Stage 2 (ES works better for RL)
trainer = WeightTrainer(genome, problem, WeightTrainerConfig(
    optimizer='es',
    learning_rate=0.02,
    pop_size=64,
))
trainer.fit(epochs=200)
```

## RL: CartPole (Gymnax)

```bash
pip install gymnax
python -m examples.train_gymnax --env cartpole
```

```python
from wann_sdk import ArchitectureSearch, SearchConfig, WeightTrainer, WeightTrainerConfig
from wann_sdk import GymnaxProblem, GymnaxEnv

# Classic control environment (JAX-native)
env = GymnaxEnv("CartPole-v1")
problem = GymnaxProblem(env, max_steps=500, num_rollouts=3)

# Stage 1: Architecture Search
search = ArchitectureSearch(problem, SearchConfig(
    max_nodes=15,
    activation_options=['tanh', 'relu', 'sigmoid'],
))
genome = search.run(generations=50)

# Stage 2: Weight Training with ES
trainer = WeightTrainer(genome, problem, WeightTrainerConfig(
    optimizer='es',
    learning_rate=0.05,
))
trainer.fit(epochs=50)
```

# Available Activations

The architecture search can discover networks using these activations:

| Activation | Function |
|------------|----------|
| `tanh` | Hyperbolic tangent |
| `relu` | Rectified linear unit |
| `sigmoid` | Sigmoid |
| `sin` | Sine |
| `cos` | Cosine |
| `abs` | Absolute value |
| `square` | x² |
| `identity` | x |
| `step` | Step function |
| `gaussian` | exp(-x²) |

# Available Environments

## Brax Environments (Physics Simulation)

| Environment | Obs Dim | Action Dim | Description |
|------------|---------|------------|-------------|
| `humanoid` | 244 | 17 | Humanoid locomotion |
| `ant` | 87 | 8 | Ant locomotion |
| `halfcheetah` | 18 | 6 | Half cheetah locomotion |
| `hopper` | 11 | 3 | Hopper locomotion |
| `walker2d` | 17 | 6 | Walker 2D locomotion |
| `inverted_pendulum` | 4 | 1 | Inverted pendulum balance |

## Gymnax Environments (Classic Control)

| Environment | Obs Dim | Action Dim | Type |
|------------|---------|------------|------|
| `CartPole-v1` | 4 | 2 | Discrete |
| `Pendulum-v1` | 3 | 1 | Continuous |
| `MountainCar-v0` | 2 | 3 | Discrete |
| `MountainCarContinuous-v0` | 2 | 1 | Continuous |
| `Acrobot-v1` | 6 | 3 | Discrete |
| `Asterix-MinAtar` | 100 | 5 | Discrete |
| `Breakout-MinAtar` | 100 | 3 | Discrete |
| `SpaceInvaders-MinAtar` | 100 | 4 | Discrete |

## Performance Tips

1. **Stage 1 (Architecture Search)**:
   - Use fewer `weight_values` (e.g., `[-1, 0.5, 1]`) for faster search
   - Increase `complexity_weight` to prefer simpler networks
   - Start with smaller `max_nodes` and increase if needed

2. **Stage 2 (Weight Training)**:
   - Use `adamw` for supervised learning
   - Use `es` for RL problems
   - Adjust `learning_rate` based on loss magnitude

3. **GPU Acceleration**:
   ```python
   import jax
   print(jax.devices())  # Check GPU availability
   ```