# WANN SDK Optimizers

This guide covers the unified optimizer system in WANN SDK, which supports both gradient-based and evolutionary optimization algorithms for Stage 2 weight training.

## Overview

WANN SDK provides a unified interface for optimizers from multiple sources:

| Source | Type | Examples | Dependencies |
|--------|------|----------|--------------|
| **Built-in** | Evolutionary | ES, OpenES, PEPG | None |
| **Optax** | Gradient | Adam, AdamW, SGD, RMSProp, Lion, Lamb | optax (included) |
| **Nevergrad** | Evolutionary | CMA-ES, DE, PSO, NGOpt | `pip install nevergrad` |

## Quick Start

### String-Based (Simple)

```python
from wann_sdk import WeightTrainer, WeightTrainerConfig

# Use optimizer by name
config = WeightTrainerConfig(
    optimizer='adam',
    learning_rate=0.001,
)
trainer = WeightTrainer(genome, problem, config)
trainer.fit(epochs=100)
```

### Class-Based (IDE Support)

```python
from wann_sdk import WeightTrainer, WeightTrainerConfig
from wann_sdk.optimizers import Adam, CMA, LBFGS

# Full IDE autocomplete and docstring support
opt = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
config = WeightTrainerConfig(optimizer=opt)

# Or with evolutionary optimizer
opt = CMA(population_size=32)
config = WeightTrainerConfig(optimizer=opt)
```

### Using optimizer_kwargs

```python
# Pass extra parameters via optimizer_kwargs
config = WeightTrainerConfig(
    optimizer='cma',
    optimizer_kwargs={
        'population_size': 64,
        'sigma': 0.5,
    }
)
```

## Available Optimizers

### Gradient-Based Optimizers

Use these when you can compute gradients through your network.

#### Adam
```python
from wann_sdk.optimizers import Adam

opt = Adam(
    learning_rate=0.001,  # Step size
    beta1=0.9,            # First moment decay
    beta2=0.999,          # Second moment decay
    eps=1e-8,             # Numerical stability
)
```
**Best for:** General deep learning, default choice.

#### AdamW
```python
from wann_sdk.optimizers import AdamW

opt = AdamW(
    learning_rate=0.001,
    weight_decay=0.01,    # Decoupled weight decay
    beta1=0.9,
    beta2=0.999,
)
```
**Best for:** When you need L2 regularization.

#### SGD
```python
from wann_sdk.optimizers import SGD

opt = SGD(
    learning_rate=0.01,
    momentum=0.9,         # Momentum coefficient
    nesterov=False,       # Use Nesterov momentum
)
```
**Best for:** Convex problems, when you want simplicity.

#### RMSProp
```python
from wann_sdk.optimizers import RMSProp

opt = RMSProp(
    learning_rate=0.001,
    decay=0.9,            # Moving average decay
    eps=1e-8,
)
```
**Best for:** Non-stationary objectives, RNNs.

#### L-BFGS
```python
from wann_sdk.optimizers import LBFGS

opt = LBFGS(
    learning_rate=1.0,
    max_iterations=20,    # Inner iterations per step
    history_size=10,      # Number of past gradients to store
    tolerance=1e-5,
)
```
**Best for:** Small networks, when you can afford full-batch gradients.

#### Lion
```python
from wann_sdk.optimizers import Lion

opt = Lion(
    learning_rate=0.0001,
    beta1=0.9,            # Momentum decay
    beta2=0.99,           # Velocity decay
    weight_decay=0.0,
)
```
**Best for:** Memory-efficient alternative to Adam, discovered through program search.

#### Lamb
```python
from wann_sdk.optimizers import Lamb

opt = Lamb(
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    eps=1e-6,
    weight_decay=0.0,
)
```
**Best for:** Large batch training with layer-wise learning rate adaptation.

### Evolutionary Optimizers

Use these when gradients are unavailable, unreliable, or when you want to escape local minima.

#### ES (Built-in)
```python
from wann_sdk.optimizers import ES

opt = ES(
    population_size=64,   # Number of perturbation pairs
    learning_rate=0.01,   # Update step size
    noise_std=0.1,        # Perturbation magnitude
    weight_decay=0.0,     # L2 regularization
)
```
**Best for:** Default evolutionary optimizer, no dependencies.

#### CMA-ES (Nevergrad)
```python
from wann_sdk.optimizers import CMA

opt = CMA(
    population_size=64,
    sigma=0.5,            # Initial step size
)
```
**Best for:** State-of-the-art evolutionary optimization, 10-1000 parameters.

#### Differential Evolution (Nevergrad)
```python
from wann_sdk.optimizers import DE

opt = DE(
    population_size=64,
    cr=0.5,               # Crossover probability
    f=0.8,                # Differential weight
)
```
**Best for:** Global optimization, many local minima.

#### PSO (Nevergrad)
```python
from wann_sdk.optimizers import PSO

opt = PSO(
    population_size=64,
    omega=0.7,            # Inertia weight
    phip=1.5,             # Personal best attraction
    phig=1.5,             # Global best attraction
)
```
**Best for:** Fast initial progress, clustered solutions.

#### NGOpt (Nevergrad Auto-Select)
```python
from wann_sdk.optimizers import NGOpt

opt = NGOpt(population_size=64)
```
**Best for:** When you don't know which optimizer to use.

## Registry Functions

### List All Optimizers

```python
from wann_sdk import list_optimizers

# All optimizers
all_opts = list_optimizers()
print(all_opts)
# {'adam': 'Adam optimizer...', 'cma': 'CMA-ES...', ...}

# Filter by category
gradient_opts = list_optimizers(category='gradient')
evolutionary_opts = list_optimizers(category='evolutionary')
```

### Get Optimizer by Name

```python
from wann_sdk import get_optimizer

AdamClass = get_optimizer('adam')
opt = AdamClass(learning_rate=0.001)
```

### Register Custom Optimizer

```python
from wann_sdk import register_optimizer
from wann_sdk.optimizers import BaseOptimizer, OptimizerState

class MyOptimizer(BaseOptimizer):
    """My custom optimizer."""
    name = "my-opt"
    is_gradient_based = True

    def __init__(self, learning_rate=0.01, my_param=1.0):
        super().__init__(learning_rate=learning_rate, my_param=my_param)
        self.learning_rate = learning_rate
        self.my_param = my_param

    def init_state(self, params):
        return OptimizerState(step=0, params=params, internal=None)

    def update(self, state, grads=None, **kwargs):
        new_params = state.params - self.learning_rate * grads * self.my_param
        return OptimizerState(
            step=state.step + 1,
            params=new_params,
            internal=None,
        )

# Register it
register_optimizer("my-opt", MyOptimizer)

# Now use it
config = WeightTrainerConfig(optimizer='my-opt')
```

## Optimizer Selection Guide

### By Problem Type

| Problem | Recommended | Why |
|---------|-------------|-----|
| Supervised learning | Adam, AdamW | Fast convergence, good defaults |
| Small network (<100 params) | L-BFGS | Second-order, fast convergence |
| RL / non-differentiable | ES, CMA | No gradient required |
| Global optimization | CMA, DE | Escape local minima |
| Unknown problem | NGOpt | Auto-selects best |

### By Parameter Count

| Parameters | Gradient | Evolutionary |
|------------|----------|--------------|
| < 100 | L-BFGS | OnePlusOne |
| 100 - 1000 | Adam | CMA |
| 1000 - 10000 | Adam, AdamW | DiagonalCMA |
| > 10000 | Adam, AdamW | ES, NGOpt |

### By Compute Budget

| Budget | Gradient | Evolutionary |
|--------|----------|--------------|
| Low (< 1000 evals) | Adam | OnePlusOne |
| Medium | Adam, L-BFGS | CMA, DE |
| High (> 10000 evals) | Any | CMA, NGOpt |

## Advanced Usage

### Combining Gradient and Evolutionary

```python
# Start with evolutionary to find good region
config1 = WeightTrainerConfig(optimizer='cma')
trainer1 = WeightTrainer(genome, problem, config1)
trainer1.fit(epochs=50)

# Fine-tune with gradient-based
config2 = WeightTrainerConfig(optimizer='adam', learning_rate=0.0001)
trainer2 = WeightTrainer(genome, problem, config2)
trainer2.network.set_params(trainer1.get_weights())  # Transfer weights
trainer2.fit(epochs=100)
```

### Ask-Tell Interface (Evolutionary)

For custom fitness evaluation:

```python
from wann_sdk.optimizers import CMA
import jax

opt = CMA(population_size=32)
state = opt.init_state(initial_params)

for generation in range(100):
    key = jax.random.PRNGKey(generation)

    # Get candidate solutions
    candidates, ask_state = opt.ask(state, key)

    # Evaluate fitness (your custom evaluation)
    fitnesses = jnp.array([evaluate(c) for c in candidates])

    # Update optimizer
    state = opt.tell(state, ask_state, fitnesses)

    print(f"Gen {generation}: Best fitness = {fitnesses.max()}")

final_params = state.params
```

## Backward Compatibility

Legacy configurations still work:

```python
# Old style (still supported)
config = WeightTrainerConfig(
    optimizer='es',
    pop_size=64,        # Maps to population_size
    noise_std=0.1,
)

# Old style for Adam
config = WeightTrainerConfig(
    optimizer='adamw',
    learning_rate=0.001,
    weight_decay=0.01,
    beta1=0.9,
    beta2=0.999,
)
```

## Dependencies

| Optimizer | Dependency | Install |
|-----------|------------|---------|
| Adam, AdamW, SGD, RMSProp, AdaGrad, Lion, Lamb | optax | Included in WANN SDK |
| ES, OpenES, PEPG | None | Built-in |
| CMA, DE, PSO, NGOpt | nevergrad | `pip install nevergrad` |
| L-BFGS (full second-order) | jaxopt | `pip install jaxopt` |

Optimizers gracefully degrade if dependencies are missing - you'll get a helpful error message suggesting the install command.

## Technical Notes

### Optax Integration

The gradient-based optimizers use [Optax](https://github.com/deepmind/optax), DeepMind's gradient transformation library for JAX. Optax is the standard choice for gradient optimization in the JAX ecosystem and provides:

- Composable gradient transformations
- Efficient state management
- Wide range of well-tested algorithms

### Nevergrad Integration

Evolutionary optimizers wrap [Nevergrad](https://github.com/facebookresearch/nevergrad), Facebook's gradient-free optimization library. Benefits include:

- State-of-the-art evolutionary algorithms
- Automatic algorithm selection (NGOpt)
- Handles non-differentiable objectives
