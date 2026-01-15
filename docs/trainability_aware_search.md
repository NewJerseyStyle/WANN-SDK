# Trainability-Aware Architecture Search

This guide explains how to use zero-cost proxies (ZCPs) to find neural network architectures that are not only weight-agnostic but also **trainable** - meaning they can be effectively optimized through gradient-based weight training in Stage 2.

## Overview

Standard WANN search finds architectures that perform well with shared weights across all connections. However, these architectures may not always be easy to train when individual weights are optimized.

**Trainability-aware search** extends WANN by incorporating zero-cost proxies that predict how well an architecture will respond to gradient-based training, without actually performing expensive training.

```
┌─────────────────────────────────────────────────────────────┐
│                    Two-Stage Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│  Stage 1: Architecture Search                               │
│  ┌─────────────────┐      ┌─────────────────┐               │
│  │  WANN Fitness   │  +   │  ZCP Score      │  → Combined   │
│  │  (shared weight)│      │  (trainability) │    Fitness    │
│  └─────────────────┘      └─────────────────┘               │
├─────────────────────────────────────────────────────────────┤
│  Stage 2: Weight Training                                   │
│  ┌─────────────────────────────────────────┐                │
│  │  Gradient-based optimization (Adam/SGD) │                │
│  │  on architecture found in Stage 1       │                │
│  └─────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```python
from wann_sdk import (
    TrainabilityAwareSearch,
    SearchConfig,
    SupervisedProblem,
    WeightTrainer,
    WeightTrainerConfig,
)

# Define problem
problem = SupervisedProblem(x_train, y_train, loss_fn='cross_entropy')

# Stage 1: Trainability-aware architecture search
search = TrainabilityAwareSearch(
    problem,
    SearchConfig(max_nodes=20, pop_size=50),
    strategy='hybrid',
    zcp_weight=0.3,  # 30% ZCP, 70% WANN fitness
    zcp_proxies=['synflow', 'naswot', 'trainability'],
)
genome = search.run(generations=100)

# Stage 2: Weight training (should converge faster!)
trainer = WeightTrainer(genome, problem, WeightTrainerConfig(optimizer='adam'))
trainer.fit(epochs=100)
```

## Integration Strategies

Three strategies are available for combining WANN fitness with ZCP scores:

### 1. Hybrid Strategy (Recommended)

Combines WANN and ZCP scores into a single fitness value at every generation.

```python
search = TrainabilityAwareSearch(
    problem,
    config,
    strategy='hybrid',
    zcp_weight=0.3,  # Balance between WANN (0.7) and ZCP (0.3)
)
```

**Combined fitness formula:**
```
fitness = (1 - zcp_weight) × WANN_score + zcp_weight × ZCP_score
```

**When to use:** General-purpose, good balance between exploration and trainability.

### 2. Sequential Strategy

First filters by WANN fitness, then refines top candidates using ZCP.

```python
from wann_sdk import TrainabilitySearchConfig

search = TrainabilityAwareSearch(
    problem,
    config,
    trainability_config=TrainabilitySearchConfig(
        strategy='sequential',
        filter_ratio=0.3,  # Compute ZCP for top 30% only
        zcp_weight=0.3,
    ),
)
```

**Process:**
1. Evaluate all genomes with WANN fitness
2. Select top N% candidates
3. Compute ZCP scores for top candidates only
4. Re-rank using combined score

**When to use:** Large populations where computing ZCP for all is expensive.

### 3. Parallel Strategy (Multi-Objective)

Treats WANN and ZCP as separate objectives for Pareto optimization.

```python
search = TrainabilityAwareSearch(
    problem,
    config,
    strategy='parallel',
)
```

**When to use:** When you want to explore the Pareto front of weight-agnostic vs trainable architectures.

## Zero-Cost Proxies

### Available Proxies

| Proxy | Type | Description | Data Required |
|-------|------|-------------|---------------|
| `synflow` | Gradient | Parameter importance via gradient flow | No |
| `naswot` | Activation | Activation pattern diversity | Yes |
| `snip` | Gradient | Single-shot pruning saliency | Yes + Labels |
| `trainability` | Gradient | Gradient flow stability | Yes + Labels |
| `expressivity` | Activation | Output diversity measure | Yes |
| `fisher` | Gradient | Fisher information approximation | Yes + Labels |
| `grasp` | Gradient | Gradient signal preservation | Yes + Labels |

### Proxy Descriptions

#### Synflow (Data-Agnostic)
Measures the "synaptic flow" through the network without requiring any data. Higher scores indicate better gradient propagation potential.

```python
from wann_sdk import compute_synflow

score = compute_synflow(forward_fn, params, input_shape=(784,))
```

#### NASWOT (Neural Architecture Search Without Training)
Evaluates architecture expressivity by measuring the diversity of activation patterns across different inputs. Networks with more diverse patterns have higher representational capacity.

```python
from wann_sdk import compute_naswot

score = compute_naswot(forward_fn, params, x_batch)
```

#### Trainability (from AZ-NAS)
Measures gradient flow stability - whether gradients are neither vanishing nor exploding. Optimal architectures have moderate gradient magnitudes with low variance.

```python
from wann_sdk import compute_trainability

score = compute_trainability(forward_fn, params, x_batch, y_batch, loss_fn)
```

#### SNIP (Single-shot Network Pruning)
Evaluates parameter importance by measuring the sensitivity of the loss to each parameter. Higher scores indicate parameters that matter more for the task.

```python
from wann_sdk import compute_snip

score = compute_snip(forward_fn, params, x_batch, y_batch, loss_fn)
```

### Recommended Proxy Combinations

| Use Case | Recommended Proxies | Rationale |
|----------|---------------------|-----------|
| **General** | `synflow`, `naswot`, `trainability` | Balanced coverage |
| **Fast search** | `synflow`, `naswot` | No label dependency |
| **Gradient focus** | `synflow`, `trainability`, `snip` | Emphasize gradient flow |
| **Expressivity focus** | `naswot`, `expressivity` | Emphasize representational capacity |

## ZCP Evaluator API

For standalone ZCP evaluation:

```python
from wann_sdk import ZCPEvaluator, ZCPConfig

# Create evaluator
evaluator = ZCPEvaluator(
    proxies=['synflow', 'naswot', 'trainability'],
    aggregation='geometric',  # or 'mean', 'weighted', 'harmonic'
    normalize=True,
)

# Evaluate a network
scores = evaluator.evaluate(
    forward_fn=lambda params, x: network.forward(params, x),
    params=network.get_params(),
    x_batch=x_train[:32],
    y_batch=y_train[:32],
    loss_fn=lambda pred, target: jnp.mean((pred - target) ** 2),
)

print(f"Synflow: {scores['synflow']:.4f}")
print(f"NASWOT: {scores['naswot']:.4f}")
print(f"Trainability: {scores['trainability']:.4f}")
print(f"Aggregated: {scores['aggregated']:.4f}")
```

### Aggregation Methods

| Method | Formula | Best For |
|--------|---------|----------|
| `mean` | `(s1 + s2 + ...) / n` | Equal weighting |
| `geometric` | `(s1 × s2 × ...)^(1/n)` | Penalize low scores |
| `weighted` | `Σ(wi × si) / Σwi` | Custom importance |
| `harmonic` | `n / Σ(1/si)` | Emphasize worst proxy |

## Configuration Reference

### TrainabilitySearchConfig

```python
from wann_sdk import TrainabilitySearchConfig

config = TrainabilitySearchConfig(
    # Integration strategy
    strategy='hybrid',           # 'hybrid', 'sequential', 'parallel'

    # Proxy selection
    zcp_proxies=['synflow', 'naswot', 'trainability'],

    # Weight balancing
    zcp_weight=0.3,              # Weight for ZCP (0-1)
    wann_weight=None,            # Auto-computed as 1 - zcp_weight

    # Dynamic weighting
    dynamic_weight=False,        # Adjust weights during evolution

    # Evaluation settings
    zcp_batch_size=32,           # Batch size for ZCP computation

    # Sequential strategy settings
    filter_ratio=0.3,            # Keep top 30% for ZCP (sequential only)
)
```

### ZCPConfig

```python
from wann_sdk import ZCPConfig

config = ZCPConfig(
    proxies=['synflow', 'naswot', 'trainability'],
    aggregation='geometric',
    weights={'synflow': 1.0, 'naswot': 1.0, 'trainability': 1.5},
    normalize=True,
    batch_size=32,
)
```

## Complete Example

```python
"""
Compare standard WANN vs trainability-aware search,
then verify with Stage 2 weight training.
"""
import jax.numpy as jnp
from wann_sdk import (
    ArchitectureSearch,
    TrainabilityAwareSearch,
    SearchConfig,
    SupervisedProblem,
    WeightTrainer,
    WeightTrainerConfig,
)

# Prepare data
x_train, y_train = load_data()
problem = SupervisedProblem(x_train, y_train, loss_fn='cross_entropy')

config = SearchConfig(
    pop_size=50,
    max_nodes=20,
    activation_options=['tanh', 'relu', 'sigmoid'],
)

# === Standard WANN Search ===
wann_search = ArchitectureSearch(problem, config)
wann_genome = wann_search.run(generations=50)
print(f"WANN fitness: {wann_genome.fitness:.4f}")

# === Trainability-Aware Search ===
trainable_search = TrainabilityAwareSearch(
    problem, config,
    strategy='hybrid',
    zcp_weight=0.3,
    zcp_proxies=['synflow', 'naswot', 'trainability'],
)
trainable_genome = trainable_search.run(generations=50)
print(f"Trainability fitness: {trainable_genome.fitness:.4f}")

# Get ZCP breakdown
zcp_scores = trainable_search.get_zcp_breakdown(trainable_genome)
print(f"ZCP breakdown: {zcp_scores}")

# === Stage 2: Weight Training Comparison ===
trainer_config = WeightTrainerConfig(
    optimizer='adam',
    learning_rate=0.01,
)

# Train WANN architecture
wann_trainer = WeightTrainer(wann_genome, problem, trainer_config)
wann_result = wann_trainer.fit(epochs=100)
print(f"WANN final loss: {wann_result['history'][-1]['loss']:.4f}")

# Train trainability-aware architecture
trainable_trainer = WeightTrainer(trainable_genome, problem, trainer_config)
trainable_result = trainable_trainer.fit(epochs=100)
print(f"Trainability final loss: {trainable_result['history'][-1]['loss']:.4f}")

# Expected: trainability-aware architecture converges faster/better
```

## Tips and Best Practices

### Choosing ZCP Weight

| zcp_weight | Behavior |
|------------|----------|
| 0.0 | Pure WANN (no trainability consideration) |
| 0.1-0.2 | Light trainability bias |
| 0.3-0.4 | Balanced (recommended starting point) |
| 0.5+ | Strong trainability focus |

### Performance Optimization

1. **Use sequential strategy for large populations** - Only computes ZCP for top candidates
2. **Start with `synflow` + `naswot`** - Fast, no label dependency
3. **Add `trainability` for gradient-focused tasks** - Requires labels but predicts training behavior
4. **Cache ZCP scores** - The search automatically caches scores for seen genomes

### Common Issues

| Issue | Solution |
|-------|----------|
| ZCP computation slow | Reduce `zcp_batch_size` or use sequential strategy |
| Trainability scores all similar | Increase network complexity (`max_nodes`) |
| Poor weight training convergence | Increase `zcp_weight` to prioritize trainability |

## References

- **Synflow**: Tanaka et al., "Pruning Neural Networks at Initialization" (NeurIPS 2020)
- **NASWOT**: Mellor et al., "Neural Architecture Search without Training" (ICML 2021)
- **SNIP**: Lee et al., "SNIP: Single-shot Network Pruning" (ICLR 2019)
- **AZ-NAS**: Inspired by training-free NAS methods combining multiple proxies
