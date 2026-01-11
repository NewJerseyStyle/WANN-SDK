# WANN SDK

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/backend-JAX-orange.svg)](https://github.com/google/jax)
[![TensorNEAT](https://img.shields.io/badge/TensorNEAT-orange.svg)](https://github.com/EMI-Group/tensorneat)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**WANN SDK** is a high-performance framework for evolving Artificial Neural Networks with **Weight Agnostic Neural Networks (WANN)** method. Built on top of [TensorNEAT](https://github.com/EMI-Group/tensorneat), this toolkit provides a streamlined API for architecture search and weight optimization, leveraging JAX for massive parallelism.

This project originated as a fork of a 2019 research collaboration with [Arthur](https://github.com/rlalpha), modernized to support modern hardware acceleration and the Brax physics engine.

## 🧠 What is WANN?

Unlike traditional neural networks where the focus is on optimizing specific weight values, and traditional architecture search requires weight optimization during training. Our experimental method was using shared weight to archtiecture search and optimize weight after fixing the network **topology**. We adapt **Weight Agnostic Neural Networks** to find architectures that can perform tasks without optimize weight to compare and find the **best topology**.

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

## ✨ Key Features

-   **TensorNEAT Integration:** Fully vectorized evolutionary operations.
-   **JAX Accelerated:** Designed for speed on CPUs, GPUs, and TPUs.
-   **Two-Stage Pipeline:** Decoupled architecture search and weight optimization.

## 🚀 Getting Started
### Installation

```bash
pip install git+https://github.dev/NewJerseyStyle/WANN-SDK
# not uploaded to pypi yet
# pip install wann-sdk
```

### Optional Dependencies

```bash
pip install wann-sdk[brax]      # For Brax physics environments
pip install wann-sdk[gymnax]    # For Gymnax classic control
pip install wann-sdk[vision]    # For image datasets
pip install wann-sdk[full]      # Everything
```

### Full Pipeline Example

```python
from wann_sdk import (
    # Stage 1
    ArchitectureSearch, SearchConfig,
    # Stage 2
    WeightTrainer, WeightTrainerConfig,
    # Problem
    SupervisedProblem,
    # Export
    export_to_pytorch,
)

# Define your problem
problem = SupervisedProblem(x_train, y_train, loss_fn='cross_entropy')

# ============================================
# Stage 1: Architecture Search
# ============================================
search = ArchitectureSearch(
    problem,
    SearchConfig(
        max_nodes=30,                              # Search space
        max_connections=100,
        activation_options=['tanh', 'relu', 'sigmoid'],
        weight_values=[-2, -1, -0.5, 0.5, 1, 2],  # Shared weights for eval
    )
)
genome = search.run(generations=100)

# ============================================
# Stage 2: Weight Training
# ============================================
trainer = WeightTrainer(
    genome, problem,
    WeightTrainerConfig(
        optimizer='adamw',       # 'es', 'sgd', 'adam', 'adamw'
        learning_rate=0.001,
        weight_decay=0.01,
    )
)
trainer.fit(epochs=100)

# Get trained network
network = trainer.get_network()
predictions = network(x_test)

# ============================================
# Export to PyTorch (Optional)
# ============================================
export_to_pytorch(genome, trainer.get_weights(), 'wann_model.py')
```

## API Reference
- [Detailed API reference](/APIs.md)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Related Work

- For the original 2019 research collaboration with Arthur (2019).
- [Weight Agnostic Neural Networks](https://weightagnostic.github.io/) (Gaier & Ha, 2019)
- [TensorNEAT](https://github.com/EMI-Group/tensorneat) - NEAT algorithms in JAX
- [Brax](https://github.com/google/brax) - Physics simulation in JAX
- [Gymnax](https://github.com/RobertTLange/gymnax) - Classic RL environments in JAX

## 📋 Citing WANN SDK, WANN, TensorNEAT
If you use this SDK in your research, we recommend you to cite both this repository and the related works.
```
@software{wann_sdk_2025,
  author = {David Xu and Arthur Yau},
  title = {WANN SDK: A JAX-based Framework for Weight Agnostic Neural Networks},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/NewJerseyStyle/WANN-SDK}},
  version = {0.1.0}
}

@article{wann2019,
  author = {Adam Gaier and David Ha},
  title  = {Weight Agnostic Neural Networks},
  eprint = {arXiv:1906.04358},
  url    = {https://weightagnostic.github.io},
  note   = "\url{https://weightagnostic.github.io}",
  year   = {2019}
}

@article{10.1145/3730406,
  author = {Wang, Lishuang and Zhao, Mengfei and Liu, Enyu and Sun, Kebin and Cheng, Ran},
  title = {TensorNEAT: A GPU-accelerated Library for NeuroEvolution of Augmenting Topologies},
  year = {2025},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3730406},
  doi = {10.1145/3730406},
  journal = {ACM Trans. Evol. Learn. Optim.},
  month = apr,
  keywords = {Neuroevolution, GPU Acceleration, Algorithm Library}
}
```
