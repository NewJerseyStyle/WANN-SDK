# WANN SDK

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/NewJerseyStyle/WANN-SDK/actions/workflows/test.yml/badge.svg)](https://github.com/NewJerseyStyle/WANN-SDK/actions/workflows/test.yml)

**WANN SDK** is a high-performance framework for evolving Artificial Neural Networks with **Weight Agnostic Neural Networks (WANN)** method. Built on top of [TensorNEAT](https://github.com/EMI-Group/tensorneat), this toolkit provides a streamlined API for architecture search and weight optimization, leveraging JAX for massive parallelism.

This project originated as a fork of a 2019 research collaboration with [Arthur](https://github.com/rlalpha), modernized to support modern hardware acceleration and the Brax physics engine. Also added weight optimization.

However be aware that the topology found by WANN does not guarantee it is trainable, it can be insensitive to weight optimization as it is Weight Agnostic. Therefore Zero-Cost Proxies (e.g. [SynFlow](https://arxiv.org/abs/2006.05467), [AZ-NAS](https://cvlab.yonsei.ac.kr/projects/AZNAS/)) for Network Architecture Search is introduced to evaluate the potential for weight optimization in the network architecture discovered by WANN.

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
-   **Trainability-Aware Architecture Search:** Apply Zero-Cost Proxies for Network Architecture Search evaluate the room for weight optimization of the architecture.

## 🚀 Getting Started
### Installation

```bash
pip install wann-sdk
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
- [Detailed API reference](docs/APIs.md)
- [Detailed Zero-Cost Proxies reference](docs/trainability_aware_search.md)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Related Work

- For the original 2019 research collaboration with Arthur (2019).
- [Weight Agnostic Neural Networks](https://weightagnostic.github.io/) (Gaier & Ha, 2019)
- [TensorNEAT](https://github.com/EMI-Group/tensorneat) - NEAT algorithms in JAX
- [Brax](https://github.com/google/brax) - Physics simulation in JAX
- [Gymnax](https://github.com/RobertTLange/gymnax) - Classic RL environments in JAX
- [GraSP](https://github.com/alecwangcq/GraSP) - Zero-Cost Proxy

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

### 📑 Zero-Cost Proxies

```
@misc{mellor2021neuralarchitecturesearchtraining,
      title={Neural Architecture Search without Training}, 
      author={Joseph Mellor and Jack Turner and Amos Storkey and Elliot J. Crowley},
      year={2021},
      eprint={2006.04647},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2006.04647}, 
}

@misc{tanaka2020pruningneuralnetworksdata,
      title={Pruning neural networks without any data by iteratively conserving synaptic flow}, 
      author={Hidenori Tanaka and Daniel Kunin and Daniel L. K. Yamins and Surya Ganguli},
      year={2020},
      eprint={2006.05467},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2006.05467}, 
}

@misc{lee2024aznasassemblingzerocostproxies,
      title={AZ-NAS: Assembling Zero-Cost Proxies for Network Architecture Search}, 
      author={Junghyup Lee and Bumsub Ham},
      year={2024},
      eprint={2403.19232},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2403.19232}, 
}

@misc{lee2019snipsingleshotnetworkpruning,
      title={SNIP: Single-shot Network Pruning based on Connection Sensitivity}, 
      author={Namhoon Lee and Thalaiyasingam Ajanthan and Philip H. S. Torr},
      year={2019},
      eprint={1810.02340},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1810.02340}, 
}

@misc{turner2020blockswapfisherguidedblocksubstitution,
      title={BlockSwap: Fisher-guided Block Substitution for Network Compression on a Budget}, 
      author={Jack Turner and Elliot J. Crowley and Michael O'Boyle and Amos Storkey and Gavin Gray},
      year={2020},
      eprint={1906.04113},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1906.04113}, 
}

@misc{wang2020pickingwinningticketstraining,
      title={Picking Winning Tickets Before Training by Preserving Gradient Flow}, 
      author={Chaoqi Wang and Guodong Zhang and Roger Grosse},
      year={2020},
      eprint={2002.07376},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2002.07376}, 
}
```

### 🔗 Optimizers
```
@article{jaxopt_implicit_diff,
  title={Efficient and Modular Implicit Differentiation},
  author={Blondel, Mathieu and Berthet, Quentin and Cuturi, Marco and Frostig, Roy 
    and Hoyer, Stephan and Llinares-L{\'o}pez, Felipe and Pedregosa, Fabian 
    and Vert, Jean-Philippe},
  journal={arXiv preprint arXiv:2105.15183},
  year={2021}
}

@misc{nevergrad,
    author = {J. Rapin and O. Teytaud},
    title = {{Nevergrad - A gradient-free optimization platform}},
    year = {2018},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://GitHub.com/FacebookResearch/Nevergrad}},
}
```
