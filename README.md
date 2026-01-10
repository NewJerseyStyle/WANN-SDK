# WANN SDK

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/backend-JAX-orange.svg)](https://github.com/google/jax)
[![TensorNEAT](https://img.shields.io/badge/TensorNEAT-orange.svg)](https://github.com/EMI-Group/tensorneat)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**WANN SDK** is a high-performance framework for evolving Artificial Neural Networks with **Weight Agnostic Neural Networks (WANN)** method. Built on top of [TensorNEAT](https://github.com/EMI-Group/tensorneat), this toolkit provides a streamlined API for architecture search and weight optimization, leveraging JAX for massive parallelism.

This project originated as a fork of a 2019 research collaboration with [Arthur](https://github.com/rlalpha), modernized to support modern hardware acceleration and the Brax physics engine.

## 🧠 What is WANN?

Unlike traditional neural networks where the focus is on optimizing specific weight values, and traditional architecture search requires weight optimization during training. Our experimental method was using shared weight to archtiecture search and optimize weight after fixing the network **topology**. We adapt **Weight Agnostic Neural Networks** to find architectures that can perform tasks without optimize weight to compare and find the **best topology**.

Our SDK uses a two-stage approach:
1.  **Search Stage:** Uses a NEAT-based WANN evolutionary algorithm to find robust topologies with fixed/shared weights.
2.  **Fine-tuning Stage:** Optimizes the weights for the discovered architecture to reach peak performance.

## ✨ Key Features

-   **TensorNEAT Integration:** Fully vectorized evolutionary operations.
-   **JAX Accelerated:** Designed for speed on CPUs, GPUs, and TPUs.
-   **Two-Stage Pipeline:** Decoupled architecture search and weight optimization.

## 🚀 Getting Started

### Prerequisites

Ensure you have a modern Python environment. For GPU acceleration, ensure your CUDA drivers are up to date.

### Installation

```bash
# 1. Install JAX (CPU version)
pip install jax 

# OR 1. Install JAX (GPU version - adjust cuda version as needed)
# pip install "jax[cuda12]"

# 2. Install dependencies and TensorNEAT
pip install git+https://github.com/EMI-Group/tensorneat.git gymnas brax
```

### Running the Example (Bipedal/Humanoid)

We provide a comprehensive example using the `Bipedal` environment in Brax. You can run the pipeline in stages or all at once.

#### Example 1: Step-by-Step Execution

1.  **Search:** Find the best neural architecture.
    ```bash
    python example/wann_bipedal.py --mode v2 --stage search
    ```
2.  **Train:** Optimize the weights of the discovered architecture.
    ```bash
    python example/wann_bipedal.py --mode v2 --stage train
    ```
3.  **Evaluate:** Test the performance of your trained model.
    ```bash
    python example/wann_bipedal.py --mode v2 --stage eval
    ```

#### Example 2: Full Pipeline
To run everything from scratch in one go:
```bash
python example/wann_bipedal.py --mode v2 --stage full
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

-   **Arthur:** For the original 2019 research collaboration.
-   **EMI Group:** For the excellent [TensorNEAT](https://github.com/EMI-Group/tensorneat) foundation.

## Citing WANN SDK, WANN, TensorNEAT
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
