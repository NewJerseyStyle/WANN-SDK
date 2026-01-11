#!/usr/bin/env python3
"""
MNIST Training Example

Demonstrates using WANN SDK for supervised learning on MNIST.
Shows how to use the Problem and Trainer APIs.

Usage:
    python train_mnist.py
    python train_mnist.py --generations 200 --pop_size 128
"""

import argparse
import jax.numpy as jnp

from wann_sdk import Trainer, TrainerConfig, SupervisedProblem


def load_mnist():
    """
    Load MNIST dataset.

    Returns:
        Tuple of (x_train, y_train, x_test, y_test)
    """
    try:
        # Try tensorflow datasets
        import tensorflow_datasets as tfds

        ds_train = tfds.load('mnist', split='train', as_supervised=True)
        ds_test = tfds.load('mnist', split='test', as_supervised=True)

        # Convert to numpy
        x_train, y_train = [], []
        for image, label in tfds.as_numpy(ds_train):
            x_train.append(image.flatten() / 255.0)
            y_train.append(label)

        x_test, y_test = [], []
        for image, label in tfds.as_numpy(ds_test):
            x_test.append(image.flatten() / 255.0)
            y_test.append(label)

        return (
            jnp.array(x_train),
            jnp.array(y_train),
            jnp.array(x_test),
            jnp.array(y_test),
        )

    except ImportError:
        pass

    try:
        # Try torchvision
        from torchvision import datasets, transforms
        import numpy as np

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.numpy().flatten())
        ])

        train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

        x_train = np.array([train_data[i][0] for i in range(len(train_data))])
        y_train = np.array([train_data[i][1] for i in range(len(train_data))])
        x_test = np.array([test_data[i][0] for i in range(len(test_data))])
        y_test = np.array([test_data[i][1] for i in range(len(test_data))])

        return (
            jnp.array(x_train),
            jnp.array(y_train),
            jnp.array(x_test),
            jnp.array(y_test),
        )

    except ImportError:
        pass

    try:
        # Try keras/tensorflow
        from tensorflow.keras.datasets import mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Flatten and normalize
        x_train = x_train.reshape(-1, 784) / 255.0
        x_test = x_test.reshape(-1, 784) / 255.0

        return (
            jnp.array(x_train),
            jnp.array(y_train),
            jnp.array(x_test),
            jnp.array(y_test),
        )

    except ImportError:
        pass

    # Generate synthetic data if no loader available
    print("Warning: No MNIST loader available. Using synthetic data.")
    print("Install one of: tensorflow-datasets, torchvision, or tensorflow")

    import jax.random as jr
    key = jr.PRNGKey(42)

    n_train, n_test = 1000, 200
    key, k1, k2, k3, k4 = jr.split(key, 5)

    x_train = jr.normal(k1, (n_train, 784))
    y_train = jr.randint(k2, (n_train,), 0, 10)
    x_test = jr.normal(k3, (n_test, 784))
    y_test = jr.randint(k4, (n_test,), 0, 10)

    return x_train, y_train, x_test, y_test


def main():
    parser = argparse.ArgumentParser(description="Train on MNIST with WANN SDK")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations")
    parser.add_argument("--pop_size", type=int, default=64, help="Population size")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for evaluation")
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 64], help="Hidden layer sizes")
    parser.add_argument("--lr", type=float, default=0.02, help="Learning rate")
    parser.add_argument("--noise_std", type=float, default=0.1, help="Noise standard deviation")
    parser.add_argument("--save", type=str, default=None, help="Path to save model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("Loading MNIST dataset...")
    x_train, y_train, x_test, y_test = load_mnist()
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")
    print()

    # Create problem
    problem = SupervisedProblem(
        x_train=x_train,
        y_train=y_train,
        x_val=x_test,
        y_val=y_test,
        loss_fn='cross_entropy',
        batch_size=args.batch_size,
    )

    # Configure trainer
    config = TrainerConfig(
        pop_size=args.pop_size,
        learning_rate=args.lr,
        noise_std=args.noise_std,
        hidden_sizes=args.hidden,
        activation='relu',
        output_activation='none',  # Cross-entropy applies softmax internally
        seed=args.seed,
    )

    # Create trainer
    trainer = Trainer(problem, config)
    print(trainer.summary())
    print()

    # Train
    results = trainer.fit(generations=args.generations, log_interval=10)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    network = trainer.get_network()
    metrics = problem.evaluate_accuracy(network, use_val=True)
    print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Test Loss: {metrics['loss']:.4f}")

    # Also evaluate on training set
    train_metrics = problem.evaluate_accuracy(network, use_val=False)
    print(f"Train Accuracy: {train_metrics['accuracy'] * 100:.2f}%")

    # Save if requested
    if args.save:
        trainer.save(args.save)

    return trainer, results


if __name__ == "__main__":
    main()
