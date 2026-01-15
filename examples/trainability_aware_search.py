"""
Trainability-Aware Architecture Search Example

This example demonstrates how to use zero-cost proxies to find
architectures that are not only weight-agnostic but also trainable.

Standard WANN finds architectures that work with any weight value.
Trainability-aware search additionally considers:
- Synflow: Data-agnostic parameter importance
- NASWOT: Activation pattern diversity
- Trainability: Gradient flow stability

This helps find networks that can be further optimized through
weight training in Stage 2.
"""

import jax
import jax.numpy as jnp
import numpy as np

from wann_sdk import (
    TrainabilityAwareSearch,
    TrainabilitySearchConfig,
    SearchConfig,
    SupervisedProblem,
    WeightTrainer,
    WeightTrainerConfig,
    ZCPEvaluator,
    compute_synflow,
    compute_naswot,
    compute_trainability,
)


def generate_classification_data(n_samples=1000, n_features=8, n_classes=3, seed=42):
    """Generate synthetic classification dataset."""
    np.random.seed(seed)

    # Generate cluster centers
    centers = np.random.randn(n_classes, n_features) * 2

    # Generate samples around centers
    samples_per_class = n_samples // n_classes
    X = []
    y = []

    for i, center in enumerate(centers):
        samples = center + np.random.randn(samples_per_class, n_features) * 0.5
        X.append(samples)
        y.extend([i] * samples_per_class)

    X = np.vstack(X).astype(np.float32)
    y = np.array(y, dtype=np.int32)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # One-hot encode labels
    y_onehot = np.eye(n_classes)[y].astype(np.float32)

    return jnp.array(X), jnp.array(y_onehot)


def compare_wann_vs_trainability():
    """Compare standard WANN search with trainability-aware search."""
    print("=" * 70)
    print("Trainability-Aware Architecture Search Comparison")
    print("=" * 70)

    # Generate data
    X, y = generate_classification_data(n_samples=500, n_features=8, n_classes=3)
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features, {y.shape[1]} classes")

    # Split train/test
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Create problem
    problem = SupervisedProblem(
        X_train, y_train,
        loss_fn='cross_entropy',
        x_val=X_test,
        y_val=y_test,
    )

    # Common search config
    base_config = SearchConfig(
        pop_size=30,
        max_nodes=15,
        activation_options=['tanh', 'relu', 'sigmoid', 'sin'],
        elite_size=5,
        seed=42,
    )

    generations = 20

    # =========================================
    # Standard WANN Search
    # =========================================
    print("\n" + "-" * 50)
    print("Running Standard WANN Search...")
    print("-" * 50)

    from wann_sdk import ArchitectureSearch
    wann_search = ArchitectureSearch(problem, base_config)
    wann_genome = wann_search.run(generations=generations)

    print(f"\nWANN Best Fitness: {wann_genome.fitness:.4f}")
    print(f"WANN Complexity: {wann_genome.complexity}")

    # =========================================
    # Trainability-Aware Search
    # =========================================
    print("\n" + "-" * 50)
    print("Running Trainability-Aware Search...")
    print("-" * 50)

    trainability_search = TrainabilityAwareSearch(
        problem,
        base_config,
        strategy='hybrid',
        zcp_weight=0.3,  # 30% ZCP, 70% WANN
        zcp_proxies=['synflow', 'naswot', 'trainability'],
    )
    trainable_genome = trainability_search.run(generations=generations)

    print(f"\nTrainability-Aware Best Fitness: {trainable_genome.fitness:.4f}")
    print(f"Trainability-Aware Complexity: {trainable_genome.complexity}")

    # Get ZCP breakdown
    zcp_breakdown = trainability_search.get_zcp_breakdown(trainable_genome)
    if zcp_breakdown:
        print("\nZCP Breakdown:")
        for proxy, score in zcp_breakdown.items():
            print(f"  {proxy}: {score:.4f}")

    # =========================================
    # Stage 2: Comprehensive Weight Training Comparison
    # =========================================
    print("\n" + "=" * 70)
    print("Stage 2: Weight Optimization Comparison")
    print("=" * 70)

    training_epochs = 100
    log_interval = 10

    # --- Training Configuration ---
    trainer_config = WeightTrainerConfig(
        optimizer='adam',
        learning_rate=0.01,
        batch_size=32,
        verbose=False,
    )

    # Initialize results storage
    wann_results = {'history': [], 'final_loss': float('inf'), 'final_val_loss': float('inf')}
    trainable_results = {'history': [], 'final_loss': float('inf'), 'final_val_loss': float('inf')}

    # --- Train WANN Architecture ---
    print("\n" + "-" * 50)
    print("Training Standard WANN Architecture...")
    print("-" * 50)
    wann_trainer = None
    try:
        wann_trainer = WeightTrainer(wann_genome, problem, trainer_config)
        result = wann_trainer.fit(epochs=training_epochs, log_interval=log_interval)

        # fit() returns {'best_fitness': ..., 'epochs': ..., 'history': [...]}
        history = result.get('history', [])
        if history:
            wann_results['history'] = history
            wann_results['final_loss'] = history[-1].get('loss', float('inf'))
            wann_results['final_val_loss'] = history[-1].get('val_loss', wann_results['final_loss'])

            # Show training progress
            print(f"\n  Epoch    Train Loss    Fitness")
            print(f"  " + "-" * 35)
            for entry in history[::max(1, len(history)//5)]:  # Show ~5 checkpoints
                epoch = entry.get('epoch', 0)
                loss = entry.get('loss', 0)
                fitness = entry.get('fitness', 0)
                print(f"  {epoch:>5}    {loss:>10.4f}    {fitness:>10.4f}")

            print(f"\n  Final: loss={wann_results['final_loss']:.4f}")
            print(f"  Best fitness: {result.get('best_fitness', 'N/A')}")
    except Exception as e:
        import traceback
        print(f"  Training failed: {e}")
        traceback.print_exc()

    # --- Train Trainability-Aware Architecture ---
    print("\n" + "-" * 50)
    print("Training Trainability-Aware Architecture...")
    print("-" * 50)
    trainable_trainer = None
    try:
        trainable_trainer = WeightTrainer(trainable_genome, problem, trainer_config)
        result = trainable_trainer.fit(epochs=training_epochs, log_interval=log_interval)

        # fit() returns {'best_fitness': ..., 'epochs': ..., 'history': [...]}
        history = result.get('history', [])
        if history:
            trainable_results['history'] = history
            trainable_results['final_loss'] = history[-1].get('loss', float('inf'))
            trainable_results['final_val_loss'] = history[-1].get('val_loss', trainable_results['final_loss'])

            # Show training progress
            print(f"\n  Epoch    Train Loss    Fitness")
            print(f"  " + "-" * 35)
            for entry in history[::max(1, len(history)//5)]:
                epoch = entry.get('epoch', 0)
                loss = entry.get('loss', 0)
                fitness = entry.get('fitness', 0)
                print(f"  {epoch:>5}    {loss:>10.4f}    {fitness:>10.4f}")

            print(f"\n  Final: loss={trainable_results['final_loss']:.4f}")
            print(f"  Best fitness: {result.get('best_fitness', 'N/A')}")
    except Exception as e:
        import traceback
        print(f"  Training failed: {e}")
        traceback.print_exc()

    # --- Convergence Analysis ---
    print("\n" + "-" * 50)
    print("Convergence Analysis")
    print("-" * 50)

    def analyze_convergence(history, name):
        """Analyze training convergence metrics."""
        if not history:
            return None

        losses = [h.get('loss', float('inf')) for h in history]
        epochs = [h.get('epoch', i) for i, h in enumerate(history)]

        # Find convergence point (when loss stops improving significantly)
        convergence_epoch = epochs[-1]
        convergence_threshold = 0.01  # 1% improvement threshold
        for i in range(len(losses) - 1):
            if i > 5:  # Need at least 5 epochs
                recent_improvement = (losses[i-5] - losses[i]) / (losses[i-5] + 1e-8)
                if recent_improvement < convergence_threshold:
                    convergence_epoch = epochs[i]
                    break

        # Calculate improvement rate (loss reduction per epoch)
        if len(losses) >= 2:
            total_improvement = losses[0] - losses[-1]
            improvement_rate = total_improvement / len(losses)
        else:
            improvement_rate = 0

        return {
            'initial_loss': losses[0] if losses else float('inf'),
            'final_loss': losses[-1] if losses else float('inf'),
            'total_improvement': losses[0] - losses[-1] if len(losses) >= 2 else 0,
            'improvement_rate': improvement_rate,
            'convergence_epoch': convergence_epoch,
            'min_loss': min(losses) if losses else float('inf'),
        }

    wann_conv = analyze_convergence(wann_results['history'], 'WANN')
    trainable_conv = analyze_convergence(trainable_results['history'], 'Trainability')

    if wann_conv and trainable_conv:
        print(f"\n  {'Metric':<25} {'WANN':>15} {'Trainability':>15}")
        print(f"  " + "-" * 55)
        print(f"  {'Initial Loss':<25} {wann_conv['initial_loss']:>15.4f} {trainable_conv['initial_loss']:>15.4f}")
        print(f"  {'Final Loss':<25} {wann_conv['final_loss']:>15.4f} {trainable_conv['final_loss']:>15.4f}")
        print(f"  {'Min Loss':<25} {wann_conv['min_loss']:>15.4f} {trainable_conv['min_loss']:>15.4f}")
        print(f"  {'Total Improvement':<25} {wann_conv['total_improvement']:>15.4f} {trainable_conv['total_improvement']:>15.4f}")
        print(f"  {'Improvement/Epoch':<25} {wann_conv['improvement_rate']:>15.6f} {trainable_conv['improvement_rate']:>15.6f}")
        print(f"  {'Convergence Epoch':<25} {wann_conv['convergence_epoch']:>15} {trainable_conv['convergence_epoch']:>15}")

    # --- Validation Performance ---
    print("\n" + "-" * 50)
    print("Validation Performance (Test Set)")
    print("-" * 50)

    def evaluate_accuracy(trainer, X_test, y_test):
        """Evaluate classification accuracy on test set."""
        if trainer is None:
            return 0.0
        try:
            # Use predict method for batch inference
            predictions = trainer.predict(X_test)

            # For classification: compare argmax
            pred_classes = jnp.argmax(predictions, axis=-1)
            true_classes = jnp.argmax(y_test, axis=-1)
            accuracy = jnp.mean(pred_classes == true_classes)
            return float(accuracy)
        except Exception as e:
            print(f"    Accuracy evaluation failed: {e}")
            return 0.0

    wann_accuracy = 0.0
    trainable_accuracy = 0.0

    if wann_trainer is not None:
        wann_accuracy = evaluate_accuracy(wann_trainer, X_test, y_test)
        print(f"  WANN Test Accuracy: {wann_accuracy*100:.2f}%")
    else:
        print(f"  WANN Test Accuracy: N/A (training failed)")

    if trainable_trainer is not None:
        trainable_accuracy = evaluate_accuracy(trainable_trainer, X_test, y_test)
        print(f"  Trainability Test Accuracy: {trainable_accuracy*100:.2f}%")
    else:
        print(f"  Trainability Test Accuracy: N/A (training failed)")

    # =========================================
    # Final Summary
    # =========================================
    print("\n" + "=" * 70)
    print("Final Summary: WANN vs Trainability-Aware Search")
    print("=" * 70)

    print(f"\n  {'Stage':<15} {'Metric':<25} {'WANN':>12} {'Trainability':>12} {'Winner':>10}")
    print(f"  " + "-" * 75)

    # Stage 1 metrics
    wann_fit = wann_genome.fitness
    train_fit = trainable_genome.fitness
    winner = "WANN" if wann_fit > train_fit else "Train" if train_fit > wann_fit else "Tie"
    print(f"  {'Stage 1':<15} {'Search Fitness':<25} {wann_fit:>12.4f} {train_fit:>12.4f} {winner:>10}")

    wann_comp = wann_genome.complexity
    train_comp = trainable_genome.complexity
    winner = "WANN" if wann_comp < train_comp else "Train" if train_comp < wann_comp else "Tie"
    print(f"  {'':<15} {'Complexity':<25} {wann_comp:>12} {train_comp:>12} {winner:>10}")

    # Stage 2 metrics
    wann_loss = wann_results['final_loss']
    train_loss = trainable_results['final_loss']
    winner = "WANN" if wann_loss < train_loss else "Train" if train_loss < wann_loss else "Tie"
    print(f"  {'Stage 2':<15} {'Final Train Loss':<25} {wann_loss:>12.4f} {train_loss:>12.4f} {winner:>10}")

    wann_val = wann_results['final_val_loss']
    train_val = trainable_results['final_val_loss']
    winner = "WANN" if wann_val < train_val else "Train" if train_val < wann_val else "Tie"
    print(f"  {'':<15} {'Final Val Loss':<25} {wann_val:>12.4f} {train_val:>12.4f} {winner:>10}")

    winner = "WANN" if wann_accuracy > trainable_accuracy else "Train" if trainable_accuracy > wann_accuracy else "Tie"
    print(f"  {'':<15} {'Test Accuracy':<25} {wann_accuracy*100:>11.2f}% {trainable_accuracy*100:>11.2f}% {winner:>10}")

    if wann_conv and trainable_conv:
        wann_rate = wann_conv['improvement_rate']
        train_rate = trainable_conv['improvement_rate']
        winner = "WANN" if wann_rate > train_rate else "Train" if train_rate > wann_rate else "Tie"
        print(f"  {'':<15} {'Convergence Speed':<25} {wann_rate:>12.6f} {train_rate:>12.6f} {winner:>10}")

    # Overall assessment
    print("\n" + "-" * 70)
    print("Interpretation:")
    print("-" * 70)

    if trainable_accuracy > wann_accuracy and train_loss < wann_loss:
        print("  ✓ Trainability-aware search found a more trainable architecture!")
        print("    The ZCP-guided search successfully identified an architecture with")
        print("    better gradient flow, leading to improved weight optimization.")
    elif wann_accuracy > trainable_accuracy:
        print("  ✓ Standard WANN search performed better in this case.")
        print("    The weight-agnostic fitness was a sufficient proxy for trainability.")
        print("    (Consider: more generations, different ZCP weights, or different data)")
    else:
        print("  ≈ Both approaches achieved similar results.")
        print("    The architectures have comparable trainability characteristics.")


def demonstrate_zcp_evaluator():
    """Demonstrate standalone ZCP evaluation."""
    print("\n" + "=" * 70)
    print("Zero-Cost Proxy Evaluator Demo")
    print("=" * 70)

    # Create a simple test network
    X = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    y = jnp.array([[1.0], [0.0], [1.0]])

    # Simple MLP parameters
    params = {
        'w1': jax.random.normal(jax.random.PRNGKey(0), (3, 8)),
        'b1': jnp.zeros(8),
        'w2': jax.random.normal(jax.random.PRNGKey(1), (8, 1)),
        'b2': jnp.zeros(1),
    }

    def forward_fn(params, x):
        h = jax.nn.relu(x @ params['w1'] + params['b1'])
        return h @ params['w2'] + params['b2']

    # Evaluate with ZCP
    evaluator = ZCPEvaluator(
        proxies=['synflow', 'naswot', 'trainability', 'expressivity'],
        aggregation='geometric',
    )

    scores = evaluator.evaluate(
        forward_fn, params, X, y,
        input_shape=(3,),
    )

    print("\nZCP Scores for test network:")
    for proxy, score in scores.items():
        print(f"  {proxy:<20}: {score:.4f}")

    # Individual proxy computation
    print("\n" + "-" * 40)
    print("Individual Proxy Functions:")
    print("-" * 40)

    synflow = compute_synflow(forward_fn, params, (3,))
    print(f"  compute_synflow: {synflow:.4f}")

    naswot = compute_naswot(forward_fn, params, X)
    print(f"  compute_naswot: {naswot:.4f}")

    trainability = compute_trainability(forward_fn, params, X, y)
    print(f"  compute_trainability: {trainability:.4f}")


def main():
    """Run all examples."""
    # Compare WANN vs trainability-aware search
    compare_wann_vs_trainability()

    # Demonstrate ZCP evaluator
    demonstrate_zcp_evaluator()


if __name__ == "__main__":
    main()
