"""
Tests for WANN SDK Zero-Cost Proxies

Tests ZCP evaluator and trainability-aware search.
"""

import pytest
import jax
import jax.numpy as jnp


class TestZCPImports:
    """Test ZCP module imports."""

    def test_import_zcp_functions(self):
        """Test ZCP function imports."""
        from wann_sdk import (
            ZCPEvaluator,
            ZCPConfig,
            compute_synflow,
            compute_naswot,
            compute_snip,
            compute_trainability,
        )

        assert ZCPEvaluator is not None
        assert ZCPConfig is not None
        assert callable(compute_synflow)
        assert callable(compute_naswot)
        assert callable(compute_snip)
        assert callable(compute_trainability)

    def test_import_trainability_search(self):
        """Test trainability search imports."""
        from wann_sdk import (
            TrainabilityAwareSearch,
            TrainabilitySearchConfig,
        )

        assert TrainabilityAwareSearch is not None
        assert TrainabilitySearchConfig is not None


class TestZCPEvaluator:
    """Test ZCPEvaluator class."""

    def test_evaluator_creation(self):
        """Test creating ZCP evaluator."""
        from wann_sdk import ZCPEvaluator

        evaluator = ZCPEvaluator(
            proxies=['synflow', 'naswot'],
            aggregation='geometric',
        )

        assert evaluator is not None

    def test_evaluator_evaluate(self, simple_params, simple_forward_fn):
        """Test evaluating with ZCP evaluator."""
        from wann_sdk import ZCPEvaluator

        evaluator = ZCPEvaluator(
            proxies=['synflow'],
            aggregation='mean',
        )

        x = jax.random.normal(jax.random.PRNGKey(0), (10, 4))
        y = jax.random.randint(jax.random.PRNGKey(1), (10,), 0, 2)

        scores = evaluator.evaluate(
            simple_forward_fn,
            simple_params,
            x, y,
            input_shape=(4,),
        )

        assert isinstance(scores, dict)
        assert 'synflow' in scores or 'aggregate' in scores


class TestZCPFunctions:
    """Test individual ZCP functions."""

    def test_compute_synflow(self, simple_params, simple_forward_fn):
        """Test synflow computation."""
        from wann_sdk import compute_synflow

        score = compute_synflow(
            simple_forward_fn,
            simple_params,
            input_shape=(4,),
        )

        assert isinstance(score, (float, jnp.ndarray))
        assert not jnp.isnan(score)

    def test_compute_naswot(self, simple_params, simple_forward_fn):
        """Test NASWOT computation."""
        from wann_sdk import compute_naswot

        x = jax.random.normal(jax.random.PRNGKey(0), (10, 4))

        score = compute_naswot(
            simple_forward_fn,
            simple_params,
            x,
        )

        assert isinstance(score, (float, jnp.ndarray))

    def test_compute_trainability(self, simple_params, simple_forward_fn):
        """Test trainability computation."""
        from wann_sdk import compute_trainability

        x = jax.random.normal(jax.random.PRNGKey(0), (10, 4))
        y = jnp.zeros((10, 2))  # One-hot encoded

        score = compute_trainability(
            simple_forward_fn,
            simple_params,
            x, y,
        )

        assert isinstance(score, (float, jnp.ndarray))


class TestTrainabilitySearchConfig:
    """Test TrainabilitySearchConfig."""

    def test_default_config(self):
        """Test default trainability search config."""
        from wann_sdk import TrainabilitySearchConfig

        config = TrainabilitySearchConfig()

        assert config.strategy in ['hybrid', 'filter', 'multi_objective']
        assert isinstance(config.zcp_weight, float)
        assert isinstance(config.zcp_proxies, list)

    def test_custom_config(self):
        """Test custom trainability search config."""
        from wann_sdk import TrainabilitySearchConfig

        config = TrainabilitySearchConfig(
            strategy='hybrid',
            zcp_weight=0.5,
            zcp_proxies=['synflow', 'naswot'],
        )

        assert config.strategy == 'hybrid'
        assert config.zcp_weight == 0.5
        assert config.zcp_proxies == ['synflow', 'naswot']


class TestTrainabilityAwareSearch:
    """Test TrainabilityAwareSearch class."""

    def test_search_creation(self, classification_data):
        """Test creating trainability-aware search."""
        from wann_sdk import (
            TrainabilityAwareSearch,
            SearchConfig,
            SupervisedProblem,
        )

        x, y = classification_data
        problem = SupervisedProblem(x, y, loss_fn='cross_entropy')

        search = TrainabilityAwareSearch(
            problem,
            SearchConfig(pop_size=5, verbose=False),
            strategy='hybrid',
            zcp_weight=0.3,
            zcp_proxies=['synflow'],
        )

        assert search is not None

    def test_quick_search(self, classification_data):
        """Test quick trainability-aware search."""
        from wann_sdk import (
            TrainabilityAwareSearch,
            SearchConfig,
            SupervisedProblem,
        )

        x, y = classification_data
        problem = SupervisedProblem(x, y, loss_fn='cross_entropy')

        search = TrainabilityAwareSearch(
            problem,
            SearchConfig(pop_size=5, max_nodes=5, verbose=False),
            strategy='hybrid',
            zcp_weight=0.2,
            zcp_proxies=['synflow'],
        )

        genome = search.run(generations=3)

        assert genome is not None
        assert genome.num_inputs == 4  # input_dim from classification_data
        assert genome.num_outputs == 2  # 2 classes

    def test_get_zcp_breakdown(self, classification_data):
        """Test getting ZCP breakdown."""
        from wann_sdk import (
            TrainabilityAwareSearch,
            SearchConfig,
            SupervisedProblem,
        )

        x, y = classification_data
        problem = SupervisedProblem(x, y, loss_fn='cross_entropy')

        search = TrainabilityAwareSearch(
            problem,
            SearchConfig(pop_size=5, verbose=False),
            strategy='hybrid',
            zcp_weight=0.3,
            zcp_proxies=['synflow'],
        )

        genome = search.run(generations=2)
        breakdown = search.get_zcp_breakdown(genome)

        # May return None or dict depending on implementation
        assert breakdown is None or isinstance(breakdown, dict)


class TestZCPIntegration:
    """Integration tests for ZCP with full pipeline."""

    def test_zcp_to_weight_training(self, classification_data):
        """Test ZCP search followed by weight training."""
        from wann_sdk import (
            TrainabilityAwareSearch,
            SearchConfig,
            SupervisedProblem,
            WeightTrainer,
            WeightTrainerConfig,
        )

        x, y = classification_data
        problem = SupervisedProblem(x, y, loss_fn='cross_entropy')

        # Stage 1: Trainability-aware search
        search = TrainabilityAwareSearch(
            problem,
            SearchConfig(pop_size=5, max_nodes=5, verbose=False),
            strategy='hybrid',
            zcp_weight=0.2,
            zcp_proxies=['synflow'],
        )
        genome = search.run(generations=2)

        # Stage 2: Weight training
        trainer_config = WeightTrainerConfig(
            optimizer='adam',
            learning_rate=0.01,
            verbose=False,
        )

        trainer = WeightTrainer(genome, problem, trainer_config)
        result = trainer.fit(epochs=3)

        assert 'best_fitness' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
