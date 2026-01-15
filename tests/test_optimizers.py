"""
Tests for WANN SDK Optimizer System

Tests both gradient-based (Optax) and evolutionary optimizers.
"""

import pytest
import jax
import jax.numpy as jnp


class TestOptimizerImports:
    """Test optimizer imports."""

    def test_import_base_classes(self):
        """Test base class imports."""
        from wann_sdk.optimizers import (
            BaseOptimizer,
            GradientOptimizer,
            EvolutionaryOptimizer,
            OptimizerState,
        )
        assert BaseOptimizer is not None
        assert GradientOptimizer is not None
        assert EvolutionaryOptimizer is not None
        assert OptimizerState is not None

    def test_import_gradient_optimizers(self):
        """Test gradient optimizer imports."""
        from wann_sdk.optimizers import (
            Adam, AdamW, SGD, RMSProp, AdaGrad, LBFGS, Lion, Lamb,
        )
        assert Adam is not None
        assert AdamW is not None
        assert SGD is not None
        assert RMSProp is not None
        assert AdaGrad is not None
        assert LBFGS is not None
        assert Lion is not None
        assert Lamb is not None

    def test_import_evolutionary_optimizers(self):
        """Test evolutionary optimizer imports."""
        from wann_sdk.optimizers import ES, CMA, DE, PSO, NGOpt
        assert ES is not None
        # CMA, DE, PSO, NGOpt may be placeholders if nevergrad not installed
        assert CMA is not None
        assert DE is not None
        assert PSO is not None
        assert NGOpt is not None

    def test_import_registry_functions(self):
        """Test registry function imports."""
        from wann_sdk.optimizers import (
            list_optimizers,
            get_optimizer,
            register_optimizer,
            create_optimizer,
            is_gradient_based,
        )
        assert callable(list_optimizers)
        assert callable(get_optimizer)
        assert callable(register_optimizer)
        assert callable(create_optimizer)
        assert callable(is_gradient_based)


class TestOptimizerRegistry:
    """Test optimizer registry functions."""

    def test_list_optimizers(self):
        """Test listing all optimizers."""
        from wann_sdk.optimizers import list_optimizers

        all_opts = list_optimizers()
        assert isinstance(all_opts, dict)
        assert 'adam' in all_opts
        assert 'sgd' in all_opts
        assert 'es' in all_opts

    def test_list_optimizers_by_category(self):
        """Test listing optimizers by category."""
        from wann_sdk.optimizers import list_optimizers

        gradient_opts = list_optimizers(category='gradient')
        assert 'adam' in gradient_opts
        assert 'es' not in gradient_opts

        evo_opts = list_optimizers(category='evolutionary')
        assert 'es' in evo_opts
        assert 'adam' not in evo_opts

    def test_get_optimizer(self):
        """Test getting optimizer by name."""
        from wann_sdk.optimizers import get_optimizer, Adam

        AdamClass = get_optimizer('adam')
        assert AdamClass is Adam

    def test_get_optimizer_case_insensitive(self):
        """Test case-insensitive lookup."""
        from wann_sdk.optimizers import get_optimizer

        assert get_optimizer('Adam') is get_optimizer('adam')
        assert get_optimizer('ADAM') is get_optimizer('adam')

    def test_get_optimizer_unknown(self):
        """Test error for unknown optimizer."""
        from wann_sdk.optimizers import get_optimizer

        with pytest.raises(KeyError):
            get_optimizer('unknown_optimizer')

    def test_create_optimizer_from_string(self):
        """Test creating optimizer from string."""
        from wann_sdk.optimizers import create_optimizer, Adam

        opt = create_optimizer('adam', learning_rate=0.01)
        assert isinstance(opt, Adam)
        assert opt.learning_rate == 0.01

    def test_create_optimizer_from_instance(self):
        """Test passing existing instance."""
        from wann_sdk.optimizers import create_optimizer, Adam

        existing = Adam(learning_rate=0.05)
        opt = create_optimizer(existing)
        assert opt is existing

    def test_is_gradient_based(self):
        """Test is_gradient_based function."""
        from wann_sdk.optimizers import is_gradient_based

        assert is_gradient_based('adam') is True
        assert is_gradient_based('sgd') is True
        assert is_gradient_based('es') is False


class TestGradientOptimizers:
    """Test gradient-based optimizers."""

    def test_adam_init_and_update(self):
        """Test Adam optimizer."""
        from wann_sdk.optimizers import Adam

        opt = Adam(learning_rate=0.001)
        params = jnp.array([1.0, 2.0, 3.0])
        grads = jnp.array([0.1, 0.2, 0.3])

        state = opt.init_state(params)
        assert state.step == 0
        assert jnp.allclose(state.params, params)

        new_state = opt.update(state, grads=grads)
        assert new_state.step == 1
        assert not jnp.allclose(new_state.params, params)  # Params changed

    def test_adamw_init_and_update(self):
        """Test AdamW optimizer."""
        from wann_sdk.optimizers import AdamW

        opt = AdamW(learning_rate=0.001, weight_decay=0.01)
        params = jnp.array([1.0, 2.0, 3.0])
        grads = jnp.array([0.1, 0.2, 0.3])

        state = opt.init_state(params)
        new_state = opt.update(state, grads=grads)

        assert new_state.step == 1
        assert not jnp.allclose(new_state.params, params)

    def test_sgd_with_momentum(self):
        """Test SGD with momentum."""
        from wann_sdk.optimizers import SGD

        opt = SGD(learning_rate=0.01, momentum=0.9)
        params = jnp.array([1.0, 2.0])
        grads = jnp.array([0.5, 0.5])

        state = opt.init_state(params)
        state = opt.update(state, grads=grads)
        state = opt.update(state, grads=grads)  # Momentum builds up

        assert state.step == 2

    def test_rmsprop(self):
        """Test RMSProp optimizer."""
        from wann_sdk.optimizers import RMSProp

        opt = RMSProp(learning_rate=0.001)
        params = jnp.array([1.0, 2.0, 3.0])
        grads = jnp.array([0.1, 0.2, 0.3])

        state = opt.init_state(params)
        new_state = opt.update(state, grads=grads)

        assert new_state.step == 1

    def test_lion(self):
        """Test Lion optimizer."""
        from wann_sdk.optimizers import Lion

        opt = Lion(learning_rate=0.0001)
        params = jnp.array([1.0, 2.0, 3.0])
        grads = jnp.array([0.1, 0.2, 0.3])

        state = opt.init_state(params)
        new_state = opt.update(state, grads=grads)

        assert new_state.step == 1

    def test_gradient_optimizer_requires_grads(self):
        """Test that gradient optimizers require gradients."""
        from wann_sdk.optimizers import Adam

        opt = Adam()
        params = jnp.array([1.0, 2.0])
        state = opt.init_state(params)

        with pytest.raises(ValueError, match="requires gradients"):
            opt.update(state)  # No grads provided


class TestEvolutionaryOptimizers:
    """Test evolutionary optimizers."""

    def test_es_init_and_update(self):
        """Test built-in ES optimizer."""
        from wann_sdk.optimizers import ES

        opt = ES(population_size=8, learning_rate=0.01, noise_std=0.1)
        params = jnp.array([1.0, 2.0, 3.0])

        state = opt.init_state(params)
        assert state.step == 0

        # ES needs loss_fn and key
        def loss_fn(p):
            return jnp.sum(p ** 2)

        key = jax.random.PRNGKey(42)
        new_state = opt.update(state, loss_fn=loss_fn, key=key)

        assert new_state.step == 1

    def test_es_ask_tell(self):
        """Test ES ask-tell interface."""
        from wann_sdk.optimizers import ES

        opt = ES(population_size=4)
        params = jnp.array([1.0, 2.0])
        state = opt.init_state(params)

        key = jax.random.PRNGKey(0)
        candidates, ask_state = opt.ask(state, key)

        assert candidates.shape[0] == 4  # population_size
        assert candidates.shape[1] == 2  # param dim

        # Evaluate candidates
        fitnesses = jnp.array([-jnp.sum(c ** 2) for c in candidates])

        # Tell results
        new_state = opt.tell(state, ask_state, fitnesses)
        assert new_state is not None

    def test_nevergrad_placeholder_error(self):
        """Test that nevergrad placeholders raise helpful errors."""
        from wann_sdk.optimizers import CMA, _NEVERGRAD_AVAILABLE

        if not _NEVERGRAD_AVAILABLE:
            with pytest.raises(ImportError, match="requires nevergrad"):
                CMA(population_size=32)


class TestOptimizerState:
    """Test OptimizerState dataclass."""

    def test_optimizer_state_creation(self):
        """Test creating optimizer state."""
        from wann_sdk.optimizers import OptimizerState

        state = OptimizerState(
            step=0,
            params=jnp.array([1.0, 2.0]),
            internal={'key': 'value'},
        )

        assert state.step == 0
        assert jnp.allclose(state.params, jnp.array([1.0, 2.0]))
        assert state.internal == {'key': 'value'}

    def test_optimizer_state_defaults(self):
        """Test optimizer state defaults."""
        from wann_sdk.optimizers import OptimizerState

        state = OptimizerState(
            step=1,
            params=jnp.array([1.0]),
        )

        assert state.internal is None


class TestCustomOptimizer:
    """Test custom optimizer registration."""

    def test_register_custom_optimizer(self):
        """Test registering a custom optimizer."""
        from wann_sdk.optimizers import (
            BaseOptimizer, OptimizerState,
            register_optimizer, get_optimizer, unregister_optimizer,
        )

        class MyOptimizer(BaseOptimizer):
            name = "test-opt"
            is_gradient_based = True

            def __init__(self, learning_rate=0.01):
                super().__init__(learning_rate=learning_rate)
                self.learning_rate = learning_rate

            def init_state(self, params):
                return OptimizerState(step=0, params=params)

            def update(self, state, grads=None, **kwargs):
                new_params = state.params - self.learning_rate * grads
                return OptimizerState(step=state.step + 1, params=new_params)

        # Register
        register_optimizer("test-opt", MyOptimizer)

        # Use it
        OptClass = get_optimizer("test-opt")
        assert OptClass is MyOptimizer

        opt = OptClass(learning_rate=0.1)
        params = jnp.array([1.0, 2.0])
        grads = jnp.array([0.1, 0.1])

        state = opt.init_state(params)
        new_state = opt.update(state, grads=grads)

        expected = params - 0.1 * grads
        assert jnp.allclose(new_state.params, expected)

        # Cleanup
        unregister_optimizer("test-opt")


class TestWeightTrainerIntegration:
    """Test optimizer integration with WeightTrainer."""

    def test_weight_trainer_with_adam(self, simple_genome, dummy_problem):
        """Test WeightTrainer with Adam optimizer."""
        from wann_sdk import WeightTrainer, WeightTrainerConfig

        config = WeightTrainerConfig(
            optimizer='adam',
            learning_rate=0.01,
            verbose=False,
        )

        trainer = WeightTrainer(simple_genome, dummy_problem, config)
        result = trainer.fit(epochs=3)

        assert 'best_fitness' in result
        assert 'history' in result

    def test_weight_trainer_with_es(self, simple_genome, dummy_problem):
        """Test WeightTrainer with ES optimizer."""
        from wann_sdk import WeightTrainer, WeightTrainerConfig

        config = WeightTrainerConfig(
            optimizer='es',
            learning_rate=0.01,
            pop_size=4,
            verbose=False,
        )

        trainer = WeightTrainer(simple_genome, dummy_problem, config)
        result = trainer.fit(epochs=3)

        assert 'best_fitness' in result

    def test_weight_trainer_with_optimizer_instance(self, simple_genome, dummy_problem):
        """Test WeightTrainer with optimizer instance."""
        from wann_sdk import WeightTrainer, WeightTrainerConfig
        from wann_sdk.optimizers import AdamW

        opt = AdamW(learning_rate=0.005, weight_decay=0.01)
        config = WeightTrainerConfig(optimizer=opt, verbose=False)

        trainer = WeightTrainer(simple_genome, dummy_problem, config)
        result = trainer.fit(epochs=3)

        assert 'best_fitness' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
