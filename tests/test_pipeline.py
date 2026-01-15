"""
Tests for WANN SDK Two-Stage Pipeline

Quick integration tests based on examples.
Uses minimal parameters for fast execution in CI.
"""

import pytest
import jax
import jax.numpy as jnp


class TestQuickstart:
    """Tests based on quickstart.py example."""

    def test_xor_pipeline(self):
        """Test minimal XOR pipeline (quickstart equivalent)."""
        from wann_sdk import (
            ArchitectureSearch, SearchConfig,
            WeightTrainer, WeightTrainerConfig,
            Problem,
        )

        class XORProblem(Problem):
            def __init__(self):
                super().__init__(input_dim=2, output_dim=1)
                self.x = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.float32)
                self.y = jnp.array([[0], [1], [1], [0]], dtype=jnp.float32)

            def evaluate(self, network, key):
                pred = jax.nn.sigmoid(network(self.x))
                return -float(jnp.mean((pred - self.y) ** 2))

            def loss(self, network, key):
                pred = jax.nn.sigmoid(network(self.x))
                return jnp.mean((pred - self.y) ** 2)

        problem = XORProblem()

        # Stage 1: Quick search
        search_config = SearchConfig(
            pop_size=10,
            max_nodes=5,
            activation_options=['tanh', 'relu'],
            weight_values=[-1.0, 1.0],
            verbose=False,
        )

        search = ArchitectureSearch(problem, search_config)
        genome = search.run(generations=5)

        assert genome is not None
        assert genome.num_inputs == 2
        assert genome.num_outputs == 1

        # Stage 2: Quick training with Adam
        trainer_config = WeightTrainerConfig(
            optimizer='adam',
            learning_rate=0.05,
            verbose=False,
        )

        trainer = WeightTrainer(
            genome, problem, trainer_config,
            activation_options=search_config.activation_options,
        )
        result = trainer.fit(epochs=5)

        assert 'best_fitness' in result

        # Test forward pass
        network = trainer.get_network()
        predictions = network(problem.x)
        assert predictions.shape == (4, 1)


class TestCustomProblem:
    """Tests based on custom_problem.py example."""

    def test_regression_pipeline(self):
        """Test regression problem pipeline."""
        from wann_sdk import (
            ArchitectureSearch, SearchConfig,
            WeightTrainer, WeightTrainerConfig,
            Problem,
        )

        class RegressionProblem(Problem):
            def __init__(self):
                super().__init__(input_dim=1, output_dim=1)
                self.x = jnp.linspace(-jnp.pi, jnp.pi, 20).reshape(-1, 1)
                self.y = jnp.sin(self.x)

            def evaluate(self, network, key):
                pred = network(self.x)
                return -float(jnp.mean((pred - self.y) ** 2))

            def loss(self, network, key):
                pred = network(self.x)
                return jnp.mean((pred - self.y) ** 2)

        problem = RegressionProblem()

        # Stage 1
        search_config = SearchConfig(
            pop_size=10,
            max_nodes=5,
            activation_options=['tanh', 'sin'],
            verbose=False,
        )

        search = ArchitectureSearch(problem, search_config)
        genome = search.run(generations=5)

        assert genome is not None

        # Stage 2 with AdamW
        trainer_config = WeightTrainerConfig(
            optimizer='adamw',
            learning_rate=0.02,
            weight_decay=0.01,
            verbose=False,
        )

        trainer = WeightTrainer(
            genome, problem, trainer_config,
            activation_options=search_config.activation_options,
        )
        trainer.fit(epochs=5)

        # Test prediction
        network = trainer.get_network()
        pred = network(problem.x)
        assert pred.shape == problem.y.shape


class TestSupervisedProblem:
    """Tests using SupervisedProblem class."""

    def test_classification_pipeline(self, classification_data):
        """Test classification with SupervisedProblem."""
        from wann_sdk import (
            ArchitectureSearch, SearchConfig,
            WeightTrainer, WeightTrainerConfig,
            SupervisedProblem,
        )

        x, y = classification_data

        problem = SupervisedProblem(
            x, y,
            loss_fn='cross_entropy',
        )

        assert problem.input_dim == 4
        assert problem.output_dim == 2

        # Stage 1
        search_config = SearchConfig(
            pop_size=10,
            max_nodes=5,
            verbose=False,
        )

        search = ArchitectureSearch(problem, search_config)
        genome = search.run(generations=3)

        assert genome is not None

        # Stage 2
        trainer_config = WeightTrainerConfig(
            optimizer='adam',
            learning_rate=0.01,
            verbose=False,
        )

        trainer = WeightTrainer(genome, problem, trainer_config)
        result = trainer.fit(epochs=3)

        assert 'best_fitness' in result

    def test_supervised_with_validation(self, classification_data):
        """Test SupervisedProblem with validation split."""
        from wann_sdk import SupervisedProblem

        x, y = classification_data
        n_train = int(0.8 * len(x))

        problem = SupervisedProblem(
            x[:n_train], y[:n_train],
            x_val=x[n_train:], y_val=y[n_train:],
            loss_fn='cross_entropy',
        )

        assert problem.input_dim == 4
        assert problem.output_dim == 2


class TestDifferentOptimizers:
    """Test pipeline with different optimizers."""

    def test_es_optimizer(self, xor_data):
        """Test pipeline with ES optimizer."""
        from wann_sdk import (
            ArchitectureSearch, SearchConfig,
            WeightTrainer, WeightTrainerConfig,
            Problem,
        )

        x, y = xor_data

        class XORProblem(Problem):
            def __init__(self):
                super().__init__(input_dim=2, output_dim=1)
                self.x = x
                self.y = y

            def evaluate(self, network, key):
                pred = jax.nn.sigmoid(network(self.x))
                return -float(jnp.mean((pred - self.y) ** 2))

        problem = XORProblem()

        search = ArchitectureSearch(problem, SearchConfig(pop_size=5, verbose=False))
        genome = search.run(generations=3)

        # ES optimizer
        trainer_config = WeightTrainerConfig(
            optimizer='es',
            learning_rate=0.1,
            pop_size=8,
            noise_std=0.1,
            verbose=False,
        )

        trainer = WeightTrainer(genome, problem, trainer_config)
        result = trainer.fit(epochs=3)

        assert 'best_fitness' in result

    def test_sgd_optimizer(self, xor_data):
        """Test pipeline with SGD optimizer."""
        from wann_sdk import (
            ArchitectureSearch, SearchConfig,
            WeightTrainer, WeightTrainerConfig,
            Problem,
        )

        x, y = xor_data

        class XORProblem(Problem):
            def __init__(self):
                super().__init__(input_dim=2, output_dim=1)
                self.x = x
                self.y = y

            def evaluate(self, network, key):
                pred = jax.nn.sigmoid(network(self.x))
                return -float(jnp.mean((pred - self.y) ** 2))

            def loss(self, network, key):
                pred = jax.nn.sigmoid(network(self.x))
                return jnp.mean((pred - self.y) ** 2)

        problem = XORProblem()

        search = ArchitectureSearch(problem, SearchConfig(pop_size=5, verbose=False))
        genome = search.run(generations=3)

        # SGD optimizer
        trainer_config = WeightTrainerConfig(
            optimizer='sgd',
            learning_rate=0.1,
            verbose=False,
        )

        trainer = WeightTrainer(genome, problem, trainer_config)
        result = trainer.fit(epochs=3)

        assert 'best_fitness' in result


class TestExport:
    """Test export functionality."""

    def test_export_to_pytorch_code(self, simple_genome):
        """Test PyTorch code generation."""
        from wann_sdk.export import _generate_pytorch_code

        config = {
            'input_ids': [0, 1],
            'hidden_ids': [],
            'output_ids': [2],
            'connections': [
                {'source': 0, 'target': 2, 'weight_idx': 0},
                {'source': 1, 'target': 2, 'weight_idx': 1},
            ],
            'activations': {},
            'num_inputs': 2,
            'num_outputs': 1,
            'num_weights': 2,
        }
        weights = [1.0, 1.0]

        code = _generate_pytorch_code(config, weights)

        assert 'class WANNModel' in code
        assert 'def forward' in code


class TestTrainableNetwork:
    """Test TrainableNetwork class."""

    def test_trainable_network_forward(self, simple_genome):
        """Test TrainableNetwork forward pass."""
        from wann_sdk import TrainableNetwork

        network = TrainableNetwork(
            genome=simple_genome,
            activation_options=['tanh'],
            init_weight=1.0,
        )

        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        output = network(x)

        assert output.shape == (2, 1)

    def test_trainable_network_params(self, simple_genome):
        """Test TrainableNetwork parameter access."""
        from wann_sdk import TrainableNetwork

        network = TrainableNetwork(
            genome=simple_genome,
            activation_options=['tanh'],
        )

        num_params = network.num_params()
        assert num_params >= 0

        params = network.get_params()
        assert params.shape[0] == num_params

        # Set new params
        new_params = jnp.ones(num_params) * 0.5
        network.set_params(new_params)
        assert jnp.allclose(network.get_params(), new_params)


class TestNetworkGenome:
    """Test NetworkGenome class."""

    def test_genome_creation(self):
        """Test genome creation."""
        from wann_sdk import NetworkGenome

        nodes = jnp.array([[0, 0, 0], [1, 0, 0], [2, 2, 0]])
        connections = jnp.array([[0, 2, 1], [1, 2, 1]])

        genome = NetworkGenome(
            nodes=nodes,
            connections=connections,
            num_inputs=2,
            num_outputs=1,
        )

        assert genome.num_inputs == 2
        assert genome.num_outputs == 1

    def test_genome_copy(self):
        """Test genome copy."""
        from wann_sdk import NetworkGenome

        genome = NetworkGenome(
            nodes=jnp.zeros((3, 3)),
            connections=jnp.zeros((2, 3)),
            num_inputs=2,
            num_outputs=1,
            fitness=0.5,
        )

        copy = genome.copy()
        assert copy.fitness == genome.fitness
        assert copy.num_inputs == genome.num_inputs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
