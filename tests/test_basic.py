"""
Tests for WANN SDK - Two-Stage Pipeline
"""

import pytest
import jax
import jax.numpy as jnp


class TestImports:
    """Test that all modules can be imported."""

    def test_import_main(self):
        """Test main module import."""
        import wann_sdk
        assert hasattr(wann_sdk, "__version__")
        assert wann_sdk.__version__ == "0.6.0"

    def test_import_stage1(self):
        """Test Stage 1 (Architecture Search) imports."""
        from wann_sdk import ArchitectureSearch, SearchConfig, NetworkGenome
        assert ArchitectureSearch is not None
        assert SearchConfig is not None
        assert NetworkGenome is not None

    def test_import_stage2(self):
        """Test Stage 2 (Weight Training) imports."""
        from wann_sdk import WeightTrainer, WeightTrainerConfig, TrainableNetwork
        assert WeightTrainer is not None
        assert WeightTrainerConfig is not None
        assert TrainableNetwork is not None

    def test_import_problem(self):
        """Test Problem imports."""
        from wann_sdk import Problem, SupervisedProblem, RLProblem
        assert Problem is not None
        assert SupervisedProblem is not None
        assert RLProblem is not None

    def test_import_export(self):
        """Test export imports."""
        from wann_sdk import export_to_pytorch, export_to_onnx
        assert export_to_pytorch is not None
        assert export_to_onnx is not None


class TestSearchConfig:
    """Test SearchConfig for Stage 1."""

    def test_default_config(self):
        """Test default configuration."""
        from wann_sdk import SearchConfig

        config = SearchConfig()
        assert config.pop_size == 100
        assert config.max_nodes == 50
        assert config.max_connections == 200
        assert len(config.activation_options) > 0
        assert len(config.weight_values) > 0

    def test_custom_config(self):
        """Test custom configuration."""
        from wann_sdk import SearchConfig

        config = SearchConfig(
            pop_size=50,
            max_nodes=20,
            activation_options=['tanh', 'relu'],
            weight_values=[-1.0, 1.0],
        )

        assert config.pop_size == 50
        assert config.max_nodes == 20
        assert config.activation_options == ['tanh', 'relu']
        assert config.weight_values == [-1.0, 1.0]


class TestWeightTrainerConfig:
    """Test WeightTrainerConfig for Stage 2."""

    def test_default_config(self):
        """Test default configuration."""
        from wann_sdk import WeightTrainerConfig

        config = WeightTrainerConfig()
        assert config.optimizer == 'adam'
        assert config.learning_rate == 0.01
        assert config.pop_size == 64

    def test_adamw_config(self):
        """Test AdamW configuration."""
        from wann_sdk import WeightTrainerConfig

        config = WeightTrainerConfig(
            optimizer='adamw',
            learning_rate=0.001,
            weight_decay=0.01,
        )

        assert config.optimizer == 'adamw'
        assert config.learning_rate == 0.001
        assert config.weight_decay == 0.01


class TestProblem:
    """Test Problem classes."""

    def test_supervised_problem(self):
        """Test SupervisedProblem."""
        from wann_sdk import SupervisedProblem

        x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = jnp.array([0, 1, 0])

        problem = SupervisedProblem(x, y, loss_fn='cross_entropy')

        assert problem.input_dim == 2
        assert problem.output_dim == 2  # 2 classes

    def test_custom_problem(self):
        """Test custom Problem."""
        from wann_sdk import Problem

        class MyProblem(Problem):
            def __init__(self):
                super().__init__(input_dim=4, output_dim=2)

            def evaluate(self, network, key):
                x = jnp.ones((1, 4))
                output = network(x)
                return -float(jnp.sum(output ** 2))

        problem = MyProblem()
        assert problem.input_dim == 4
        assert problem.output_dim == 2


class TestNetworkGenome:
    """Test NetworkGenome."""

    def test_create_genome(self):
        """Test creating a genome."""
        from wann_sdk import NetworkGenome

        nodes = jnp.array([
            [0, 0, 0],  # Input node
            [1, 0, 0],  # Input node
            [2, 2, 0],  # Output node
        ])
        connections = jnp.array([
            [0, 2, 1],  # Input 0 -> Output
            [1, 2, 1],  # Input 1 -> Output
        ])

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


class TestArchitectureSearch:
    """Test Stage 1: Architecture Search."""

    def test_search_init(self):
        """Test search initialization."""
        from wann_sdk import ArchitectureSearch, SearchConfig, Problem

        class DummyProblem(Problem):
            def __init__(self):
                super().__init__(input_dim=2, output_dim=1)

            def evaluate(self, network, key):
                return 0.0

        problem = DummyProblem()
        config = SearchConfig(pop_size=10, max_nodes=5)
        search = ArchitectureSearch(problem, config)

        assert search.problem is problem
        assert search.config.pop_size == 10

    def test_minimal_genome_creation(self):
        """Test minimal genome creation."""
        from wann_sdk import ArchitectureSearch, SearchConfig, Problem

        class DummyProblem(Problem):
            def __init__(self):
                super().__init__(input_dim=3, output_dim=2)

            def evaluate(self, network, key):
                return 0.0

        problem = DummyProblem()
        search = ArchitectureSearch(problem, SearchConfig(pop_size=5))

        genome = search._create_minimal_genome()

        # Should have 3 input + 2 output nodes
        assert len(genome.nodes) == 5
        assert genome.num_inputs == 3
        assert genome.num_outputs == 2


class TestWeightTrainer:
    """Test Stage 2: Weight Training."""

    def test_trainable_network(self):
        """Test TrainableNetwork."""
        from wann_sdk import TrainableNetwork, NetworkGenome

        # Simple genome: 2 inputs -> 1 output
        nodes = jnp.array([
            [0, 0, 0],  # Input
            [1, 0, 0],  # Input
            [2, 2, 0],  # Output
        ])
        connections = jnp.array([
            [0, 2, 1],
            [1, 2, 1],
        ])
        genome = NetworkGenome(nodes, connections, num_inputs=2, num_outputs=1)

        network = TrainableNetwork(
            genome=genome,
            activation_options=['tanh'],
            init_weight=1.0,
        )

        assert network.num_params() == 2

        # Test forward pass
        x = jnp.array([[1.0, 1.0]])
        output = network(x)
        assert output.shape == (1, 1)

    def test_weight_trainer_init(self):
        """Test WeightTrainer initialization."""
        from wann_sdk import WeightTrainer, WeightTrainerConfig, NetworkGenome, Problem

        class DummyProblem(Problem):
            def __init__(self):
                super().__init__(input_dim=2, output_dim=1)

            def evaluate(self, network, key):
                return 0.0

        nodes = jnp.array([[0, 0, 0], [1, 0, 0], [2, 2, 0]])
        connections = jnp.array([[0, 2, 1], [1, 2, 1]])
        genome = NetworkGenome(nodes, connections, num_inputs=2, num_outputs=1)

        problem = DummyProblem()
        config = WeightTrainerConfig(optimizer='adam', verbose=False)

        trainer = WeightTrainer(genome, problem, config)

        assert trainer.network.num_params() == 2


class TestListEnvironments:
    """Test environment listing."""

    def test_list_environments(self):
        """Test listing available environments."""
        from wann_sdk import list_environments

        envs = list_environments()
        assert isinstance(envs, dict)
        assert "humanoid" in envs
        assert "ant" in envs


class TestExport:
    """Test export functionality."""

    def test_pytorch_code_generation(self):
        """Test that PyTorch code can be generated."""
        from wann_sdk.export import _generate_pytorch_code

        config = {
            'input_ids': [0, 1],
            'hidden_ids': [2],
            'output_ids': [3],
            'connections': [
                {'source': 0, 'target': 2, 'weight_idx': 0},
                {'source': 1, 'target': 2, 'weight_idx': 1},
                {'source': 2, 'target': 3, 'weight_idx': 2},
            ],
            'activations': {2: 'tanh'},
            'num_inputs': 2,
            'num_outputs': 1,
            'num_weights': 3,
        }
        weights = [1.0, 1.0, 1.0]

        code = _generate_pytorch_code(config, weights)

        assert 'class WANNModel' in code
        assert 'def forward' in code
        assert 'nn.Module' in code  # Uses import torch.nn as nn


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_mini_pipeline(self):
        """Test a minimal two-stage pipeline."""
        from wann_sdk import (
            ArchitectureSearch, SearchConfig,
            WeightTrainer, WeightTrainerConfig,
            Problem,
        )

        class SimpleProblem(Problem):
            def __init__(self):
                super().__init__(input_dim=2, output_dim=1)
                self.x = jnp.array([[0, 0], [1, 1]], dtype=jnp.float32)
                self.y = jnp.array([[0], [1]], dtype=jnp.float32)

            def evaluate(self, network, key):
                pred = network(self.x)
                return -float(jnp.mean((pred - self.y) ** 2))

        problem = SimpleProblem()

        # Stage 1: Quick search
        search_config = SearchConfig(
            pop_size=5,
            max_nodes=3,
            weight_values=[-1.0, 1.0],
            verbose=False,
        )
        search = ArchitectureSearch(problem, search_config)
        genome = search.run(generations=3, log_interval=10)

        assert genome is not None
        assert genome.num_inputs == 2
        assert genome.num_outputs == 1

        # Stage 2: Quick training
        trainer_config = WeightTrainerConfig(
            optimizer='es',
            pop_size=5,
            verbose=False,
        )
        trainer = WeightTrainer(
            genome, problem, trainer_config,
            activation_options=search_config.activation_options,
        )
        trainer.fit(epochs=3)

        # Get network and test
        network = trainer.get_network()
        output = network(problem.x)
        assert output.shape == (2, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
