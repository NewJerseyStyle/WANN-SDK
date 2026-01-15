"""
Pytest configuration and shared fixtures for WANN SDK tests.
"""

import pytest
import jax
import jax.numpy as jnp


@pytest.fixture
def rng_key():
    """Provide a JAX random key."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def xor_data():
    """XOR dataset for testing."""
    x = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.float32)
    y = jnp.array([[0], [1], [1], [0]], dtype=jnp.float32)
    return x, y


@pytest.fixture
def classification_data(rng_key):
    """Small classification dataset for testing."""
    k1, k2 = jax.random.split(rng_key)
    n_samples = 50

    # Generate two clusters
    x1 = jax.random.normal(k1, (n_samples // 2, 4)) + jnp.array([2, 2, 0, 0])
    x2 = jax.random.normal(k2, (n_samples // 2, 4)) + jnp.array([-2, -2, 0, 0])
    x = jnp.concatenate([x1, x2], axis=0)
    y = jnp.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    return x, y


@pytest.fixture
def simple_genome():
    """Create a simple genome for testing."""
    from wann_sdk import NetworkGenome

    # Simple 2-input, 1-output genome
    nodes = jnp.array([
        [0, 0, 0],  # Input node
        [1, 0, 0],  # Input node
        [2, 2, 0],  # Output node (type 2 = output)
    ])
    connections = jnp.array([
        [0, 2, 1],  # Input 0 -> Output (active)
        [1, 2, 1],  # Input 1 -> Output (active)
    ])

    return NetworkGenome(
        nodes=nodes,
        connections=connections,
        num_inputs=2,
        num_outputs=1,
    )


@pytest.fixture
def simple_params(rng_key):
    """Simple neural network parameters for testing."""
    k1, k2 = jax.random.split(rng_key)
    return {
        'w1': jax.random.normal(k1, (4, 8)),
        'b1': jnp.zeros(8),
        'w2': jax.random.normal(k2, (8, 2)),
        'b2': jnp.zeros(2),
    }


@pytest.fixture
def simple_forward_fn():
    """Simple forward function for testing."""
    def forward(params, x):
        h = jax.nn.relu(x @ params['w1'] + params['b1'])
        return h @ params['w2'] + params['b2']
    return forward


class DummyProblem:
    """Minimal problem for quick tests."""

    def __init__(self, input_dim=2, output_dim=1):
        from wann_sdk import Problem

        class _Problem(Problem):
            def __init__(inner_self):
                super().__init__(input_dim=input_dim, output_dim=output_dim)
                inner_self.x = jnp.ones((4, input_dim))
                inner_self.y = jnp.zeros((4, output_dim))

            def evaluate(inner_self, network, key):
                pred = network(inner_self.x)
                return -float(jnp.mean(pred ** 2))

            def loss(inner_self, network, key):
                pred = network(inner_self.x)
                return jnp.mean(pred ** 2)

        self._problem = _Problem()

    @property
    def problem(self):
        return self._problem


@pytest.fixture
def dummy_problem():
    """Provide a dummy problem instance."""
    return DummyProblem().problem
