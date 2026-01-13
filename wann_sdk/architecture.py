"""
Neural Network Architecture Specifications

Provides classes for defining, saving, and loading network architectures
found through WANN architecture search.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, Protocol, Tuple
from dataclasses import dataclass, field
import pickle
from pathlib import Path


class NetworkArchitecture(Protocol):
    """Protocol defining the interface for network architectures."""

    def forward(self, x: jnp.ndarray, params: Dict) -> jnp.ndarray:
        """Forward pass through the network."""
        ...

    def init_params(self, key: jax.random.PRNGKey) -> Dict:
        """Initialize network parameters."""
        ...

    def get_num_params(self) -> int:
        """Return the number of trainable parameters."""
        ...


@dataclass
class ArchitectureSpec:
    """
    Specification for a network architecture found through WANN search.

    Stores the network topology (nodes and connections) along with metadata
    about the search process.

    Args:
        nodes: Node configuration array
        connections: Connection configuration array
        num_inputs: Number of input nodes
        num_outputs: Number of output nodes
        num_hidden: Number of hidden nodes
        num_params: Number of trainable parameters
        search_fitness: Fitness achieved during architecture search
        search_complexity: Complexity measure from search
        activation_functions: Mapping of node IDs to activation function names
        metadata: Additional metadata about the architecture

    Example:
        >>> spec = ArchitectureSpec(
        ...     nodes=nodes_array,
        ...     connections=conns_array,
        ...     num_inputs=24,
        ...     num_outputs=4,
        ...     num_hidden=15,
        ...     num_params=87,
        ...     search_fitness=250.0,
        ...     search_complexity=87,
        ... )
        >>> spec.save("my_architecture.pkl")
    """

    nodes: jnp.ndarray
    connections: jnp.ndarray
    num_inputs: int
    num_outputs: int
    num_hidden: int
    num_params: int
    search_fitness: float
    search_complexity: float
    activation_functions: Dict[int, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> None:
        """
        Save architecture specification to file.

        Args:
            path: File path to save to
        """
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "nodes": self.nodes,
            "connections": self.connections,
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "num_hidden": self.num_hidden,
            "num_params": self.num_params,
            "search_fitness": self.search_fitness,
            "search_complexity": self.search_complexity,
            "activation_functions": self.activation_functions,
            "metadata": self.metadata,
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "ArchitectureSpec":
        """
        Load architecture specification from file.

        Args:
            path: File path to load from

        Returns:
            Loaded ArchitectureSpec instance
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        return cls(
            nodes=data["nodes"],
            connections=data["connections"],
            num_inputs=data["num_inputs"],
            num_outputs=data["num_outputs"],
            num_hidden=data["num_hidden"],
            num_params=data["num_params"],
            search_fitness=data["search_fitness"],
            search_complexity=data["search_complexity"],
            activation_functions=data.get("activation_functions", {}),
            metadata=data.get("metadata", {}),
        )

    def summary(self) -> str:
        """Return a summary string of the architecture."""
        return (
            f"ArchitectureSpec(\n"
            f"  inputs={self.num_inputs}, outputs={self.num_outputs}, "
            f"hidden={self.num_hidden}\n"
            f"  params={self.num_params}, fitness={self.search_fitness:.2f}\n"
            f")"
        )


class WANNArchitecture:
    """
    Trainable WANN architecture wrapper.

    Wraps an ArchitectureSpec to provide a trainable network with
    individual connection weights.

    Args:
        spec: Architecture specification
        genome: Optional WANNGenome instance for forward pass

    Example:
        >>> spec = ArchitectureSpec.load("architecture.pkl")
        >>> arch = WANNArchitecture(spec)
        >>> params = arch.init_params(jax.random.PRNGKey(0))
        >>> output = arch.forward(observation, params)
    """

    def __init__(self, spec: ArchitectureSpec, genome: Optional[Any] = None):
        self.spec = spec
        self.genome = genome

        # Store architecture info
        self.num_inputs = spec.num_inputs
        self.num_outputs = spec.num_outputs
        self.num_hidden = spec.num_hidden
        self.num_params = spec.num_params

    def init_params(self, key: jax.random.PRNGKey) -> Dict[str, jnp.ndarray]:
        """
        Initialize trainable parameters.

        Args:
            key: JAX random key

        Returns:
            Dictionary containing initialized weights
        """
        # Initialize weights for each connection
        weights = jax.random.normal(key, (self.num_params,)) * 0.1

        return {"weights": weights}

    def forward(self, x: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Forward pass through the network.

        Args:
            x: Input array
            params: Network parameters

        Returns:
            Network output
        """
        if self.genome is not None:
            # Use genome forward pass
            weights = params["weights"]
            conns_updated = self.spec.connections.at[:, 2].set(weights)
            transformed = self.genome.transform(None, self.spec.nodes, conns_updated)
            return self.genome.forward(None, transformed, x)
        else:
            # Fallback: simple MLP-like forward pass
            return self._simple_forward(x, params)

    def _simple_forward(
        self, x: jnp.ndarray, params: Dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """Simple fallback forward pass."""
        # Basic linear transformation as fallback
        weights = params["weights"]
        w = weights[: self.num_inputs * self.num_outputs].reshape(
            self.num_inputs, self.num_outputs
        )
        return jnp.tanh(x @ w)

    def get_num_params(self) -> int:
        """Return the number of trainable parameters."""
        return self.num_params

    def get_architecture_info(self) -> Dict[str, Any]:
        """Return architecture information."""
        return {
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "num_hidden": self.num_hidden,
            "num_params": self.num_params,
            "search_fitness": self.spec.search_fitness,
            "search_complexity": self.spec.search_complexity,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize architecture to dictionary."""
        return {
            "spec": {
                "nodes": self.spec.nodes,
                "connections": self.spec.connections,
                "num_inputs": self.spec.num_inputs,
                "num_outputs": self.spec.num_outputs,
                "num_hidden": self.spec.num_hidden,
                "num_params": self.spec.num_params,
                "search_fitness": self.spec.search_fitness,
                "search_complexity": self.spec.search_complexity,
                "activation_functions": self.spec.activation_functions,
                "metadata": self.spec.metadata,
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WANNArchitecture":
        """Deserialize architecture from dictionary."""
        spec_data = data["spec"]
        spec = ArchitectureSpec(
            nodes=spec_data["nodes"],
            connections=spec_data["connections"],
            num_inputs=spec_data["num_inputs"],
            num_outputs=spec_data["num_outputs"],
            num_hidden=spec_data["num_hidden"],
            num_params=spec_data["num_params"],
            search_fitness=spec_data["search_fitness"],
            search_complexity=spec_data["search_complexity"],
            activation_functions=spec_data.get("activation_functions", {}),
            metadata=spec_data.get("metadata", {}),
        )
        return cls(spec)
