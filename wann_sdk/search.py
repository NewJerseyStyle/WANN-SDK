"""
Architecture Search for WANN SDK

Implements Weight Agnostic Neural Network architecture search:
- Evolve network topology (nodes, connections, activations)
- Evaluate with shared weights across all connections
- Find architectures that perform well regardless of weight value

References:
    Gaier & Ha (2019) "Weight Agnostic Neural Networks"
    https://weightagnostic.github.io/
"""

import jax
import jax.numpy as jnp
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time

from .problem import Problem


@dataclass
class SearchConfig:
    """
    Configuration for WANN architecture search.

    Args:
        pop_size: Population size for evolution
        max_nodes: Maximum number of hidden nodes
        max_connections: Maximum number of connections
        activation_options: List of activation functions to search over
        weight_values: Shared weight values to evaluate (-2, -1, -0.5, 0.5, 1, 2)
        complexity_weight: Penalty for network complexity (0-1)
        mutation_rate: Probability of mutation
        add_node_rate: Probability of adding a node
        add_connection_rate: Probability of adding a connection
        change_activation_rate: Probability of changing activation

    Example:
        >>> config = SearchConfig(
        ...     pop_size=100,
        ...     max_nodes=20,
        ...     activation_options=['tanh', 'relu', 'sigmoid', 'sin'],
        ... )
    """
    # Population
    pop_size: int = 100
    elite_size: int = 10

    # Architecture constraints
    max_nodes: int = 50
    max_connections: int = 200
    activation_options: List[str] = field(
        default_factory=lambda: ['tanh', 'relu', 'sigmoid', 'sin', 'abs', 'square']
    )

    # WANN evaluation - shared weight values
    weight_values: List[float] = field(
        default_factory=lambda: [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
    )

    # Fitness
    complexity_weight: float = 0.1  # Penalty for complexity

    # Mutation rates
    mutation_rate: float = 0.8
    add_node_rate: float = 0.03
    add_connection_rate: float = 0.05
    change_activation_rate: float = 0.1
    change_weight_rate: float = 0.0  # 0 for true WANN (shared weights)

    # Other
    seed: int = 42
    verbose: bool = True


@dataclass
class NetworkGenome:
    """
    Genome representing a neural network architecture.

    Encodes:
    - Node genes: (node_id, type, activation)
      - type: 0=input, 1=hidden, 2=output
    - Connection genes: (in_node, out_node, enabled)
    """
    nodes: jnp.ndarray  # (num_nodes, 3) - id, type, activation_idx
    connections: jnp.ndarray  # (num_connections, 3) - in, out, enabled
    num_inputs: int
    num_outputs: int
    fitness: float = -float('inf')
    complexity: int = 0

    def copy(self) -> 'NetworkGenome':
        """Create a copy of this genome."""
        return NetworkGenome(
            nodes=self.nodes.copy(),
            connections=self.connections.copy(),
            num_inputs=self.num_inputs,
            num_outputs=self.num_outputs,
            fitness=self.fitness,
            complexity=self.complexity,
        )


class ArchitectureSearch:
    """
    WANN Architecture Search.

    Evolves neural network topologies using NEAT-like algorithm,
    evaluating with shared weights to find weight-agnostic architectures.

    Example:
        >>> problem = SupervisedProblem(x_train, y_train)
        >>> search = ArchitectureSearch(problem, SearchConfig(max_nodes=30))
        >>> best_genome = search.run(generations=100)
        >>> network = search.genome_to_network(best_genome)
    """

    def __init__(
        self,
        problem: Problem,
        config: Optional[SearchConfig] = None,
    ):
        """
        Initialize architecture search.

        Args:
            problem: Problem instance defining the task
            config: SearchConfig for search parameters
        """
        self.problem = problem
        self.config = config or SearchConfig()

        self.key = jax.random.PRNGKey(self.config.seed)

        # Build activation function map
        self._activations = self._build_activation_map()

        # Initialize population
        self.population: List[NetworkGenome] = []
        self.best_genome: Optional[NetworkGenome] = None
        self.history: List[Dict[str, float]] = []

        # Innovation tracking for NEAT
        self._innovation_number = 0
        self._innovation_history: Dict[Tuple[int, int], int] = {}

    def _build_activation_map(self) -> Dict[str, Callable]:
        """Build map of activation functions."""
        return {
            'tanh': jnp.tanh,
            'relu': jax.nn.relu,
            'sigmoid': jax.nn.sigmoid,
            'sin': jnp.sin,
            'cos': jnp.cos,
            'abs': jnp.abs,
            'square': lambda x: x ** 2,
            'identity': lambda x: x,
            'step': lambda x: jnp.where(x > 0, 1.0, 0.0),
            'gaussian': lambda x: jnp.exp(-x ** 2),
        }

    def _get_activation(self, idx: int) -> Callable:
        """Get activation function by index."""
        name = self.config.activation_options[idx % len(self.config.activation_options)]
        return self._activations.get(name, jnp.tanh)

    def _create_minimal_genome(self) -> NetworkGenome:
        """Create a minimal genome with just input-output connections."""
        num_inputs = self.problem.input_dim
        num_outputs = self.problem.output_dim

        # Node genes: input nodes + output nodes
        nodes = []
        for i in range(num_inputs):
            nodes.append([i, 0, 0])  # Input nodes (type=0)
        for i in range(num_outputs):
            nodes.append([num_inputs + i, 2, 0])  # Output nodes (type=2)

        nodes = jnp.array(nodes, dtype=jnp.float32)

        # Connection genes: connect each input to each output
        connections = []
        for i in range(num_inputs):
            for j in range(num_outputs):
                connections.append([i, num_inputs + j, 1])  # enabled=1

        connections = jnp.array(connections, dtype=jnp.float32)

        return NetworkGenome(
            nodes=nodes,
            connections=connections,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
        )

    def _initialize_population(self):
        """Initialize population with minimal genomes."""
        self.population = []
        for _ in range(self.config.pop_size):
            genome = self._create_minimal_genome()
            # Apply some random mutations to create diversity
            self.key, mutate_key = jax.random.split(self.key)
            genome = self._mutate(genome, mutate_key)
            self.population.append(genome)

    def _mutate(self, genome: NetworkGenome, key: jax.random.PRNGKey) -> NetworkGenome:
        """Apply mutations to a genome."""
        genome = genome.copy()

        k1, k2, k3, k4 = jax.random.split(key, 4)

        # Add node mutation
        if float(jax.random.uniform(k1)) < self.config.add_node_rate:
            genome = self._mutate_add_node(genome, k1)

        # Add connection mutation
        if float(jax.random.uniform(k2)) < self.config.add_connection_rate:
            genome = self._mutate_add_connection(genome, k2)

        # Change activation mutation
        if float(jax.random.uniform(k3)) < self.config.change_activation_rate:
            genome = self._mutate_activation(genome, k3)

        return genome

    def _mutate_add_node(self, genome: NetworkGenome, key: jax.random.PRNGKey) -> NetworkGenome:
        """Add a new node by splitting an existing connection."""
        if len(genome.connections) == 0:
            return genome

        if len(genome.nodes) >= self.config.max_nodes + genome.num_inputs + genome.num_outputs:
            return genome

        k1, k2 = jax.random.split(key)

        # Select a random enabled connection to split
        enabled_mask = genome.connections[:, 2] == 1
        if not jnp.any(enabled_mask):
            return genome

        enabled_indices = jnp.where(enabled_mask)[0]
        conn_idx = int(jax.random.choice(k1, enabled_indices))

        in_node = int(genome.connections[conn_idx, 0])
        out_node = int(genome.connections[conn_idx, 1])

        # Create new node
        new_node_id = int(jnp.max(genome.nodes[:, 0])) + 1
        activation_idx = int(jax.random.randint(k2, (), 0, len(self.config.activation_options)))

        new_node = jnp.array([[new_node_id, 1, activation_idx]])  # Hidden node (type=1)
        genome.nodes = jnp.concatenate([genome.nodes, new_node], axis=0)

        # Disable old connection
        genome.connections = genome.connections.at[conn_idx, 2].set(0)

        # Add two new connections
        new_connections = jnp.array([
            [in_node, new_node_id, 1],
            [new_node_id, out_node, 1],
        ])
        genome.connections = jnp.concatenate([genome.connections, new_connections], axis=0)

        return genome

    def _mutate_add_connection(self, genome: NetworkGenome, key: jax.random.PRNGKey) -> NetworkGenome:
        """Add a new connection between existing nodes."""
        if len(genome.connections) >= self.config.max_connections:
            return genome

        k1, k2 = jax.random.split(key)

        # Get valid source nodes (input or hidden)
        source_mask = genome.nodes[:, 1] != 2  # Not output
        target_mask = genome.nodes[:, 1] != 0  # Not input

        if not jnp.any(source_mask) or not jnp.any(target_mask):
            return genome

        source_indices = jnp.where(source_mask)[0]
        target_indices = jnp.where(target_mask)[0]

        source_node = int(genome.nodes[int(jax.random.choice(k1, source_indices)), 0])
        target_node = int(genome.nodes[int(jax.random.choice(k2, target_indices)), 0])

        # Check if connection already exists
        existing = (
            (genome.connections[:, 0] == source_node) &
            (genome.connections[:, 1] == target_node)
        )
        if jnp.any(existing):
            return genome

        # Add new connection
        new_connection = jnp.array([[source_node, target_node, 1]])
        genome.connections = jnp.concatenate([genome.connections, new_connection], axis=0)

        return genome

    def _mutate_activation(self, genome: NetworkGenome, key: jax.random.PRNGKey) -> NetworkGenome:
        """Change activation function of a hidden node."""
        hidden_mask = genome.nodes[:, 1] == 1  # Hidden nodes only
        if not jnp.any(hidden_mask):
            return genome

        k1, k2 = jax.random.split(key)

        hidden_indices = jnp.where(hidden_mask)[0]
        node_idx = int(jax.random.choice(k1, hidden_indices))

        new_activation = int(jax.random.randint(k2, (), 0, len(self.config.activation_options)))
        genome.nodes = genome.nodes.at[node_idx, 2].set(new_activation)

        return genome

    def _evaluate_genome(
        self,
        genome: NetworkGenome,
        key: jax.random.PRNGKey,
    ) -> Tuple[float, int]:
        """
        Evaluate genome with shared weights (WANN style).

        Tests the network with multiple fixed weight values and
        returns the mean fitness across all weight values.
        """
        fitness_list = []

        for weight_value in self.config.weight_values:
            key, eval_key = jax.random.split(key)

            # Create network with shared weight
            network = self._genome_to_network(genome, weight_value)

            # Evaluate
            fitness = self.problem.evaluate(network, eval_key)
            fitness_list.append(fitness)

        # Mean fitness across weight values (weight-agnostic evaluation)
        mean_fitness = float(jnp.mean(jnp.array(fitness_list)))

        # Compute complexity (number of enabled connections + hidden nodes)
        num_hidden = int(jnp.sum(genome.nodes[:, 1] == 1))
        num_enabled = int(jnp.sum(genome.connections[:, 2] == 1))
        complexity = num_hidden + num_enabled

        return mean_fitness, complexity

    def _genome_to_network(
        self,
        genome: NetworkGenome,
        shared_weight: float,
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """
        Convert genome to a network function with shared weight.

        Args:
            genome: NetworkGenome to convert
            shared_weight: Single weight value for all connections

        Returns:
            Network function: (input) -> output
        """
        # Build node lookup
        node_ids = genome.nodes[:, 0].astype(int)
        node_types = genome.nodes[:, 1].astype(int)
        node_activations = genome.nodes[:, 2].astype(int)

        # Sort nodes topologically (inputs -> hidden -> outputs)
        input_ids = node_ids[node_types == 0]
        hidden_ids = node_ids[node_types == 1]
        output_ids = node_ids[node_types == 2]

        # Build connection map
        enabled_conns = genome.connections[genome.connections[:, 2] == 1]

        def network(x: jnp.ndarray) -> jnp.ndarray:
            # Handle batched input
            if len(x.shape) == 1:
                x = x.reshape(1, -1)

            batch_size = x.shape[0]

            # Initialize node values
            node_values = {}

            # Set input values
            for i, nid in enumerate(input_ids):
                node_values[int(nid)] = x[:, i] if i < x.shape[1] else jnp.zeros(batch_size)

            # Process hidden nodes (simple iteration, assumes feedforward)
            for nid in hidden_ids:
                nid = int(nid)
                # Find incoming connections
                incoming = enabled_conns[enabled_conns[:, 1] == nid]
                if len(incoming) == 0:
                    node_values[nid] = jnp.zeros(batch_size)
                else:
                    total = jnp.zeros(batch_size)
                    for conn in incoming:
                        source_id = int(conn[0])
                        if source_id in node_values:
                            total = total + shared_weight * node_values[source_id]
                    # Apply activation
                    act_idx = int(node_activations[node_ids == nid][0])
                    activation = self._get_activation(act_idx)
                    node_values[nid] = activation(total)

            # Process output nodes
            outputs = []
            for nid in output_ids:
                nid = int(nid)
                incoming = enabled_conns[enabled_conns[:, 1] == nid]
                if len(incoming) == 0:
                    outputs.append(jnp.zeros(batch_size))
                else:
                    total = jnp.zeros(batch_size)
                    for conn in incoming:
                        source_id = int(conn[0])
                        if source_id in node_values:
                            total = total + shared_weight * node_values[source_id]
                    outputs.append(total)

            return jnp.stack(outputs, axis=-1)

        return network

    def _select_parents(self, key: jax.random.PRNGKey) -> List[NetworkGenome]:
        """Select parents using tournament selection."""
        parents = []
        tournament_size = 3

        for _ in range(self.config.pop_size - self.config.elite_size):
            key, select_key = jax.random.split(key)
            indices = jax.random.choice(
                select_key,
                len(self.population),
                shape=(tournament_size,),
                replace=False,
            )
            tournament = [self.population[int(i)] for i in indices]
            winner = max(tournament, key=lambda g: g.fitness)
            parents.append(winner)

        return parents

    def run(
        self,
        generations: int = 100,
        log_interval: int = 10,
    ) -> NetworkGenome:
        """
        Run architecture search.

        Args:
            generations: Number of generations to evolve
            log_interval: How often to log progress

        Returns:
            Best genome found
        """
        # Initialize
        self._initialize_population()
        self.problem.setup()

        if self.config.verbose:
            print(f"WANN Architecture Search")
            print(f"Population: {self.config.pop_size}")
            print(f"Max nodes: {self.config.max_nodes}")
            print(f"Weight values: {self.config.weight_values}")
            print("-" * 60)

        start_time = time.time()

        for gen in range(generations):
            # Evaluate population
            for genome in self.population:
                self.key, eval_key = jax.random.split(self.key)
                fitness, complexity = self._evaluate_genome(genome, eval_key)

                # Apply complexity penalty
                adjusted_fitness = fitness - self.config.complexity_weight * complexity
                genome.fitness = adjusted_fitness
                genome.complexity = complexity

            # Sort by fitness
            self.population.sort(key=lambda g: g.fitness, reverse=True)

            # Track best
            if self.best_genome is None or self.population[0].fitness > self.best_genome.fitness:
                self.best_genome = self.population[0].copy()

            # Log metrics
            fitnesses = [g.fitness for g in self.population]
            complexities = [g.complexity for g in self.population]
            metrics = {
                'generation': gen,
                'best_fitness': self.best_genome.fitness,
                'mean_fitness': float(jnp.mean(jnp.array(fitnesses))),
                'mean_complexity': float(jnp.mean(jnp.array(complexities))),
                'best_complexity': self.best_genome.complexity,
            }
            self.history.append(metrics)

            if self.config.verbose and (gen % log_interval == 0 or gen == generations - 1):
                elapsed = time.time() - start_time
                print(
                    f"Gen {gen:4d} [{elapsed:6.1f}s] | "
                    f"Best: {metrics['best_fitness']:8.4f} | "
                    f"Mean: {metrics['mean_fitness']:8.4f} | "
                    f"Complexity: {metrics['best_complexity']:3d}"
                )

            # Create next generation
            # Keep elites
            elites = [g.copy() for g in self.population[:self.config.elite_size]]

            # Select and mutate
            self.key, select_key = jax.random.split(self.key)
            parents = self._select_parents(select_key)

            offspring = []
            for parent in parents:
                self.key, mutate_key = jax.random.split(self.key)
                child = self._mutate(parent, mutate_key)
                offspring.append(child)

            self.population = elites + offspring

        self.problem.teardown()

        if self.config.verbose:
            print("-" * 60)
            print(f"Search completed in {time.time() - start_time:.1f}s")
            print(f"Best fitness: {self.best_genome.fitness:.4f}")
            print(f"Best complexity: {self.best_genome.complexity}")

        return self.best_genome

    def get_best_network(self, weight: float = 1.0) -> Callable:
        """
        Get the best network found.

        Args:
            weight: Shared weight value to use

        Returns:
            Network function
        """
        if self.best_genome is None:
            raise ValueError("No search has been run yet")
        return self._genome_to_network(self.best_genome, weight)

    def genome_to_network(self, genome: NetworkGenome, weight: float = 1.0) -> Callable:
        """Convert a genome to a network function."""
        return self._genome_to_network(genome, weight)
