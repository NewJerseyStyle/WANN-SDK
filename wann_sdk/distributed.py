"""
Distributed Evolution with Ray

Provides multi-node parallel evaluation for large-scale architecture search.
Uses Ray for distributed computing across CPU/GPU clusters.

Cluster Setup:
    # Head node
    ray start --head --port=6379

    # Worker nodes
    ray start --address='<head-ip>:6379'

    # Or use Ray cluster launcher
    ray up cluster.yaml

See: https://docs.ray.io/en/latest/cluster/getting-started.html
"""

import jax
import jax.numpy as jnp
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

# Ray import with availability check
try:
    import ray
    from ray.util import ActorPool
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None

from .search import ArchitectureSearch, SearchConfig, NetworkGenome
from .problem import Problem


def require_ray():
    """Check Ray availability."""
    if not RAY_AVAILABLE:
        raise ImportError(
            "Ray is required for distributed training. "
            "Install with: pip install ray"
        )


def init_ray(
    address: Optional[str] = None,
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Initialize Ray runtime.

    Args:
        address: Ray cluster address ('auto' for existing cluster, None for local)
        num_cpus: Number of CPUs (local mode only)
        num_gpus: Number of GPUs (local mode only)
        **kwargs: Additional ray.init arguments

    Returns:
        Ray context info

    Example:
        >>> # Local mode
        >>> init_ray(num_cpus=8)

        >>> # Connect to cluster
        >>> init_ray(address='auto')

        >>> # Connect to specific address
        >>> init_ray(address='ray://<head-ip>:10001')
    """
    require_ray()

    if ray.is_initialized():
        return ray.cluster_resources()

    init_kwargs = {
        'ignore_reinit_error': True,
        **kwargs
    }

    if address:
        init_kwargs['address'] = address
    else:
        if num_cpus:
            init_kwargs['num_cpus'] = num_cpus
        if num_gpus:
            init_kwargs['num_gpus'] = num_gpus

    ray.init(**init_kwargs)
    return ray.cluster_resources()


def shutdown_ray():
    """Shutdown Ray runtime."""
    if RAY_AVAILABLE and ray.is_initialized():
        ray.shutdown()


# ============================================================
# Remote Functions
# ============================================================

def _create_evaluate_genome_remote():
    """Create Ray remote function for genome evaluation."""
    require_ray()

    @ray.remote
    def evaluate_genome_remote(
        genome_data: Dict,
        problem_class: type,
        problem_kwargs: Dict,
        weight_values: List[float],
        activation_options: List[str],
        seed: int,
    ) -> Tuple[float, int]:
        """
        Evaluate a genome on a remote worker.

        Args:
            genome_data: Serialized genome dict
            problem_class: Problem class to instantiate
            problem_kwargs: Arguments for problem constructor
            weight_values: Shared weight values for WANN evaluation
            activation_options: Activation function names
            seed: Random seed

        Returns:
            (fitness, complexity) tuple
        """
        # Reconstruct genome
        genome = NetworkGenome(
            nodes=jnp.array(genome_data['nodes']),
            connections=jnp.array(genome_data['connections']),
            num_inputs=genome_data['num_inputs'],
            num_outputs=genome_data['num_outputs'],
        )

        # Create problem instance
        problem = problem_class(**problem_kwargs)

        # Build activation map
        activations = _build_activation_map(activation_options)

        # Evaluate with shared weights
        key = jax.random.PRNGKey(seed)
        fitness_list = []

        for weight_value in weight_values:
            key, eval_key = jax.random.split(key)
            network = _genome_to_network(genome, weight_value, activation_options, activations)
            fitness = problem.evaluate(network, eval_key)
            fitness_list.append(fitness)

        mean_fitness = float(jnp.mean(jnp.array(fitness_list)))

        # Compute complexity
        num_hidden = int(jnp.sum(genome.nodes[:, 1] == 1))
        num_enabled = int(jnp.sum(genome.connections[:, 2] == 1))
        complexity = num_hidden + num_enabled

        return mean_fitness, complexity

    return evaluate_genome_remote


def _build_activation_map(activation_options: List[str]) -> Dict[str, Callable]:
    """Build activation function map."""
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


def _genome_to_network(
    genome: NetworkGenome,
    shared_weight: float,
    activation_options: List[str],
    activations: Dict[str, Callable],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Convert genome to network function with shared weight."""
    node_ids = genome.nodes[:, 0].astype(int)
    node_types = genome.nodes[:, 1].astype(int)
    node_activations = genome.nodes[:, 2].astype(int)

    input_ids = node_ids[node_types == 0]
    hidden_ids = node_ids[node_types == 1]
    output_ids = node_ids[node_types == 2]

    enabled_conns = genome.connections[genome.connections[:, 2] == 1]

    def get_activation(idx: int) -> Callable:
        name = activation_options[idx % len(activation_options)]
        return activations.get(name, jnp.tanh)

    def network(x: jnp.ndarray) -> jnp.ndarray:
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        batch_size = x.shape[0]
        node_values = {}

        for i, nid in enumerate(input_ids):
            node_values[int(nid)] = x[:, i] if i < x.shape[1] else jnp.zeros(batch_size)

        for nid in hidden_ids:
            nid = int(nid)
            incoming = enabled_conns[enabled_conns[:, 1] == nid]
            if len(incoming) == 0:
                node_values[nid] = jnp.zeros(batch_size)
            else:
                total = jnp.zeros(batch_size)
                for conn in incoming:
                    source_id = int(conn[0])
                    if source_id in node_values:
                        total = total + shared_weight * node_values[source_id]
                act_idx = int(node_activations[node_ids == nid][0])
                activation = get_activation(act_idx)
                node_values[nid] = activation(total)

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


# ============================================================
# Distributed Architecture Search
# ============================================================

class DistributedSearch(ArchitectureSearch):
    """
    Distributed WANN Architecture Search using Ray.

    Parallelizes genome evaluation across multiple workers/nodes
    for faster large-scale architecture search.

    Args:
        problem: Problem instance (must be serializable or use problem_class)
        config: SearchConfig for search parameters
        problem_class: Problem class for remote instantiation
        problem_kwargs: Arguments for problem constructor
        num_workers: Number of parallel workers (None = auto)

    Example:
        >>> # Local parallel search
        >>> from wann_sdk import DistributedSearch, SearchConfig, init_ray
        >>> from wann_sdk import SupervisedProblem
        >>>
        >>> init_ray(num_cpus=8)
        >>>
        >>> search = DistributedSearch(
        ...     problem_class=SupervisedProblem,
        ...     problem_kwargs={'x_train': x, 'y_train': y, 'loss_fn': 'mse'},
        ...     config=SearchConfig(pop_size=200, max_nodes=50),
        ... )
        >>> genome = search.run(generations=100)

        >>> # Multi-node cluster search
        >>> init_ray(address='auto')  # Connect to Ray cluster
        >>> search = DistributedSearch(...)
        >>> genome = search.run(generations=100)

    Cluster Setup:
        See: https://docs.ray.io/en/latest/cluster/getting-started.html

        # Head node
        ray start --head --port=6379

        # Worker nodes
        ray start --address='<head-ip>:6379'
    """

    def __init__(
        self,
        problem: Optional[Problem] = None,
        config: Optional[SearchConfig] = None,
        problem_class: Optional[type] = None,
        problem_kwargs: Optional[Dict] = None,
        num_workers: Optional[int] = None,
    ):
        require_ray()

        # For distributed, we need problem_class for remote instantiation
        if problem_class is None and problem is None:
            raise ValueError("Either problem or problem_class must be provided")

        self.problem_class = problem_class
        self.problem_kwargs = problem_kwargs or {}

        # Create local problem for initialization
        if problem is None:
            problem = problem_class(**self.problem_kwargs)

        super().__init__(problem, config)

        # Auto-detect workers
        if num_workers is None:
            resources = ray.cluster_resources()
            num_workers = int(resources.get('CPU', 4))
        self.num_workers = num_workers

        # Create remote function
        self._evaluate_remote = _create_evaluate_genome_remote()

    def _serialize_genome(self, genome: NetworkGenome) -> Dict:
        """Serialize genome for remote transfer."""
        return {
            'nodes': genome.nodes.tolist(),
            'connections': genome.connections.tolist(),
            'num_inputs': genome.num_inputs,
            'num_outputs': genome.num_outputs,
        }

    def _evaluate_population_distributed(
        self,
        population: List[NetworkGenome],
        base_seed: int,
    ) -> List[Tuple[float, int]]:
        """Evaluate population in parallel using Ray."""
        # Submit all evaluation tasks
        futures = []
        for i, genome in enumerate(population):
            future = self._evaluate_remote.remote(
                genome_data=self._serialize_genome(genome),
                problem_class=self.problem_class,
                problem_kwargs=self.problem_kwargs,
                weight_values=self.config.weight_values,
                activation_options=self.config.activation_options,
                seed=base_seed + i,
            )
            futures.append(future)

        # Gather results
        return ray.get(futures)

    def run(
        self,
        generations: int = 100,
        log_interval: int = 10,
    ) -> NetworkGenome:
        """
        Run distributed architecture search.

        Args:
            generations: Number of generations to evolve
            log_interval: How often to log progress

        Returns:
            Best genome found
        """
        self._initialize_population()
        self.problem.setup()

        if self.config.verbose:
            print(f"Distributed WANN Architecture Search")
            print(f"Population: {self.config.pop_size}")
            print(f"Workers: {self.num_workers}")
            print(f"Max nodes: {self.config.max_nodes}")
            print(f"Weight values: {self.config.weight_values}")
            print("-" * 60)

        start_time = time.time()

        for gen in range(generations):
            # Distributed evaluation
            self.key, eval_key = jax.random.split(self.key)
            base_seed = int(jax.random.randint(eval_key, (), 0, 2**30))

            results = self._evaluate_population_distributed(self.population, base_seed)

            # Apply results
            for genome, (fitness, complexity) in zip(self.population, results):
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
            elites = [g.copy() for g in self.population[:self.config.elite_size]]

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


# ============================================================
# Utility Functions
# ============================================================

def get_cluster_info() -> Dict[str, Any]:
    """
    Get Ray cluster information.

    Returns:
        Dictionary with cluster resources and node info
    """
    require_ray()

    if not ray.is_initialized():
        raise RuntimeError("Ray is not initialized. Call init_ray() first.")

    return {
        'resources': ray.cluster_resources(),
        'available_resources': ray.available_resources(),
        'nodes': ray.nodes(),
    }


def wait_for_workers(min_workers: int, timeout: float = 60.0) -> bool:
    """
    Wait for minimum number of workers to join cluster.

    Args:
        min_workers: Minimum number of workers required
        timeout: Maximum wait time in seconds

    Returns:
        True if enough workers joined, False if timeout
    """
    require_ray()

    start = time.time()
    while time.time() - start < timeout:
        resources = ray.cluster_resources()
        if resources.get('CPU', 0) >= min_workers:
            return True
        time.sleep(1.0)

    return False
