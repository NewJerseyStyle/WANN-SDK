# Distributed Architecture Search

WANN SDK supports multi-node distributed evolution using [Ray](https://docs.ray.io/).

## Quick Start

```python
from wann_sdk import DistributedSearch, SearchConfig, init_ray, SupervisedProblem

# Initialize Ray (local)
init_ray(num_cpus=8)

# Run distributed search
search = DistributedSearch(
    problem_class=SupervisedProblem,
    problem_kwargs={'x_train': x, 'y_train': y, 'loss_fn': 'mse'},
    config=SearchConfig(pop_size=200),
)
genome = search.run(generations=100)
```

## Cluster Setup

### 1. Start Head Node
```bash
ray start --head --port=6379
```

### 2. Start Worker Nodes
```bash
# On each worker machine
ray start --address='<head-ip>:6379'
```

### 3. Connect from Code
```python
from wann_sdk import DistributedSearch, init_ray

# Connect to cluster
init_ray(address='auto')

# Search runs across all nodes
search = DistributedSearch(
    problem_class=MyProblem,
    problem_kwargs={...},
    config=SearchConfig(pop_size=1000),
)
genome = search.run(generations=100)
```

## API Reference

### `init_ray(address, num_cpus, num_gpus)`
Initialize Ray runtime.

| Param | Description |
|-------|-------------|
| `address` | `'auto'` for cluster, `None` for local |
| `num_cpus` | CPU count (local mode) |
| `num_gpus` | GPU count (local mode) |

### `DistributedSearch`
Parallel architecture search.

```python
DistributedSearch(
    problem_class=SupervisedProblem,  # Problem class (not instance)
    problem_kwargs={...},              # Args for problem constructor
    config=SearchConfig(...),          # Search configuration
    num_workers=None,                  # Auto-detect if None
)
```

### Utilities

```python
from wann_sdk import get_cluster_info, wait_for_workers, shutdown_ray

# Check cluster status
info = get_cluster_info()
print(info['resources'])

# Wait for workers
wait_for_workers(min_workers=8, timeout=60)

# Cleanup
shutdown_ray()
```

## Ray Documentation

- [Cluster Setup Guide](https://docs.ray.io/en/latest/cluster/getting-started.html)
- [Kubernetes Deployment](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)
- [AWS/GCP/Azure](https://docs.ray.io/en/latest/cluster/vms/index.html)
- [Slurm Integration](https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html)
