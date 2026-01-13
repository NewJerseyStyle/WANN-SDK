"""
Export WANN models to PyTorch for downstream fine-tuning.

Converts NetworkGenome and trained weights to a PyTorch nn.Module
that can be used in standard PyTorch training pipelines.
"""

import jax.numpy as jnp
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
import pickle
import json

from .search import NetworkGenome


def export_to_pytorch(
    genome: NetworkGenome,
    weights: jnp.ndarray,
    activation_options: List[str],
    output_path: str,
    include_code: bool = True,
) -> str:
    """
    Export WANN model to PyTorch.

    Creates a standalone PyTorch module file that can be imported
    and used for fine-tuning on downstream tasks.

    Args:
        genome: NetworkGenome from architecture search
        weights: Trained weights from WeightTrainer
        activation_options: Activation functions used in search
        output_path: Path to save the PyTorch model (.pt or .py)
        include_code: If True, generates a .py file with the model class

    Returns:
        Path to the exported model

    Example:
        >>> # After Stage 1 & 2
        >>> export_to_pytorch(
        ...     genome=best_genome,
        ...     weights=trainer.get_weights(),
        ...     activation_options=['tanh', 'relu', 'sigmoid'],
        ...     output_path='wann_model.py',
        ... )
        >>>
        >>> # In PyTorch
        >>> from wann_model import WANNModel
        >>> model = WANNModel()
        >>> model.load_state_dict(torch.load('wann_model.pt'))
    """
    output_path = Path(output_path)

    # Extract topology
    node_ids = genome.nodes[:, 0].astype(int).tolist()
    node_types = genome.nodes[:, 1].astype(int).tolist()
    node_activations = genome.nodes[:, 2].astype(int).tolist()

    input_ids = [nid for nid, t in zip(node_ids, node_types) if t == 0]
    hidden_ids = [nid for nid, t in zip(node_ids, node_types) if t == 1]
    output_ids = [nid for nid, t in zip(node_ids, node_types) if t == 2]

    # Get enabled connections
    enabled_mask = genome.connections[:, 2] == 1
    enabled_conns = genome.connections[enabled_mask]

    connections = []
    for i, conn in enumerate(enabled_conns):
        connections.append({
            'source': int(conn[0]),
            'target': int(conn[1]),
            'weight_idx': i,
        })

    # Build activation map
    act_map = {}
    for nid, act_idx in zip(node_ids, node_activations):
        if act_idx < len(activation_options):
            act_map[nid] = activation_options[act_idx]
        else:
            act_map[nid] = 'tanh'

    # Model config
    config = {
        'input_ids': input_ids,
        'hidden_ids': hidden_ids,
        'output_ids': output_ids,
        'connections': connections,
        'activations': act_map,
        'num_inputs': genome.num_inputs,
        'num_outputs': genome.num_outputs,
        'num_weights': len(weights),
    }

    # Convert weights to list for JSON serialization
    weights_list = weights.tolist()

    if include_code or output_path.suffix == '.py':
        # Generate Python file with PyTorch model class
        code = _generate_pytorch_code(config, weights_list)

        py_path = output_path.with_suffix('.py')
        with open(py_path, 'w') as f:
            f.write(code)

        print(f"PyTorch model exported to {py_path}")
        return str(py_path)

    else:
        # Save as pickle with config and weights
        data = {
            'config': config,
            'weights': weights_list,
        }
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"Model data exported to {output_path}")
        return str(output_path)


def _generate_pytorch_code(config: Dict, weights: List[float]) -> str:
    """Generate PyTorch model code."""

    # Build activation mapping code
    act_code_map = {
        'tanh': 'torch.tanh',
        'relu': 'F.relu',
        'sigmoid': 'torch.sigmoid',
        'sin': 'torch.sin',
        'cos': 'torch.cos',
        'abs': 'torch.abs',
        'square': 'lambda x: x ** 2',
        'identity': 'lambda x: x',
        'step': 'lambda x: (x > 0).float()',
        'gaussian': 'lambda x: torch.exp(-x ** 2)',
    }

    # Build connection structure code
    connections_by_target = {}
    for conn in config['connections']:
        target = conn['target']
        if target not in connections_by_target:
            connections_by_target[target] = []
        connections_by_target[target].append((conn['source'], conn['weight_idx']))

    code = f'''"""
WANN Model - Exported from WANN SDK

This is a Weight Agnostic Neural Network architecture discovered through
neuroevolution. The topology was found to work well across different
weight values.

Network Stats:
- Input nodes: {config['num_inputs']}
- Hidden nodes: {len(config['hidden_ids'])}
- Output nodes: {config['num_outputs']}
- Connections: {len(config['connections'])}
- Total parameters: {config['num_weights']}

Usage:
    >>> from {Path(__file__).stem} import WANNModel
    >>> model = WANNModel()
    >>> output = model(input_tensor)
    >>>
    >>> # Fine-tune with PyTorch
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    >>> for epoch in range(100):
    ...     output = model(x)
    ...     loss = criterion(output, y)
    ...     loss.backward()
    ...     optimizer.step()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class WANNModel(nn.Module):
    """
    Weight Agnostic Neural Network.

    Architecture discovered through neuroevolution with shared-weight evaluation.
    Can be fine-tuned using standard PyTorch training loops.
    """

    def __init__(self):
        super().__init__()

        # Network topology
        self.input_ids = {config['input_ids']}
        self.hidden_ids = {config['hidden_ids']}
        self.output_ids = {config['output_ids']}

        # Initialize weights as learnable parameters
        self.weights = nn.Parameter(torch.tensor({weights}, dtype=torch.float32))

        # Connection structure: target -> [(source, weight_idx), ...]
        self.connections: Dict[int, List[Tuple[int, int]]] = {{
'''

    # Add connections
    for target, sources in connections_by_target.items():
        code += f"            {target}: {sources},\n"

    code += '''        }

        # Activation functions for each node
        self.activations: Dict[int, callable] = {
'''

    # Add activations
    for nid in config['hidden_ids']:
        act_name = config['activations'].get(nid, 'tanh')
        act_code = act_code_map.get(act_name, 'torch.tanh')
        code += f"            {nid}: {act_code},\n"

    code += '''        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, input_dim)

        Returns:
            Output tensor of shape (batch, output_dim)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]

        # Node values dictionary
        node_values: Dict[int, torch.Tensor] = {}

        # Set input values
        for i, nid in enumerate(self.input_ids):
            if i < x.shape[1]:
                node_values[nid] = x[:, i]
            else:
                node_values[nid] = torch.zeros(batch_size, device=x.device)

        # Process hidden nodes
        for nid in self.hidden_ids:
            incoming = self.connections.get(nid, [])
            if not incoming:
                node_values[nid] = torch.zeros(batch_size, device=x.device)
            else:
                total = torch.zeros(batch_size, device=x.device)
                for source_id, weight_idx in incoming:
                    if source_id in node_values:
                        total = total + self.weights[weight_idx] * node_values[source_id]
                # Apply activation
                activation = self.activations.get(nid, torch.tanh)
                node_values[nid] = activation(total)

        # Process output nodes
        outputs = []
        for nid in self.output_ids:
            incoming = self.connections.get(nid, [])
            if not incoming:
                outputs.append(torch.zeros(batch_size, device=x.device))
            else:
                total = torch.zeros(batch_size, device=x.device)
                for source_id, weight_idx in incoming:
                    if source_id in node_values:
                        total = total + self.weights[weight_idx] * node_values[source_id]
                outputs.append(total)

        return torch.stack(outputs, dim=-1)

    def get_num_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Convenience function to create and load model
def load_wann_model() -> WANNModel:
    """Create a new WANNModel with discovered weights."""
    return WANNModel()


if __name__ == "__main__":
    # Test the model
    model = WANNModel()
    print(f"WANN Model loaded with {model.get_num_params()} parameters")

    # Test forward pass
    x = torch.randn(1, ''' + str(config['num_inputs']) + ''')
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
'''

    return code


def export_to_onnx(
    genome: NetworkGenome,
    weights: jnp.ndarray,
    activation_options: List[str],
    output_path: str,
) -> str:
    """
    Export WANN model to ONNX format.

    Args:
        genome: NetworkGenome from architecture search
        weights: Trained weights
        activation_options: Activation functions used
        output_path: Path to save .onnx file

    Returns:
        Path to exported ONNX model
    """
    try:
        import torch
        import torch.onnx
    except ImportError:
        raise ImportError("PyTorch is required for ONNX export. Install with: pip install torch")

    # First export to PyTorch
    py_path = export_to_pytorch(
        genome, weights, activation_options,
        output_path.replace('.onnx', '.py'),
        include_code=True,
    )

    # Import the generated module and export to ONNX
    import importlib.util
    spec = importlib.util.spec_from_file_location("wann_model", py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.WANNModel()
    model.eval()

    # Dummy input
    dummy_input = torch.randn(1, genome.num_inputs)

    # Export
    output_path = Path(output_path).with_suffix('.onnx')
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch'},
            'output': {0: 'batch'},
        },
    )

    print(f"ONNX model exported to {output_path}")
    return str(output_path)
