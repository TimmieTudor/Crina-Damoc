import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import prune
import math
from typing import Dict, List, Optional, Union

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LIF Neuron with Adaptive Threshold (Fixed for Multi-Dim Inputs)
class LIFNeuron(nn.Module):
    def __init__(self, d_model):
        super(LIFNeuron, self).__init__()
        self.threshold = nn.Parameter(torch.ones(d_model) * 1.0)  # Trainable threshold
        self.tau = nn.Parameter(torch.ones(d_model) * 0.5)  # Trainable leak rate
        self.v_reset = nn.Parameter(torch.zeros(d_model))  # Trainable reset
        self.v = torch.zeros(d_model).to(device)  # Non-trainable state

    def forward(self, x):
        # Integrate: v = tau * v + x (simplified single step)
        v = self.tau * self.v + x
        spike = (v >= self.threshold).float()  # Spike
        v = v * (1 - spike) + self.v_reset * spike  # Reset
        self.v = v.detach()  # Detach for state
        
        # Surrogate for gradients
        if self.training:
            surrogate = F.sigmoid(25 * (v - self.threshold))  # Trainable slope if needed
            return surrogate * spike
        return spike
    
# Sparse Linear Layer
class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features, sparsity_level=0.3, bias=True):
        super(SparseLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.sparsity_level = sparsity_level
        self.apply_pruning()

    def apply_pruning(self):
        prune.l1_unstructured(self.linear, name='weight', amount=self.sparsity_level)

    def forward(self, x):
        return self.linear(x)

# SALM Simulator
class SALM:
    def __init__(self, tree_depth=4, d_model=64, sparsity_level=0.3):
        self.tree_depth = tree_depth
        self.d_model = d_model
        self.num_nodes = (1 << tree_depth) - 1
        self.node_layers = [SparseLinear(d_model, d_model, sparsity_level) for _ in range(self.num_nodes)]
        self.lif_neurons = [LIFNeuron(d_model) for _ in range(self.num_nodes)]
        self.lambda_terms = {}  # Store lambda terms per node (simplified as dict)

    def build_spiking_tree(self):
        """Build complete binary tree with (2^tree_depth - 1) nodes."""
        def _build(node_id, depth_left):
            if depth_left <= 0:
                return None
            node = {
                'id': node_id,
                'v': torch.zeros(self.d_model),
                'spike': torch.zeros(self.d_model),
                'tau': torch.ones(self.d_model) * 0.5,
                'threshold': torch.ones(self.d_model),
                'lambda': self.lambda_terms.get(node_id, {'type': 'var', 'var': f'v{node_id}'}),  # Default lambda
                'children': []
            }
            left = _build(2 * node_id + 1, depth_left - 1)
            right = _build(2 * node_id + 2, depth_left - 1)
            node['children'] = [left, right] if left or right else None
            return node
        
        return _build(0, self.tree_depth)

    def lambda_reduce(self, term, inputs):
        """Simplified lambda reduction (symbolic, e.g., Church encoding)."""
        if term['type'] == 'var':
            return inputs[0] if inputs else torch.zeros(self.d_model)
        elif term['type'] == 'apply':
            return self.lambda_reduce(term['func'], inputs) + self.lambda_reduce(term['arg'], inputs)
        elif term['type'] == 'abs':
            return self.lambda_reduce(term['body'], inputs)
        return torch.zeros(self.d_model)
    
    def get_node_at_id(self, tree, node_id):
        if tree['id'] == node_id:
            return tree
        elif tree['children'] is None:
            if tree['id'] == node_id:
                return tree
            else:
                return
        else:
            left = self.get_node_at_id(tree['children'][0], node_id)
            right = self.get_node_at_id(tree['children'][1], node_id)
            return left or right

    def propagate_spikes(self, tree, node_id):
        """Recursive spiking propagation from node_id."""
        node = self.get_node_at_id(tree, node_id)
        if node['children'] is None:  # Leaf
            return node['spike']
        
        # Integrate from children
        left_spike = self.propagate_spikes(tree, node['children'][0]['id']) if node['children'][0] else torch.zeros(self.d_model)
        right_spike = self.propagate_spikes(tree, node['children'][1]['id']) if node['children'][1] else torch.zeros(self.d_model)
        node['v'] = node['tau'] * node['v'] + left_spike + right_spike  # Leaky integration (assume tau param)
        
        # Spike if threshold exceeded
        node['spike'] = (node['v'] >= node['threshold']).float()
        
        # Augment with Lambda Reduction (on spike)
        if node['spike'].sum() > 0:
            reduced = self.lambda_reduce(node['lambda'], [left_spike, right_spike])
            node['v'] = node['v'] * (1 - node['spike']) + reduced  # Soft reset with reduction
        
        return node['spike']

    def compute(self, lambda_term, input_spikes):
        """Main SALM computation."""
        self.lambda_terms = {0: lambda_term}  # Assign lambda to root
        tree = self.build_spiking_tree()
        
        # Initialize leaves with input spikes
        leaves = self.get_leaves(tree)  # Helper to get leaf nodes
        for i, leaf in enumerate(leaves):
            leaf['v'] = input_spikes[i] if i < len(input_spikes) else torch.zeros(self.d_model)
            leaf['spike'] = (leaf['v'] >= leaf['threshold']).float()
        
        # Execute propagation from root
        output_spikes = self.propagate_spikes(tree, 0)
        
        return output_spikes

    def get_leaves(self, tree):
        """Helper to get leaf nodes."""
        leaves = []
        def _collect(node):
            if node is None:
                return
            if node['children'] is None:
                leaves.append(node)
                return
            _collect(node['children'][0])
            _collect(node['children'][1])
        _collect(tree)
        return leaves

# Example Usage
if __name__ == "__main__":
    salm = SALM(tree_depth=3, d_model=8, sparsity_level=0.3)
    lambda_term = {'type': 'apply', 'func': {'type': 'abs', 'var': 'x', 'body': {'type': 'var', 'var': 'x'}}, 'arg': {'type': 'var', 'var': 'y'}}
    input_spikes = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    output = salm.compute(lambda_term, input_spikes)
    print(f"Output spikes: {output}")