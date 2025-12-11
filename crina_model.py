import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import prune
import torch.nn.functional as F
import json
import os
import sys
import math

# Hyperparameters
d_model = 256
n_heads = 8
n_layers = 6
dropout = 0.1
time_steps = 10
sparsity_level = 0.7
batch_size = 32

# Input sizes
text_seq_len = 1024
image_channels, image_size = 3, 32
audio_len = 128
video_frames, video_channels, video_size = 10, 3, 32
vocab_size = 30522

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Document class for multimodal output
class Document:
    def __init__(self, text, image, audio, video):
        self.text = text  # [batch, seq_len]
        self.image = image  # [batch, channels, height, width]
        self.audio = audio  # [batch, len]
        self.video = video  # [batch, channels, frames, height, width]

    def save(self, path):
        torch.save(self.__dict__, path)

    def to_json(self, path):
        data = {
            "text": self.text.tolist(),
            "image": self.image.tolist(),
            "audio": self.audio.tolist(),
            "video": self.video.tolist()
        }
        with open(path, 'w') as f:
            json.dump(data, f)

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
    def __init__(self, in_features, out_features, sparsity_level=sparsity_level, bias=True):
        super(SparseLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.sparsity_level = sparsity_level
        self.apply_pruning()

    def apply_pruning(self):
        prune.l1_unstructured(self.linear, name='weight', amount=self.sparsity_level)

    def forward(self, x):
        return self.linear(x)

class TreeCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, sparsity_level, tree_depth=4):
        super(TreeCrossAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.tree_depth = tree_depth
        self.head_dim = d_model // n_heads
        num_nodes = (1 << tree_depth) - 1
        self.node_projs = nn.ModuleList([SparseLinear(self.head_dim, self.head_dim, sparsity_level) for _ in range(num_nodes)])
        self.lif_neurons = nn.ModuleList([LIFNeuron(self.head_dim) for _ in range(num_nodes)])  # SNN per node
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key_value):  # query [batch, seq_q, d_model], key_value [batch, n_modalities, d_model]
        batch, seq_q, _ = query.shape
        _, n_modalities, _ = key_value.shape

        #print(seq_q, n_modalities)

        # Project query and key_value to heads
        q = query.view(batch, seq_q, self.n_heads, self.head_dim).transpose(1, 2)  # [batch, n_heads, seq_q, head_dim]
        kv = key_value.view(batch, n_modalities, self.n_heads, self.head_dim).transpose(1, 2)  # [batch, n_heads, n_modalities, head_dim]
        k, v = kv, kv  # Use key_value as both keys and values for cross-modal

        # Adjust tree_depth dynamically based on n_modalities
        effective_depth = min(self.tree_depth, (n_modalities-1).bit_length())
        num_leaves = 1 << effective_depth
        num_nodes = (1 << effective_depth) - 1

        # Pad n_modalities to num_leaves
        padded_n_mod = num_leaves
        if padded_n_mod > n_modalities:
            k = F.pad(k, (0, 0, 0, padded_n_mod - n_modalities), value=0.0)
            v = F.pad(v, (0, 0, 0, padded_n_mod - n_modalities), value=0.0)
            n_modalities = padded_n_mod

        # Reshape for leaves (sub_mod = 1 since n_modalities is the sequence)
        leaves_k = k.view(batch, self.n_heads, num_leaves, 1, self.head_dim)  # [batch, n_heads, num_leaves, 1, head_dim]
        leaves_v = v.view(batch, self.n_heads, num_leaves, 1, self.head_dim)  # Same for values

        # Iterative bottom-up processing for keys and values
        node_outputs_k = [None] * num_nodes
        node_outputs_v = [None] * num_nodes

        # Process leaves
        leaf_start = (1 << (effective_depth - 1)) - 1
        for leaf_local in range(num_leaves):
            leaf_global = leaf_start + leaf_local
            if leaf_global >= num_nodes:
                continue
            leaf_k = leaves_k[:, :, leaf_local, :, :]  # [batch, n_heads, 1, head_dim]
            leaf_v = leaves_v[:, :, leaf_local, :, :]  # [batch, n_heads, 1, head_dim]
            node_out_k = self.node_projs[leaf_global](leaf_k)  # [batch, n_heads, 1, head_dim]
            node_out_v = self.node_projs[leaf_global](leaf_v)  # [batch, n_heads, 1, head_dim]
            spike_k = self.lif_neurons[leaf_global](leaf_k)  # [batch, n_heads, head_dim]
            spike_v = self.lif_neurons[leaf_global](leaf_v)  # [batch, n_heads, head_dim]
            #print(node_out_k.shape)
            #print(spike_k.shape)
            #sys.exit()
            node_outputs_k[leaf_global] = node_out_k * spike_k  # Weighted by spike
            node_outputs_v[leaf_global] = node_out_v * spike_v

        # Bottom-up fusion
        for level in range(effective_depth - 2, -1, -1):
            level_start = (1 << level) - 1
            level_size = 1 << level
            for node_local in range(level_size):
                node_global = level_start + node_local
                left_child = 2 * node_global + 1
                right_child = 2 * node_global + 2
                if left_child >= num_nodes or right_child >= num_nodes or node_outputs_k[left_child] is None or node_outputs_k[right_child] is None:
                    continue
                left_out_k = node_outputs_k[left_child]
                right_out_k = node_outputs_k[right_child]
                left_out_v = node_outputs_v[left_child]
                right_out_v = node_outputs_v[right_child]
                #print(left_out_k.shape)
                fused_k = torch.cat([left_out_k, right_out_k], dim=2)  # [batch, n_heads, 2, head_dim]
                fused_v = torch.cat([left_out_v, right_out_v], dim=2)  # [batch, n_heads, 2, head_dim]
                fused_k = fused_k.mean(dim=2, keepdim=True)  # Average fusion
                fused_v = fused_v.mean(dim=2, keepdim=True)  # Average fusion
                node_out_k = self.node_projs[node_global](fused_k)
                node_out_v = self.node_projs[node_global](fused_v)
                spike_k = self.lif_neurons[node_global](torch.cat([left_out_k, right_out_k], dim=2).mean(dim=2))
                spike_v = self.lif_neurons[node_global](torch.cat([left_out_v, right_out_v], dim=2).mean(dim=2))
                node_outputs_k[node_global] = node_out_k * spike_k.unsqueeze(2)
                node_outputs_v[node_global] = node_out_v * spike_v.unsqueeze(2)

        # Root outputs
        tree_k = node_outputs_k[0]  # [batch, n_heads, 1, head_dim]
        tree_v = node_outputs_v[0]  # [batch, n_heads, 1, head_dim]

        # Query-tree attention
        attn_weights = torch.matmul(q, tree_k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch, n_heads, seq_q, 1]
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, tree_v)  # [batch, n_heads, seq_q, head_dim]
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_q, self.d_model)

        output = attn_out
        output = self.norm(output + query)  # Residual
        return output

class TreeHSNLayer(nn.Module):
    def __init__(self, d_model, n_heads, sparsity_level, tree_depth=4):
        super(TreeHSNLayer, self).__init__()
        self.d_model = d_model
        self.tree_depth = tree_depth
        num_nodes = (1 << tree_depth) - 1
        self.node_layers = nn.ModuleList([SparseLinear(d_model, d_model, sparsity_level) for _ in range(num_nodes)])
        self.lif_neurons = nn.ModuleList([LIFNeuron(d_model) for _ in range(num_nodes)])  # One per node
        self.fusion = TreeCrossAttention(d_model, n_heads, sparsity_level, tree_depth)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):  # x [batch, n_modalities, d_model]
        batch, n_modalities, _ = x.shape
        # Adjust tree_depth dynamically if n_modalities < 2^tree_depth
        effective_depth = min(self.tree_depth, (n_modalities-1).bit_length())
        num_leaves = 1 << effective_depth
        num_nodes = (1 << effective_depth) - 1
        # Pad n_modalities to num_leaves
        padded_n_mod = num_leaves
        if padded_n_mod > n_modalities:
            x = F.pad(x, (0, 0, 0, padded_n_mod - n_modalities), value=0.0)
            n_modalities = padded_n_mod
        sub_mod = 1  # Set to 1 since n_modalities is small
        leaves = x.view(batch, num_leaves, sub_mod, self.d_model)  # [batch, num_leaves, 1, d_model]
        
        # Iterative bottom-up processing
        node_outputs = [None] * num_nodes
        # Process leaves (bottom level, indices leaf_start to num_nodes-1)
        leaf_start = (1 << (effective_depth - 1)) - 1
        for leaf_local in range(num_leaves):
            leaf_global = leaf_start + leaf_local
            if leaf_global >= num_nodes:
                continue
            leaf = leaves[:, leaf_local, :, :]  # [batch, 1, d_model]
            node_out = self.node_layers[leaf_global](leaf)  # [batch, 1, d_model]
            spike = self.lif_neurons[leaf_global](node_out)  # Apply SNN spike
            node_outputs[leaf_global] = node_out * spike
        
        # Bottom-up fusion (internal nodes, from bottom level up)
        for level in range(effective_depth - 2, -1, -1):
            level_start = (1 << level) - 1
            level_size = 1 << level
            for node_local in range(level_size):
                node_global = level_start + node_local
                left_child = 2 * node_global + 1
                right_child = 2 * node_global + 2
                if left_child >= num_nodes or right_child >= num_nodes or node_outputs[left_child] is None or node_outputs[right_child] is None:
                    continue
                left_out = node_outputs[left_child]
                right_out = node_outputs[right_child]
                left_and_right_out = torch.cat([left_out, right_out], dim=1)
                fused = self.fusion(left_and_right_out, left_and_right_out)  # [batch, 1*2, d_model]
                fused = 0.5 * fused + 0.5 * left_and_right_out # Residual
                node_out = self.node_layers[node_global](fused)  # [batch, 1, d_model]
                spike = self.lif_neurons[node_global](node_out)  # SNN spike
                node_outputs[node_global] = node_out * spike
        
        tree_out = node_outputs[0]  # Root [batch, 1, d_model]
        return self.norm(tree_out.squeeze(1))  # [batch, d_model]

class TreeTextEmbed(nn.Module):
    def __init__(self, d_model, n_heads, sparsity_level, tree_depth=4, seq_len=1024, char_group_size=2, subword_group_size=4):
        super(TreeTextEmbed, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.tree_depth = tree_depth
        self.seq_len = seq_len
        self.char_group_size = char_group_size
        self.subword_group_size = subword_group_size
        num_nodes = (1 << tree_depth) - 1

        # 1. Character-level embedding
        self.id_proj = nn.Linear(1, d_model)
        self.pos_enc = self._generate_positional_encoding(seq_len, d_model)
        self.embed_norm = nn.LayerNorm(d_model)

        # 2. Hierarchical Fusion Layers
        self.subword_fusion_proj = SparseLinear(d_model, d_model, sparsity_level)
        self.subword_fusion_lif = LIFNeuron(d_model)
        self.subword_fusion_norm = nn.LayerNorm(d_model)

        self.word_fusion_proj = SparseLinear(d_model, d_model, sparsity_level)
        self.word_fusion_lif = LIFNeuron(d_model)
        self.word_fusion_norm = nn.LayerNorm(d_model)

        # 3. Tree-based Attention over words
        self.qkv_proj = SparseLinear(d_model, 3 * d_model, sparsity_level)
        self.node_projs = nn.ModuleList([SparseLinear(self.head_dim, self.head_dim, sparsity_level) for _ in range(num_nodes)])
        self.lif_neurons = nn.ModuleList([LIFNeuron(self.head_dim) for _ in range(num_nodes)])
        self.dropout = nn.Dropout(0.1)
        #self.output_proj = SparseLinear(d_model, d_model, sparsity_level)
        self.output_norm = nn.LayerNorm(d_model)

    def _generate_positional_encoding(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, seq_len, d_model]

    def _fuse_level(self, x, proj_layer, lif_neuron, norm_layer, group_size):
        """Helper to fuse sequences by a grouping factor."""
        batch, seq_len, dim = x.shape

        if seq_len % group_size != 0:
            pad_size = group_size - (seq_len % group_size)
            x = F.pad(x, (0, 0, 0, pad_size), "constant", 0)
            seq_len += pad_size

        x = x.view(batch, seq_len // group_size, group_size, dim)
        fused_x = x.mean(dim=2)

        projected_x = proj_layer(fused_x)
        spikes = lif_neuron(projected_x)
        gated_x = projected_x * spikes

        output = norm_layer(gated_x + fused_x)
        return output

    def forward(self, text_input):  # text_input [batch, seq_len]
        batch, seq_len = text_input.shape

        # 1. Character Embedding
        ids_float = text_input.float().unsqueeze(-1)  # [batch, seq_len, 1]
        id_embed = self.id_proj(ids_float)
        pos = self.pos_enc[:, :seq_len, :].to(device)
        char_embeddings = self.embed_norm(id_embed + pos)

        # 2. Hierarchical Fusion
        subword_reps = self._fuse_level(
            char_embeddings, self.subword_fusion_proj, self.subword_fusion_lif, self.subword_fusion_norm, self.char_group_size
        )
        word_reps = self._fuse_level(
            subword_reps, self.word_fusion_proj, self.word_fusion_lif, self.word_fusion_norm, self.subword_group_size
        )

        # 3. Tree-based Attention over Word Representations
        word_seq_len = word_reps.shape[1]
        qkv = self.qkv_proj(word_reps).view(batch, word_seq_len, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [batch, n_heads, seq_len, head_dim]

        # Pad word_seq_len to power of 2 for tree
        padded_word_seq_len = 1 << ((word_seq_len - 1).bit_length())
        if padded_word_seq_len > word_seq_len:
            pad_size = padded_word_seq_len - word_seq_len
            q = F.pad(q, (0, 0, 0, pad_size), value=0.0)
            k = F.pad(k, (0, 0, 0, pad_size), value=0.0)
            v = F.pad(v, (0, 0, 0, pad_size), value=0.0)

        # Tree traversal for keys and values
        tree_k = self._tree_traversal(k, self.node_projs, self.lif_neurons)  # [batch, n_heads, log n, head_dim]
        tree_v = self._tree_traversal(v, self.node_projs, self.lif_neurons)  # Same for values

        # Query-tree attention (log n paths)
        attn_weights = torch.matmul(q, tree_k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch, n_heads, seq_len, log n]
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, tree_v)  # [batch, n_heads, seq_len, head_dim]
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, padded_word_seq_len, self.d_model)

        # Final projection, residual, and norm
        #output = self.output_proj(attn_out)
        output = attn_out
        # Trim padding before residual connection
        output = output[:, :word_seq_len, :]
        output = self.output_norm(output + word_reps)

        # Upsample back to original sequence length for compatibility with the decoder
        output = F.interpolate(output.transpose(1, 2), size=self.seq_len, mode='linear', align_corners=False).transpose(1, 2)
        return output

    def _tree_traversal(self, seq, node_projs, lif_neurons):
        batch, n_heads, seq_len, head_dim = seq.shape
        effective_depth = min(self.tree_depth, int(math.log2(seq_len + 1)))
        num_leaves = 1 << effective_depth
        num_nodes = (1 << effective_depth) - 1

        # Reshape for leaves
        sub_seq = seq_len // num_leaves if seq_len // num_leaves > 0 else 1
        leaves = seq.view(batch, n_heads, num_leaves, sub_seq, head_dim)  # [batch, n_heads, num_leaves, sub_seq, head_dim]

        # Bottom-up
        node_outputs = [None] * num_nodes
        leaf_start = (1 << (effective_depth - 1)) - 1
        for leaf_local in range(num_leaves):
            leaf_global = leaf_start + leaf_local
            if leaf_global >= num_nodes:
                continue
            leaf = leaves[:, :, leaf_local, :, :]  # [batch, n_heads, sub_seq, head_dim]
            node_out = node_projs[leaf_global](leaf)  # [batch, n_heads, sub_seq, head_dim]
            spike = lif_neurons[leaf_global](node_out)  # [batch, n_heads, head_dim]
            node_outputs[leaf_global] = node_out * spike

        # Fusion
        for level in range(effective_depth - 2, -1, -1):
            level_start = (1 << level) - 1
            level_size = 1 << level
            for node_local in range(level_size):
                node_global = level_start + node_local
                left_child = 2 * node_global + 1
                right_child = 2 * node_global + 2
                if left_child >= num_nodes or right_child >= num_nodes or node_outputs[left_child] is None or node_outputs[right_child] is None:
                    continue
                left_out = node_outputs[left_child]
                right_out = node_outputs[right_child]
                fused = (left_out + right_out) / 2  # Average fusion
                node_out = node_projs[node_global](fused)
                spike = lif_neurons[node_global](node_out)
                node_outputs[node_global] = node_out * spike

        return node_outputs[0]  # [batch, n_heads, sub_seq, head_dim]

class TreeTextDecoder(nn.Module):
    def __init__(self, d_model, sparsity_level, tree_depth=4, seq_len=64):
        super(TreeTextDecoder, self).__init__()
        self.d_model = d_model
        self.tree_depth = tree_depth
        self.seq_len = seq_len
        num_nodes = (1 << tree_depth) - 1
        self.node_projs = nn.ModuleList([SparseLinear(d_model, d_model, sparsity_level) for _ in range(num_nodes)])
        self.lif_neurons = nn.ModuleList([LIFNeuron(d_model) for _ in range(num_nodes)])
        self.output_proj = nn.Linear(d_model, seq_len)  # Direct to seq_len for MSE
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, fused_x):  # fused_x [batch, d_model]
        batch = fused_x.shape[0]
        
        # Seed tree with repeated fused_x as leaves
        padded_seq = 1 << ((self.seq_len-1).bit_length())
        num_leaves = 1 << self.tree_depth
        sub_seq = padded_seq // num_leaves
        leaves = fused_x.unsqueeze(1).repeat(1, num_leaves, 1).unsqueeze(2).repeat(1, 1, sub_seq, 1)  # [batch, num_leaves, sub_seq, d_model]
        
        # Bottom-up tree generation
        node_outputs = [None] * ((1 << self.tree_depth) - 1)
        leaf_start = (1 << (self.tree_depth - 1)) - 1
        for leaf_local in range(num_leaves):
            leaf_global = leaf_start + leaf_local
            if leaf_global >= len(node_outputs):
                continue
            leaf = leaves[:, leaf_local, :, :]  # [batch, sub_seq, d_model]
            node_out = self.node_projs[leaf_global](leaf)  # [batch, sub_seq, d_model]
            spike = self.lif_neurons[leaf_global](node_out)  # [batch, d_model]
            node_outputs[leaf_global] = node_out * spike  # Gated [batch, sub_seq, d_model]
        
        # Bottom-up fusion
        for level in range(self.tree_depth - 2, -1, -1):
            level_start = (1 << level) - 1
            level_size = 1 << level
            for node_local in range(level_size):
                node_global = level_start + node_local
                left_child = 2 * node_global + 1
                right_child = 2 * node_global + 2
                if left_child >= len(node_outputs) or right_child >= len(node_outputs) or node_outputs[left_child] is None or node_outputs[right_child] is None:
                    continue
                left_out = node_outputs[left_child]
                right_out = node_outputs[right_child]
                fused = (left_out + right_out) / 2  # Average fusion
                node_out = self.node_projs[node_global](fused)
                spike = self.lif_neurons[node_global](node_out)
                node_outputs[node_global] = node_out * spike
        
        # Root output
        root_out = node_outputs[0]  # [batch, sub_seq, d_model]
        root_out = self.norm(root_out)
        
        output = self.output_proj(root_out.mean(dim=1))  # [batch, seq_len]
        output = F.layer_norm(output, output.shape) + output
        
        return output[:, :self.seq_len]  # Trim padding

# Crina-Synapse Model
class CrinaSynapse(nn.Module):
    def __init__(self, d_model=d_model, n_heads=n_heads, n_layers=n_layers, sparsity_level=sparsity_level, dropout=dropout):
        super(CrinaSynapse, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.sparsity_level = sparsity_level
        
        self.tree_hsn = TreeHSNLayer(d_model, n_heads, sparsity_level, tree_depth=2)
        
        self.cross_modal = TreeCrossAttention(d_model, n_heads, sparsity_level, tree_depth=2)
        self.sgf_tree = TreeHSNLayer(d_model, n_heads//2, sparsity_level, tree_depth=2)  # Small tree for n_mods=4
        
        #self.text_embed = SparseLinear(text_seq_len, d_model, sparsity_level)
        self.text_tree_embed = TreeTextEmbed(d_model, n_heads, sparsity_level, tree_depth=n_layers, seq_len=text_seq_len, char_group_size=2, subword_group_size=4)
        self.image_embed = nn.Conv2d(image_channels, d_model//16, kernel_size=3, padding=1)
        self.image_snn = LIFNeuron(d_model//16)
        self.audio_conv1d = nn.Conv1d(1, d_model//4, kernel_size=3, padding=1)  # Local features
        self.audio_snn_gate = LIFNeuron(d_model//4)  # Gating
        self.audio_pool1 = nn.AdaptiveAvgPool1d(audio_len//2)  # Pool to 64
        self.audio_pool2 = nn.AdaptiveAvgPool1d(audio_len//4)  # Pool to 32
        self.audio_scale_proj = nn.Linear(d_model//2, d_model)  # Concat scales to d_model
        self.video_embed = nn.Conv3d(video_channels, d_model//32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.video_snn = LIFNeuron(d_model//32)

        # Projection layers for common shape
        self.image_pooled_bottleneck = SparseLinear((image_size//2)**2 * (d_model//16), d_model//16, sparsity_level)
        self.image_pooled_proj = SparseLinear(d_model//16, d_model, sparsity_level)
        self.video_pooled_bottleneck = SparseLinear((video_size//2)**2 * video_frames//2 * (d_model//32), d_model//32, sparsity_level)
        self.video_pooled_proj = SparseLinear(d_model//32, d_model, sparsity_level)

        self.image_mod_scale = SparseLinear(d_model, d_model//16, sparsity_level)  # [batch, d_model//16]
        self.image_mod_bias = SparseLinear(d_model, d_model//16, sparsity_level)  # [batch, d_model//16]

        self.video_mod_scale = SparseLinear(d_model, d_model//32, sparsity_level)  # [batch, d_model//32]
        self.video_mod_bias = SparseLinear(d_model, d_model//32, sparsity_level)  # [batch, d_model//32]

        # Projection layers for recursive feedback
        #self.text_feedback_proj = nn.Linear(1, d_model)
        #self.image_feedback_proj = nn.Conv2d(image_channels, d_model // 16, kernel_size=1)
        #self.audio_feedback_proj = nn.Linear(1, d_model // 4)
        #self.video_feedback_proj = nn.Conv3d(video_channels, d_model // 32, kernel_size=1)

        self.text_tree_decoder = TreeTextDecoder(d_model, sparsity_level, tree_depth=n_layers, seq_len=text_seq_len)
        #self.text_output = SparseLinear(d_model, text_seq_len, sparsity_level)
        self.image_output = nn.ConvTranspose2d(d_model//16, image_channels, kernel_size=3, padding=1)
        #self.audio_output = SparseLinear(d_model, audio_len, sparsity_level)
        self.audio_deconv1 = nn.ConvTranspose1d(d_model, d_model//4, kernel_size=4, stride=2)  # Upsample to ~4
        #self.audio_deconv2 = nn.ConvTranspose1d(d_model//2, d_model//4, kernel_size=8, stride=4)  # Upsample to ~64
        self.audio_dilated = nn.Conv1d(d_model//4, d_model//4, kernel_size=3, padding=1)  # Long-range
        self.audio_snn_gate = LIFNeuron(d_model//4)
        self.audio_final = SparseLinear(d_model, audio_len, sparsity_level)
        self.video_output = nn.ConvTranspose3d(d_model//32, video_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def _forward_step(self, text_input, image_input, audio_input, video_input, batch_size):
        # Embed inputs
        text_emb = torch.zeros(batch_size, self.d_model, device=device) if text_input is None else self.text_tree_embed(text_input)

        image_features = torch.zeros(batch_size, self.d_model//16, image_size, image_size, device=device) if image_input is None else self.image_embed(image_input)
        image_features_pooled = F.adaptive_avg_pool2d(image_features, (image_size//2, image_size//2))
        image_features_flat = image_features_pooled.view(batch_size, -1)
        image_features_bottleneck = self.image_pooled_bottleneck(image_features_flat)
        image_features_bottleneck = F.layer_norm(image_features_bottleneck, image_features_bottleneck.shape) + image_features_bottleneck
        image_spike_map = self.image_snn(image_features_bottleneck)
        image_spike_map_expanded = image_spike_map.unsqueeze(2).unsqueeze(3)
        image_emb = image_features * image_spike_map_expanded
        image_bottleneck = image_features_bottleneck * image_spike_map
        image_bottleneck = F.layer_norm(image_bottleneck, image_bottleneck.shape) + image_bottleneck

        audio_reshaped = torch.zeros(batch_size, 1, audio_len, device=device) if audio_input is None else audio_input.unsqueeze(1)
        audio_conv = self.audio_conv1d(audio_reshaped)
        audio_spike = self.audio_snn_gate(audio_conv.mean(dim=2))
        audio_gated = audio_conv * audio_spike.unsqueeze(2)
        
        audio_pooled1 = self.audio_pool1(audio_gated)
        audio_pooled2 = self.audio_pool2(audio_gated)
        audio_flat = torch.cat([audio_pooled1.mean(dim=2), audio_pooled2.mean(dim=2)], dim=1)
        audio_emb = self.audio_scale_proj(audio_flat)

        video_features = torch.zeros(batch_size, self.d_model//32, video_frames, video_size, video_size, device=device) if video_input is None else self.video_embed(video_input)
        video_features_pooled = F.adaptive_avg_pool3d(video_features, (video_frames//2, video_size//2, video_size//2))
        video_features_flat = video_features_pooled.view(batch_size, -1)
        video_features_bottleneck = self.video_pooled_bottleneck(video_features_flat)
        video_features_bottleneck = F.layer_norm(video_features_bottleneck, video_features_bottleneck.shape) + video_features_bottleneck
        video_spike_map = self.video_snn(video_features_bottleneck)
        video_spike_map_expanded = video_spike_map.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        video_emb = video_features * video_spike_map_expanded
        video_bottleneck = video_features_bottleneck * video_spike_map
        video_bottleneck = F.layer_norm(video_bottleneck, video_bottleneck.shape) + video_bottleneck
        
        # Project for fusion
        text_fusion = text_emb.mean(dim=1) if text_input is not None else None
        image_fusion = self.image_pooled_proj(image_bottleneck) if image_input is not None else None
        audio_fusion = audio_emb if audio_input is not None else None
        video_fusion = self.video_pooled_proj(video_bottleneck) if video_input is not None else None
        
        # Stack fusion embeddings for HSN
        available_fusions = [f for f in [text_fusion, image_fusion, audio_fusion, video_fusion] if f is not None]
        if not available_fusions:
            fusion_emb = torch.zeros(batch_size, 1, self.d_model, device=device)
        else:
            fusion_emb = torch.stack(available_fusions, dim=1)
        
        x = self.tree_hsn(fusion_emb)
        x = self.cross_modal(x, x)
        x = self.sgf_tree(x)
        x = self.layer_norm(x)

        # --- Output Generation ---
        text_out = self.text_tree_decoder(x) if text_input is not None else torch.zeros(batch_size, text_seq_len, device=device)

        if image_input is not None:
            scale = self.image_mod_scale(x).unsqueeze(2).unsqueeze(3)
            bias = self.image_mod_bias(x).unsqueeze(2).unsqueeze(3)
            image_context = image_emb * scale + bias
            image_out = self.image_output(image_context)
        else:
            image_out = torch.zeros(batch_size, image_channels, image_size, image_size, device=device)

        if audio_input is not None:
            audio_context = x
            audio_deconv1 = self.audio_deconv1(audio_context.unsqueeze(-1))
            audio_dilated = self.audio_dilated(audio_deconv1)
            audio_spike = self.audio_snn_gate(audio_dilated.mean(dim=2))
            audio_gated = audio_dilated * audio_spike.unsqueeze(-1)
            audio_out = self.audio_final(audio_context + audio_gated.view(batch_size, -1))
        else:
            audio_out = torch.zeros(batch_size, audio_len, device=device)

        if video_input is not None:
            scale = self.video_mod_scale(x).unsqueeze(2).unsqueeze(3).unsqueeze(4)
            bias = self.video_mod_bias(x).unsqueeze(2).unsqueeze(3).unsqueeze(4)
            video_context = video_emb * scale + bias
            video_out = self.video_output(video_context)
        else:
            video_out = torch.zeros(batch_size, video_channels, video_frames, video_size, video_size, device=device)
        
        return Document(text_out, image_out, audio_out, video_out)

    def forward(self, text_input=None, image_input=None, audio_input=None, video_input=None, recursion_depth=1):
        # Handle empty inputs with placeholders
        batch_size = 1
        if text_input is not None and text_input.nelement() > 0:
            if text_input.dim() > 2:
                text_input = text_input.view(text_input.size(0), -1)
            batch_size = text_input.size(0)
            text_input = text_input.to(device)
        if image_input is not None:
            if image_input.dim() > 4:
                image_input = image_input.view(image_input.size(0), image_channels, image_size, image_size)
            batch_size = max(batch_size, image_input.size(0))
            image_input = image_input.to(device)
        if audio_input is not None:
            if audio_input.dim() > 2:
                audio_input = audio_input.view(audio_input.size(0), -1)
            batch_size = max(batch_size, audio_input.size(0))
            audio_input = audio_input.to(device)
        if video_input is not None:
            if video_input.dim() > 5:
                video_input = video_input.view(video_input.size(0), video_channels, video_frames, video_size, video_size)
            batch_size = max(batch_size, video_input.size(0))
            video_input = video_input.to(device)
        
        current_text, current_image, current_audio, current_video = text_input, image_input, audio_input, video_input

        # T-1 no-grad recursions for reasoning
        if recursion_depth > 1:
            with torch.no_grad():
                for _ in range(recursion_depth - 1):
                    doc = self._forward_step(current_text, current_image, current_audio, current_video, batch_size)
                    # Prepare inputs for the next recursive step
                    current_text = doc.text.detach()
                    current_image = doc.image.detach()
                    current_audio = doc.audio.detach()
                    current_video = doc.video.detach()

        # Final gradient-enabled recursion
        final_doc = self._forward_step(current_text, current_image, current_audio, current_video, batch_size)

        return final_doc