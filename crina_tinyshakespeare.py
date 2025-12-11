# crina_tinyshakespeare.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import time
import os
import requests

# ---------------------- TinyShakespeare Dataset ----------------------
class TinyShakespeareDataset(Dataset):
    def __init__(self, data, block_size=128):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return x, y

# ---------------------- LIF Neuron (from your project) ----------------------
class LIFNeuron(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.threshold = nn.Parameter(torch.ones(size) * 1.0)
        self.tau_mem = 0.95
        self.tau_syn = 0.95
        self.reset()

    def reset(self):
        self.v = torch.zeros(1, self.size).cuda()
        self.s = torch.zeros(1, self.size).cuda()

    def forward(self, i_inj):
        # Expand states to batch size
        B = i_inj.shape[0]
        v = self.v.expand(B, -1)
        s = self.s.expand(B, -1)

        s = self.tau_syn * s + i_inj
        v = self.tau_mem * v + s - s.detach()  # detach synaptic current
        spike = (v >= self.threshold).float()
        v = v - spike * self.threshold

        # Save for next step (detached!)
        self.v = v.detach().mean(0, keepdim=True)
        self.s = s.detach().mean(0, keepdim=True)

        return spike

# ---------------------- TreeSelfAttention (our novel mechanism) ----------------------
class TreeSelfAttention(nn.Module):
    def __init__(self, d_model, sparsity_level=0.7, tree_depth=4):
        super().__init__()
        self.d_model = d_model
        self.tree_depth = tree_depth
        self.num_nodes = (1 << tree_depth) - 1  # e.g., 15 for depth=4

        self.node_projs = nn.ModuleList([
            nn.Linear(d_model * 2, d_model) for _ in range(self.num_nodes)
        ])
        self.lif_neurons = nn.ModuleList([
            LIFNeuron(d_model) for _ in range(self.num_nodes)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, D = x.shape
        
        # Compute effective tree size (cap at self.tree_depth)
        padded_T = 1 << math.ceil(math.log2(T))
        effective_depth = min(self.tree_depth, int(math.log2(padded_T)))
        num_leaves = 1 << effective_depth  # e.g., 16 for depth=4
        sub_seq = T // num_leaves  # e.g., 128 // 16 = 8
        
        # Pad x if needed
        if padded_T > T:
            x = F.pad(x, (0, 0, 0, padded_T - T))
        
        # Reshape to leaves: [B, num_leaves, sub_seq, D]
        leaves = x.view(B, num_leaves, sub_seq, D)
        
        # Mean-pool sub_seq within each leaf (to fit one vector per leaf)
        leaf_vectors = leaves.mean(dim=2)  # [B, num_leaves, D]
        
        # Tree node storage
        node_states = [None] * self.num_nodes
        leaf_start = (1 << (effective_depth - 1)) - 1  # e.g., 7 for depth=4
        
        # Process leaves
        for leaf_idx in range(num_leaves):
            global_idx = leaf_start + leaf_idx
            if global_idx >= self.num_nodes:  # Cap to allocated nodes
                continue
            leaf_vec = leaf_vectors[:, leaf_idx]  # [B, D]
            # Leaves have no children, so project directly (use identity-like for simplicity)
            proj = self.node_projs[global_idx](torch.cat([leaf_vec, leaf_vec], dim=-1))  # Dummy cat for consistency
            spike = self.lif_neurons[global_idx](proj)
            node_states[global_idx] = proj * spike

        # Process internal nodes (bottom-up)
        for level in range(effective_depth - 2, -1, -1):
            level_start = (1 << level) - 1
            level_size = 1 << level
            for local_idx in range(level_size):
                node_idx = level_start + local_idx
                if node_idx >= self.num_nodes:
                    continue
                left_idx = 2 * node_idx + 1
                right_idx = 2 * node_idx + 2
                left_out = node_states[left_idx] if left_idx < self.num_nodes and node_states[left_idx] is not None else torch.zeros_like(node_states[0])
                right_out = node_states[right_idx] if right_idx < self.num_nodes and node_states[right_idx] is not None else torch.zeros_like(node_states[0])
                fused_in = torch.cat([left_out, right_out], dim=-1)  # [B, 2*D]
                proj = self.node_projs[node_idx](fused_in)
                spike = self.lif_neurons[node_idx](proj)
                node_states[node_idx] = proj * spike

        # Root output (node 0)
        root_out = node_states[0] if node_states[0] is not None else torch.zeros(B, self.d_model, device=x.device)
        root_out = self.norm(root_out)
        
        # Broadcast root back to original seq_len (residual-style)
        output = root_out.unsqueeze(1).expand(-1, T, -1)  # [B, T, D]
        
        return output

# ---------------------- Crina-Synapse Block ----------------------
class CrinaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = TreeSelfAttention(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

# ---------------------- Full Model ----------------------
class CrinaSynapse(nn.Module):
    def __init__(self, vocab_size=65, d_model=256, n_layers=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([CrinaBlock(d_model) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

def reset_all_lif_neurons(model):
    for module in model.modules():
        if isinstance(module, LIFNeuron):
            module.reset()

# ---------------------- Training Loop ----------------------
def train():
    if not os.path.exists('tiny_shakespeare.txt'):
        print('Downloading TinyShakespeare dataset...')
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        response = requests.get(url)
        with open('tiny_shakespeare.txt', 'w') as f:
            f.write(response.text)

    # Load TinyShakespeare
    with open('tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    train_ds = TinyShakespeareDataset(train_data, block_size=128)
    val_ds = TinyShakespeareDataset(val_data, block_size=128)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    print(len(train_loader))

    model = CrinaSynapse(vocab_size=vocab_size, d_model=256, n_layers=8).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print(f"Model parameters: {sum([p.numel() for p in model.parameters()]):_}")

    for epoch in range(10):
        reset_all_lif_neurons(model)
        model.train()
        #print(list(enumerate(train_loader))[0])
        for i, xy in enumerate(train_loader):
            x, y = xy[0].cuda(), xy[1].cuda()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"Epoch {epoch} Iter {i} | Train loss: {loss.item():.4f}")
        print(f"Epoch {epoch} | Train loss: {loss.item():.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for x, y in val_loader:
                x, y = x.cuda(), y.cuda()
                logits = model(x)
                val_loss += F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), reduction='sum').item()
            val_loss /= len(val_data)
        print(f"Val loss: {val_loss:.4f}")

if __name__ == "__main__":
    train()