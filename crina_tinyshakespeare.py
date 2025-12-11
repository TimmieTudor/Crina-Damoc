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
        self.threshold = nn.Parameter(torch.ones(size) * 1.0)
        self.tau_mem = 0.95
        self.tau_syn = 0.95
        self.v = torch.zeros(1, size)
        self.s = torch.zeros(1, size)

    def forward(self, i_inj):
        self.s = self.tau_syn * self.s + i_inj
        self.v = self.tau_mem * self.v + self.s - self.s.detach()
        spike = (self.v >= self.threshold).float()
        self.v = self.v - spike * self.threshold
        return spike

# ---------------------- TreeSelfAttention (our novel mechanism) ----------------------
class TreeSelfAttention(nn.Module):
    def __init__(self, d_model, sparsity_level=0.7, tree_depth=4):
        super().__init__()
        self.d_model = d_model
        self.tree_depth = tree_depth
        num_nodes = (1 << tree_depth) - 1

        self.node_projs = nn.ModuleList([
            nn.Linear(d_model * 2, d_model) for _ in range(num_nodes)
        ])
        self.lif_neurons = nn.ModuleList([
            LIFNeuron(d_model) for _ in range(num_nodes)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, D = x.shape
        padded_T = 1 << math.ceil(math.log2(T))
        if padded_T > T:
            x = F.pad(x, (0, 0, 0, padded_T - T))

        # Build tree bottom-up
        depth = int(math.log2(padded_T))
        leaves_per_node = padded_T // (1 << depth)
        leaves = x.view(B, 1 << depth, leaves_per_node, D).mean(dim=2)  # avg pool leaves

        node_states = [None] * ((1 << (depth + 1)) - 1)
        leaf_offset = (1 << depth) - 1

        # Leaves
        for i in range(1 << depth):
            node_states[leaf_offset + i] = leaves[:, i]

        # Internal nodes
        for d in range(depth - 1, -1, -1):
            for i in range(1 << d):
                node_id = (1 << d) - 1 + i
                left = node_states[2 * node_id + 1]
                right = node_states[2 * node_id + 2]
                fused = torch.cat([left, right], dim=-1)
                proj = self.node_projs[node_id](fused)
                spike = self.lif_neurons[node_id](proj)
                node_states[node_id] = proj * spike

        out = node_states[0]  # root
        out = out.unsqueeze(1).expand(-1, T, -1)
        return self.norm(out[:, :T])

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

    model = CrinaSynapse(vocab_size=vocab_size, d_model=256, n_layers=8).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(10):
        model.train()
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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

train()