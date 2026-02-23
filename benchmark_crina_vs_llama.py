# benchmark_crina_vs_llama.py
# Run with: python benchmark_crina_vs_llama.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import time
import math
import os
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from crina_tinyshakespeare import CrinaSynapse

# ---------------------- Dataset (TinyShakespeare) ----------------------
class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    def __len__(self): return len(self.data) - self.block_size
    def __getitem__(self, i): return self.data[i:i+self.block_size], self.data[i+1:i+self.block_size+1]

# ---------------------- LIF Neuron (your version) ----------------------
class LIFNeuron(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.threshold = nn.Parameter(torch.ones(size))
        self.tau_mem = 0.95
        self.tau_syn = 0.95

        # Persistent state: [1, 1, size] → can broadcast to [B, N_nodes, size]
        self.register_buffer("v", torch.zeros(1, 1, size))
        self.register_buffer("s", torch.zeros(1, 1, size))

    def reset(self):
        self.v.zero_()
        self.s.zero_()

    def forward(self, i_inj):
        # i_inj shape: [B, N_nodes, d_model] or [B, d_model] — works for both
        if len(i_inj.shape) == 2:
            v = self.v.expand_as(i_inj.unsqueeze(0))
            s = self.s.expand_as(i_inj.unsqueeze(0))
        else:
            v = self.v.expand_as(i_inj)
            s = self.s.expand_as(i_inj)

        s = self.tau_syn * s + i_inj
        v = self.tau_mem * v + s - s.detach()
        spike = (v >= self.threshold).float()
        v = v - spike * self.threshold

        if len(i_inj.shape) == 2:
            spike = spike.squeeze(0)

        # Save mean over batch + node dimensions
        if len(i_inj.shape) == 2:
            self.v = v.detach().mean(dim=0, keepdim=True)
            self.s = s.detach().mean(dim=0, keepdim=True)
        else:
            self.v = v.detach().mean(dim=[0,1], keepdim=True)
            self.s = s.detach().mean(dim=[0,1], keepdim=True)

        # Surrogate for gradients
        if self.training:
            surrogate = F.sigmoid(25 * (v - self.threshold))  # Trainable slope if needed
            return surrogate * spike
        return spike

# ---------------------- Your TreeSelfAttention ----------------------
class TreeSelfAttention(nn.Module):
    def __init__(self, d_model, sparsity_level=0.7, tree_depth=4):
        super().__init__()
        self.d_model = d_model
        self.tree_depth = tree_depth
        self.num_nodes = (1 << tree_depth) - 1

        self.node_projs = nn.ModuleList([
            nn.Linear(d_model * 2, d_model) for _ in range(self.num_nodes)
        ])
        self.lif_neurons = nn.ModuleList([
            LIFNeuron(d_model) for _ in range(self.num_nodes)
        ])

        # Learnable weights for mixing node outputs per position
        self.node_weights = nn.Parameter(torch.randn(self.num_nodes, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, D = x.shape
        device = x.device

        # --- Tree setup ---
        padded_T = 1 << math.ceil(math.log2(T))
        effective_depth = min(self.tree_depth, int(math.log2(padded_T)))
        num_leaves = 1 << effective_depth
        sub_seq = T // num_leaves

        if padded_T > T:
            x = F.pad(x, (0, 0, 0, padded_T - T))

        leaves = x.view(B, num_leaves, sub_seq, D).mean(dim=2)  # [B, leaves, D]

        node_states = [None] * self.num_nodes
        leaf_start = (1 << (effective_depth - 1)) - 1

        # --- Bottom-up spiking propagation ---
        for leaf_idx in range(num_leaves):
            gid = leaf_start + leaf_idx
            if gid >= self.num_nodes:
                continue
            vec = leaves[:, leaf_idx]
            proj = self.node_projs[gid](torch.cat([vec, vec], dim=-1))
            spike = self.lif_neurons[gid](proj)
            node_states[gid] = proj * spike

        # Process internal nodes (bottom-up)
        for level in range(effective_depth - 2, -1, -1):
            level_start = (1 << level) - 1
            level_size = 1 << level
            for i in range(level_size):
                nid = level_start + i
                if nid >= self.num_nodes:
                    continue
                left = 2 * nid + 1
                right = 2 * nid + 2
                l_out = node_states[left] if left < self.num_nodes and node_states[left] is not None else torch.zeros(B, D, device=device)
                r_out = node_states[right] if right < self.num_nodes and node_states[right] is not None else torch.zeros(B, D, device=device)
                fused = torch.cat([l_out, r_out], dim=-1)
                proj = self.node_projs[nid](fused)
                spike = self.lif_neurons[nid](proj)
                node_states[nid] = proj * spike

        # --- Per-position mixture using learned node weights ---
        # Stack all active node outputs: [B, num_active_nodes, D]
        active_states = torch.stack([s for s in node_states if s is not None], dim=1)  # [B, N, D]
        active_weights = self.node_weights[:active_states.shape[1]]  # [N, D]

        # Weighted sum per position (broadcast weights)
        weights = F.softmax(active_weights, dim=0)  # [N, D]
        mixture = torch.einsum('nd,bnd->bd', weights, active_states)  # [B, D]

        # Expand to sequence length
        output = mixture.unsqueeze(1).expand(-1, T, -1)  # [B, T, D]
        output = self.norm(output + x[:, :T])  # Residual + norm

        return output[:, :T]  # Trim padding

class TreeSelfAttentionGPU(nn.Module):
    """
    Fully vectorized, GPU-optimized version of your TreeSelfAttention
    - No Python loops
    - O(log T) depth
    - Per-position hierarchical mixture via learned node weights
    - LIF spiking preserved
    """
    def __init__(self, d_model, sparsity_level=0.7, tree_depth=4):
        super().__init__()
        self.d_model = d_model
        self.tree_depth = tree_depth
        self.num_nodes = (1 << tree_depth) - 1

        # Single shared projection (broadcast to all nodes)
        self.node_proj = nn.Linear(d_model * 2, d_model, bias=False)
        
        # Single shared LIF neuron — we broadcast it across all nodes
        self.lif = LIFNeuron(d_model)
        
        # Learnable mixing weights for all nodes
        self.node_weights = nn.Parameter(torch.randn(self.num_nodes, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

        # Pre-compute tree topology (parent → children mapping)
        parent = []
        left_child = []
        right_child = []
        for node in range(self.num_nodes):
            l = 2 * node + 1
            r = 2 * node + 2
            if l < self.num_nodes:
                parent.append(node)
                left_child.append(l)
                right_child.append(r if r < self.num_nodes else -1)
        
        self.register_buffer('parent_idx', torch.tensor(parent, dtype=torch.long))
        self.register_buffer('left_idx', torch.tensor(left_child, dtype=torch.long))
        self.register_buffer('right_idx', torch.tensor([r if r != -1 else 0 for r in right_child], dtype=torch.long))

    def forward(self, x):
        B, T, D = x.shape
        device = x.device

        # Pad to power of 2
        padded_T = 1 << math.ceil(math.log2(T))
        if padded_T > T:
            x = F.pad(x, (0, 0, 0, padded_T - T))

        # Effective tree depth
        eff_depth = min(self.tree_depth, int(math.log2(padded_T)))
        num_leaves = 1 << eff_depth
        sub_seq = padded_T // num_leaves

        # Mean-pool into leaves: [B, num_leaves, D]
        leaves = x.view(B, num_leaves, sub_seq, D).mean(dim=2)

        # Initialize node states: [B, num_nodes, D]
        node_states = torch.zeros(B, self.num_nodes, D, device=device)

        # === Set leaf states ===
        leaf_start = (1 << (eff_depth - 1)) - 1
        valid_leaves = min(num_leaves, self.num_nodes - leaf_start)
        node_states[:, leaf_start:leaf_start + valid_leaves] = leaves[:, :valid_leaves]

        # === Vectorized bottom-up propagation ===
        # We process levels from bottom to top using precomputed parent/child indices
        for level in range(eff_depth - 1, -1, -1):
            level_start = (1 << level) - 1
            level_end = min((1 << (level + 1)) - 1, self.num_nodes)
            nodes = torch.arange(level_start, level_end, device=device)

            if len(nodes) == 0:
                continue

            # Gather children
            left = 2 * nodes + 1
            right = 2 * nodes + 2
            valid_mask = left < self.num_nodes

            if not valid_mask.any():
                continue

            l_states = node_states[:, left[valid_mask]]
            r_states = node_states[:, right[valid_mask]]
            # Pad right if missing
            if l_states.shape[1] != r_states.shape[1]:
                pad = torch.zeros_like(l_states)
                r_states = torch.cat([r_states, pad], dim=1)[:, :l_states.shape[1]]

            fused = torch.cat([l_states, r_states], dim=-1)  # [B, N_level, 2*D]
            proj = self.node_proj(fused)  # [B, N_level, D]

            # Broadcast single LIF across all nodes in level
            spike = self.lif(proj)  # [B, N_level, D]
            gated = proj * spike

            node_states[:, nodes[valid_mask]] = gated

        # === Per-position hierarchical mixture ===
        # Use all computed node states
        active_states = node_states  # [B, num_nodes, D]
        weights = F.softmax(self.node_weights, dim=0)  # [num_nodes, D]

        # Weighted sum over all tree nodes
        mixture = torch.einsum('nd,bnd->bd', weights, active_states)  # [B, D]

        # Expand to full sequence
        output = mixture.unsqueeze(1).expand(-1, T, -1)
        output = self.norm(output + x[:, :T])

        return output[:, :T]

# ---------------------- Standard Llama-2-style Attention ----------------------
class LlamaAttention(nn.Module):
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        attn = attn.masked_fill(torch.tril(torch.ones(T,T,device=x.device))==0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        y = attn @ v
        y = y.transpose(1,2).contiguous().view(B,T,D)
        return self.norm(self.out_proj(y) + x)

# ---------------------- Simple Block ----------------------
class Block(nn.Module):
    def __init__(self, d_model, use_tree=False):
        super().__init__()
        self.attn = TreeSelfAttentionGPU(d_model) if use_tree else LlamaAttention(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
    def forward(self, x): 
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

# ---------------------- Full Model ----------------------
class Model(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=8, use_tree=False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, use_tree) for _ in range(n_layers)])
        self.ln_f = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks: x = block(x)
        return self.head(self.ln_f(x))

def reset_all_lif_neurons(model):
    for module in model.modules():
        if isinstance(module, LIFNeuron):
            module.reset()

# ---------------------- Training Function ----------------------
def run_experiment(use_tree: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load data
    if not os.path.exists("input.txt"):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        open("input.txt","wb").write(requests.get(url).content)
    text = open("input.txt","r").read()
    chars = sorted(list(set(text)))
    data = torch.tensor([chars.index(c) for c in text], dtype=torch.long)
    n = int(0.9*len(data))
    train_data, val_data = data[:n], data[n:]

    max_iters = 2110

    block_size = 128
    batch_size = 64
    train_ds = CharDataset(train_data, block_size)
    val_ds = CharDataset(val_data, block_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    if not use_tree: 
        model = Model(vocab_size=len(chars), use_tree=use_tree).to(device)
    else:
        model = CrinaSynapse(vocab_size=len(chars)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)

    losses = []

    print(f"\n{'='*20} {'Crina Tree' if use_tree else 'Llama-2 Style'} Attention {'='*20}")
    #reset_all_lif_neurons(model)
    for i in range(max_iters):
        model.train()
        x, y = next(iter(train_loader))
        #x, y = train_ds[np.random.randint(0, len(train_ds))]
        x, y = x.cuda(), y.cuda()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, len(chars)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % 20 == 0:
            print(f"Iter {i} | Train loss: {loss.item():.4f}")

        if i % 40 == 0:
            # Validation
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for j in range(25):
                    #x, y = val_ds[np.random.randint(0, len(val_ds))]
                    x, y = next(iter(val_loader))
                    x, y = x.cuda(), y.cuda()
                    logits = model(x)
                    loss_item = F.cross_entropy(logits.view(-1, len(chars)), y.view(-1)).item()
                    val_loss += loss_item
                val_loss /= 25
            print(f"Iter {i} | Val loss: {val_loss:.4f}")
            losses.append(val_loss)
    
    """for epoch in range(1):
        model.train()
        pbar = tqdm(train_loader, desc="Training...")
        for xb,yb in pbar:
            xb,yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, len(chars)), yb.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            pbar2 = tqdm(val_loader, desc="Validating...")
            for xb,yb in pbar2:
                xb,yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss_item = F.cross_entropy(logits.view(-1,len(chars)), yb.view(-1)).item()
                val_loss += loss_item
                pbar2.set_postfix({'val_loss': loss_item})
        val_loss /= len(val_loader)
        try:
            print(f"Epoch {epoch+1:2d} | Val loss: {val_loss:.4f} | Perplexity: {math.exp(val_loss):.2f}")
        except OverflowError:
            print(f"Epoch {epoch+1:2d} | Val loss: {val_loss:.4f} | Perplexity: too high to compute")
        losses.append(val_loss)"""

    return losses

if __name__ == "__main__":
    start_time = time.perf_counter()
    losses_llama = run_experiment(use_tree=False)  # Llama-2 style first
    losses_crina = run_experiment(use_tree=True)   # Then your TreeSelfAttention
    plt.title("Llama-2 vs Crina")
    plt.plot(losses_llama, label="Llama-2")
    plt.plot(losses_crina, label="Crina")
    plt.xlabel("Validation Checkpoints")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.savefig("llama_vs_crina.png")
    plt.show()
    end_time = time.perf_counter()
    total_time = end_time - start_time
    total_time_minutes = total_time / 60
    total_time_hours = total_time / 3600
    print(f"\nTotal benchmarking time: {total_time:.2f} seconds ({total_time_minutes:.2f} minutes / {total_time_hours:.2f} hours)")