# crina_tinyshakespeare.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import math
import time
import os
import requests
from tqdm import tqdm

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
        self.threshold = nn.Parameter(torch.ones(size) * 0.5)
        self.tau_mem = 0.99
        self.register_buffer("v", torch.zeros(1, 1, size))

    def reset(self):
        self.v.zero_()

    def forward(self, i_inj):
        B, N, D = i_inj.shape

        v = self.v.expand(B, N, -1)

        v = self.tau_mem * v + i_inj * 5.0   # ← input scaling

        spike = (v >= self.threshold).float()
        v = v - spike * self.threshold

        # Save last state (no averaging over nodes)
        self.v = v.detach()[:, -1:, :]

        if self.training:
            surrogate = torch.sigmoid(25.0 * (v - self.threshold))
            spike = surrogate + spike.detach() - spike.detach()

        return spike

class ALIFNeuron(nn.Module):
    """
    Adaptive Leaky Integrate-and-Fire (ALIF) Neuron
    - Standard LIF dynamics
    - Adaptive threshold that increases with each spike (spike-frequency adaptation)
    - Fully vectorized, supports [B, N, D] shapes (e.g., for tree nodes)
    - Surrogate gradient for training
    """
    def __init__(self, size, threshold=0.5, tau_mem=0.99, tau_adapt=0.95, adapt_strength=0.1):
        super().__init__()
        self.size = size

        # Base parameters
        self.base_threshold = nn.Parameter(torch.ones(size) * threshold)
        self.adapt_strength = nn.Parameter(torch.ones(size) * adapt_strength)  # How much threshold rises per spike
        self.tau_mem = tau_mem
        self.tau_adapt = tau_adapt  # Decay rate of adaptation

        # Persistent states: [1, 1, size] for broadcasting
        self.register_buffer("v", torch.zeros(1, 1, size))        # Membrane potential
        self.register_buffer("adapt", torch.zeros(1, 1, size))    # Adaptation state

        # Surrogate gradient slope
        self.surrogate_slope = 25.0

    def reset(self):
        """Reset membrane and adaptation state"""
        self.v.zero_()
        self.adapt.zero_()

    def forward(self, i_inj):
        """
        i_inj: [B, N, size] or [B, size] — input current
        Returns: spike tensor same shape as i_inj
        """
        # Expand persistent states to input shape
        v = self.v.expand_as(i_inj)
        adapt = self.adapt.expand_as(i_inj)

        # Membrane update
        v = self.tau_mem * v + i_inj

        # Dynamic threshold = base + adaptation
        threshold = self.base_threshold + adapt

        # Hard spike
        spike_hard = (v >= threshold).float()

        if self.training:
            # Fast sigmoid surrogate for gradient
            surrogate = torch.sigmoid(self.surrogate_slope * (v - threshold))
            spike = surrogate + spike_hard - spike_hard.detach()
        else:
            spike = spike_hard

        # Reset / subtract
        v = v - spike * threshold

        # Adaptation update: rises on spike, decays slowly
        adapt = self.tau_adapt * adapt + self.adapt_strength * spike_hard

        # Save states (mean over batch & node dims)
        self.v = v.detach().mean(dim=[0, 1], keepdim=True)
        self.adapt = adapt.detach().mean(dim=[0, 1], keepdim=True)

        return spike

class FeedForwardSNN(nn.Module):
    def __init__(self, d_model, hidden_dim, sparsity_level=0.3):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.lif1 = ALIFNeuron(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.lif2 = ALIFNeuron(d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = self.lif1(x)
        x = self.fc2(x)
        x = self.lif2(x)
        return x

# ---------------------- TreeSelfAttention (our novel mechanism) ----------------------
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

class TreeSelfAttentionHybrid(nn.Module):
    def __init__(self, d_model, sparsity_level=0.7, tree_depth=4, chunk_size=64):
        super().__init__()
        self.d_model = d_model
        self.tree_depth = tree_depth
        self.chunk_size = chunk_size
        self.num_nodes = (1 << tree_depth) - 1

        self.node_projs = nn.ModuleList([
            nn.Linear(d_model * 2, d_model) for _ in range(self.num_nodes)
        ])
        self.lif_neurons = nn.ModuleList([
            LIFNeuron(d_model) for _ in range(self.num_nodes)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Persistent state for recurrent/chunkwise
        self.register_buffer("node_states", torch.zeros(1, self.num_nodes, d_model))
        self.register_buffer("leaf_buffer", torch.zeros(1, 1 << (tree_depth-1), d_model))
        self.register_buffer("step_counter", torch.zeros(1, dtype=torch.long))

    def reset(self):
        self.node_states.zero_()
        self.leaf_buffer.zero_()
        self.step_counter.zero_()

    def forward(self, x, mode="parallel"):
        """
        x: [B, T, d_model] or [B, 1, d_model] in recurrent mode
        mode: "parallel" | "recurrent" | "chunkwise"
        """
        if mode == "parallel":
            return self._forward_parallel(x)
        elif mode == "recurrent":
            return self._forward_recurrent(x)
        elif mode == "chunkwise":
            return self._forward_chunkwise(x)
        else:
            raise ValueError("mode must be parallel, recurrent, or chunkwise")

    def _forward_parallel(self, x):
        # Your existing parallel implementation (with mean-pool leaves)
        B, T, D = x.shape
        padded_T = 1 << math.ceil(math.log2(T))
        eff_depth = min(self.tree_depth, int(math.log2(padded_T)))
        num_leaves = 1 << eff_depth
        sub_seq = T // num_leaves

        if padded_T > T:
            x = F.pad(x, (0, 0, 0, padded_T - T))
        leaves = x.view(B, num_leaves, sub_seq, D).mean(dim=2)

        node_states = [None] * self.num_nodes
        leaf_start = (1 << (eff_depth - 1)) - 1

        # Leaves
        for i in range(num_leaves):
            gid = leaf_start + i
            if gid >= self.num_nodes: continue
            vec = leaves[:, i]
            proj = self.node_projs[gid](torch.cat([vec, vec], dim=-1))
            spike = self.lif_neurons[gid](proj)
            node_states[gid] = proj * spike

        # Internal nodes
        for level in range(eff_depth - 2, -1, -1):
            start = (1 << level) - 1
            size = 1 << level
            for i in range(size):
                nid = start + i
                if nid >= self.num_nodes: continue
                l = 2 * nid + 1
                r = 2 * nid + 2
                l_out = node_states[l] if l < self.num_nodes and node_states[l] is not None else torch.zeros(B, D, device=x.device)
                r_out = node_states[r] if r < self.num_nodes and node_states[r] is not None else torch.zeros(B, D, device=x.device)
                fused = torch.cat([l_out, r_out], dim=-1)
                proj = self.node_projs[nid](fused)
                spike = self.lif_neurons[nid](proj)
                node_states[nid] = proj * spike

        root = node_states[0] if node_states[0] is not None else torch.zeros(B, D, device=x.device)
        output = root.unsqueeze(1).expand(-1, T, -1)
        return self.norm(output[:, :T] + x[:, :T])

    def _forward_recurrent(self, x_t):
        # One token at a time
        B, _, D = x_t.shape
        leaf_idx = int(self.step_counter.item() % (1 << (self.tree_depth - 1)))
        self.leaf_buffer[:, leaf_idx] = x_t.squeeze(1)
        self.step_counter += 1

        # Rebuild tree from current leaf buffer
        return self._run_tree_from_leaves(self.leaf_buffer.clone())

    def _forward_chunkwise(self, x_chunk):
        # Process chunk in parallel, update persistent state
        B, T, D = x_chunk.shape
        assert T <= self.chunk_size, "Chunk size exceeded"

        # Pad chunk and run parallel
        padded = F.pad(x_chunk, (0, 0, 0, self.chunk_size - T))
        output = self._forward_parallel(padded)

        # Update persistent leaf buffer (sliding window)
        start_idx = int(self.step_counter.item() % (1 << (self.tree_depth - 1)))
        self.leaf_buffer[:, start_idx:start_idx + T] = x_chunk.mean(dim=1)  # mean-pool chunk into leaves
        self.step_counter += T

        return output[:, :T]

    def _run_tree_from_leaves(self, leaves):
        # Shared bottom-up logic used by recurrent/chunkwise
        B, num_leaves, D = leaves.shape
        node_states = self.node_states.clone()

        leaf_start = (1 << (self.tree_depth - 1)) - 1
        for i in range(num_leaves):
            gid = leaf_start + i
            if gid >= self.num_nodes: continue
            vec = leaves[:, i]
            proj = self.node_projs[gid](torch.cat([vec, vec], dim=-1))
            spike = self.lif_neurons[gid](proj)
            node_states[:, gid] = proj * spike

        for level in range(self.tree_depth - 2, -1, -1):
            start = (1 << level) - 1
            size = 1 << level
            for i in range(size):
                nid = start + i
                if nid >= self.num_nodes: continue
                l = 2 * nid + 1
                r = 2 * nid + 2
                l_out = node_states[:, l] if l < self.num_nodes else torch.zeros(B, D, device=leaves.device)
                r_out = node_states[:, r] if r < self.num_nodes else torch.zeros(B, D, device=leaves.device)
                fused = torch.cat([l_out, r_out], dim=-1)
                proj = self.node_projs[nid](fused)
                spike = self.lif_neurons[nid](proj)
                node_states[:, nid] = proj * spike

        self.node_states = node_states.detach()
        root = node_states[:, 0]
        return self.norm(root.unsqueeze(1))

class TreeSelfAttentionGPU(nn.Module):
    def __init__(self, d_model, tree_depth=4):
        super().__init__()
        self.d_model = d_model
        self.tree_depth = tree_depth
        self.num_nodes = (1 << tree_depth) - 1

        self.node_proj = nn.Linear(d_model * 2, d_model, bias=False)
        self.lif_levels = nn.ModuleList([ALIFNeuron(d_model) for _ in range(tree_depth)])
        self.node_weights = nn.Parameter(torch.randn(self.num_nodes, d_model) * 0.02)
        self.norm = nn.RMSNorm(d_model)

        # Proper topology
        parent, left, right = [], [], []
        for n in range(self.num_nodes):
            l = 2 * n + 1
            r = 2 * n + 2
            if l < self.num_nodes:
                parent.append(n)
                left.append(l)
                right.append(r if r < self.num_nodes else -1)
        self.register_buffer('parent_idx', torch.tensor(parent))
        self.register_buffer('left_idx', torch.tensor(left))
        self.register_buffer('right_idx', torch.tensor(right))

    def forward(self, x):
        B, T, D = x.shape
        device = x.device

        padded_T = 1 << math.ceil(math.log2(T))
        if padded_T > T:
            x = F.pad(x, (0, 0, 0, padded_T - T))

        eff_depth = min(self.tree_depth, int(math.log2(padded_T)))
        num_leaves = 1 << eff_depth
        sub_seq = padded_T // num_leaves

        leaves = x.view(B, num_leaves, sub_seq, D)
        leaves = leaves.mean(dim=2) if sub_seq > 1 else leaves.squeeze(2)

        node_states = torch.zeros(B, self.num_nodes, D, device=device)
        leaf_start = (1 << (eff_depth - 1)) - 1
        valid = min(num_leaves, self.num_nodes - leaf_start)
        node_states[:, leaf_start:leaf_start + valid] = leaves[:, :valid]

        # Vectorized bottom-up
        for level in range(eff_depth - 1, -1, -1):
            level_start = (1 << level) - 1
            level_end = min((1 << (level + 1)) - 1, self.num_nodes)
            nodes = torch.arange(level_start, level_end, device=device)
            if len(nodes) == 0: continue

            l_idx = 2 * nodes + 1
            r_idx = 2 * nodes + 2
            valid = l_idx < self.num_nodes

            l_states = node_states[:, l_idx[valid]]
            r_states = node_states[:, r_idx[valid]] if (r_idx[valid] < self.num_nodes).any() else torch.zeros_like(l_states)

            fused = torch.cat([l_states, r_states], dim=-1)
            proj = self.node_proj(fused)
            spike = self.lif_levels[level](proj)
            node_states[:, nodes[valid]] = proj * spike

        # Hierarchical mixture with depth bias
        weights = F.softmax(self.node_weights, dim=0)
        mixture = torch.einsum('nd,bnd->bd', weights, node_states)
        output = mixture.unsqueeze(1).expand(-1, T, -1)
        return self.norm(output + x[:, :T])[:, :T]

# ---------------------- Crina-Synapse Block ----------------------
class CrinaBlock(nn.Module):
    def __init__(self, d_model, tree_depth=4):
        super().__init__()
        self.pre_norm_attn = nn.RMSNorm(d_model)
        self.attn = TreeSelfAttentionGPU(d_model, tree_depth=tree_depth)
        self.post_norm_attn = nn.RMSNorm(d_model)
        
        self.pre_norm_ff = nn.RMSNorm(d_model)
        self.ff = FeedForwardSNN(d_model, hidden_dim=4 * d_model)
        self.post_norm_ff = nn.RMSNorm(d_model)

        # Learnable residual scaling factors (initialized <1.0 for stability)
        self.residual_scale_attn = nn.Parameter(torch.tensor(0.5))  # Start at 0.5
        self.residual_scale_ff = nn.Parameter(torch.tensor(0.5))    # Start at 0.5

    def forward(self, x):
        # Sandwich attention with scaled residual
        attn_out = self.post_norm_attn(self.attn(self.pre_norm_attn(x)))
        x = x + self.residual_scale_attn * attn_out
        
        # Sandwich FFN with scaled residual
        ff_out = self.post_norm_ff(self.ff(self.pre_norm_ff(x)))
        x = x + self.residual_scale_ff * ff_out
        
        return x

# ---------------------- Full Model ----------------------
class CrinaSynapse(nn.Module):
    def __init__(self, vocab_size=65, d_model=256, n_layers=8, tree_depth=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([CrinaBlock(d_model, tree_depth=tree_depth) for _ in range(n_layers)])
        self.ln_f = nn.RMSNorm(d_model)
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
        if isinstance(module, LIFNeuron) or isinstance(module, ALIFNeuron):
            module.reset()

class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr=1e-6, eta_min=1e-5):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

    def get_lr(self, epoch):
        default_lr = self.optimizer.defaults['lr']
        if epoch < self.warmup_epochs:
            # Linear warmup
            return (default_lr - self.warmup_start_lr) / self.warmup_epochs * epoch + self.warmup_start_lr
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return self.eta_min + 0.5 * (default_lr - self.eta_min) * (1 + math.cos(math.pi * progress))

    def step(self, epoch):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_lr(epoch)

# ---------------------- Training Loop ----------------------
def train():
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"
    print(f"Using device: {device}")
    
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

    num_epochs = 10
    max_train_iters = num_epochs * len(train_loader)
    warmup_iters = 100
    model = CrinaSynapse(vocab_size=vocab_size, d_model=256, n_layers=8, tree_depth=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = CosineWarmupScheduler(optimizer, warmup_start_lr=1e-4, warmup_epochs=warmup_iters, max_epochs=max_train_iters)

    print(f"Model parameters: {sum([p.numel() for p in model.parameters()]):_}")
    current_iter = 0
    for epoch in range(10):
        reset_all_lif_neurons(model)
        model.train()
        #print(list(enumerate(train_loader))[0])
        pbar = tqdm(train_loader, desc="Training...")
        for x, y in pbar:
            #x, y = x.cuda(), y.cuda()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})
            scheduler.step(current_iter)
            current_iter += 1
        print(f"Epoch {epoch} | Train loss: {loss.item():.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            pbar2 = tqdm(val_loader, desc="Validating...")
            for x, y in pbar2:
                #x, y = x.cuda(), y.cuda()
                logits = model(x)
                loss_item = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item()
                val_loss += loss_item
                pbar2.set_postfix({'val_loss': loss_item})
            val_loss /= len(val_loader)
        print(f"Val loss: {val_loss:.4f}")

if __name__ == "__main__":
    train()