import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# Leaky Integrate and Fire Neuron
class LIFNeuron(nn.Module):
    def __init__(self, d_model):
        super(LIFNeuron, self).__init__()
        self.d_model = d_model
        self.threshold = nn.Parameter(torch.ones(d_model) * 1.0)
        self.tau = nn.Parameter(torch.ones(d_model) * 0.5)
        self.v_reset = nn.Parameter(torch.zeros(1, 1, d_model))
        #self.v = torch.zeros(1, 1, d_model).to(device)
        self.register_buffer("v", torch.zeros(1, 1, d_model))
        self.time_steps = 1
    
    def forward(self, x):
        spike_acc = torch.zeros_like(x)
        v = self.v # [B, 1, D]
        
        tau = self.tau
        threshold = self.threshold
        v_reset = self.v_reset
        
        for _ in range(self.time_steps):
            v = tau * v + x
            spike = (v >= threshold).float()
            v = v * (1.0 - spike) + v_reset * spike
            spike_acc += spike
        
        self.v = v[:, -1:, :].detach()
        spike_acc = spike_acc / self.time_steps
        norm = torch.sum(spike_acc, dim=-1, keepdim=True) + 1e-8
        return spike_acc / norm
    
    def reset(self):
        self.v = torch.zeros(1, 1, self.d_model).to(self.threshold.device)

class FeedForwardSNN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.lif = LIFNeuron(d_model * 4)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.lif(x)
        x = self.linear2(x)
        return x

class TestModel(nn.Module):
    def __init__(self, d_model, tree_depth=4):
        super().__init__()
        self.d_model = d_model
        self.tree_depth = tree_depth
        
        # We need projections for different phases
        # 1. Up-Sweep: S_parent = Merge(concat(S_left, S_right))
        self.up_projs = nn.ModuleList([nn.Linear(d_model * 2, d_model) for _ in range(tree_depth)])
        self.up_lifs = nn.ModuleList([LIFNeuron(d_model) for _ in range(tree_depth)])
        
        # 2. Down-Sweep: C_right = Distribute(concat(C_parent, S_left))
        # C_left inherits C_parent directly
        self.down_projs = nn.ModuleList([nn.Linear(d_model * 2, d_model) for _ in range(tree_depth)])
        self.down_lifs = nn.ModuleList([LIFNeuron(d_model) for _ in range(tree_depth)])
        
        # 3. Leaf mixing: output = Final(concat(x, C_leaf))
        self.leaf_proj = nn.Linear(d_model * 2, d_model)
        self.leaf_lif = LIFNeuron(d_model)

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        num_levels = int(math.log2(T))
        
        # --- PHASE 1: UP-SWEEP (Reduction) ---
        # summaries[l] stores the nodes at level l
        # summaries[0] is the leaves (tokens x)
        summaries = [x]
        for l in range(num_levels):
            curr = summaries[-l-1] if l == 0 else summaries[-1]
            # Pair nodes: [B, T/2^l, D] -> [B, T/2^(l+1), 2, D]
            T_curr = curr.shape[1]
            if T_curr < 2: break
            
            pairs = curr.view(B, T_curr // 2, 2, D)
            # Concat left and right children
            merged = pairs.reshape(B, T_curr // 2, 2 * D)
            
            proj_idx = min(l, self.tree_depth - 1)
            summary_next = self.up_lifs[proj_idx](self.up_projs[proj_idx](merged))
            summaries.append(summary_next)
        
        # --- PHASE 2: DOWN-SWEEP (Distribution) ---
        # contexts[l] stores the left-prefix context for nodes at level l
        # Start at the root (highest level) with 0 context
        root_ctx = torch.zeros(B, 1, D, device=x.device)
        contexts = [None] * (num_levels + 1)
        contexts[num_levels] = root_ctx
        
        for l in range(num_levels, 0, -1):
            parent_ctx = contexts[l] # [B, T/2^l, D]
            # summaries[l-1] is the children level: [B, T/2^(l-1), D]
            child_summaries = summaries[l-1].view(B, parent_ctx.shape[1], 2, D)
            left_child_S = child_summaries[:, :, 0, :] # [B, T/2^l, D]
            
            # Left child inherits parent context
            ctx_left = parent_ctx
            
            # Right child context = Distribute(concat(parent_ctx, left_child_S))
            combined = torch.cat([parent_ctx, left_child_S], dim=-1)
            proj_idx = min(l-1, self.tree_depth - 1)
            ctx_right = self.down_lifs[proj_idx](self.down_projs[proj_idx](combined))
            
            # Interleave left and right contexts to form the next level's contexts
            # [B, T/2^l, 2, D] -> [B, T/2^(l-1), D]
            curr_contexts = torch.stack([ctx_left, ctx_right], dim=2).view(B, -1, D)
            contexts[l-1] = curr_contexts
            
        # --- PHASE 3: LEAF INTEGRATION ---
        # contexts[0] is the prefix context for each individual token
        final_combined = torch.cat([x, contexts[0]], dim=-1)
        output = self.leaf_lif(self.leaf_proj(final_combined))
        
        return output

class TestLayer(nn.Module):
    def __init__(self, d_model, tree_depth):
        super().__init__()
        self.pre_norm_attn = nn.RMSNorm(d_model)
        self.attention = TestModel(d_model, tree_depth)
        self.post_norm_attn = nn.RMSNorm(d_model)
        self.pre_norm_ffn = nn.RMSNorm(d_model)
        self.feed_forward = FeedForwardSNN(d_model)
        self.post_norm_ffn = nn.RMSNorm(d_model)

        self.residual_scale_attn = nn.Parameter(torch.tensor(0.01))
        self.residual_scale_ffn = nn.Parameter(torch.tensor(0.01))
    
    def forward(self, x):
        x = x + self.residual_scale_attn * self.post_norm_attn(self.attention(self.pre_norm_attn(x)))
        x = x + self.residual_scale_ffn * self.post_norm_ffn(self.feed_forward(self.pre_norm_ffn(x)))
        return x

class TestCrina(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, tree_depth):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TestLayer(d_model, tree_depth) for _ in range(num_layers)])
        self.ln_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x

if __name__ == "__main__":
    B, T, D = 1, 256, 256
    model = TestCrina(256, 256, 6, 4).to(device)
    x = torch.randint(0, 256, (B, T)).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Warm up
    model.eval()
    with torch.no_grad():
        y = model(x)
        
    start = time.time()
    with torch.no_grad():
        for _ in range(5):
            y = model(x)
    end = time.time()
    
    print(f"Output shape: {y.shape}")
    print(f"Average inference time (batch=1, seq=256): {(end-start)/5:.4f}s")