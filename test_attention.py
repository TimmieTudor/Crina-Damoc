import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Surrogate Gradient for Spiking Neurons ---
class SurrogateSpike(torch.autograd.Function):
    """
    Standard spike in forward, Fast Sigmoid in backward.
    Includes a gain factor to compensate for gradient decay.
    """
    @staticmethod
    def forward(ctx, input, threshold, alpha=5.0, gain=1.0):
        ctx.save_for_backward(input, threshold)
        ctx.alpha = alpha
        ctx.gain = gain
        return (input >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, threshold = ctx.saved_tensors
        alpha = ctx.alpha
        gain = ctx.gain
        
        # Fast Sigmoid derivative: 1 / (1 + alpha * |v - threshold|)^2
        # Scaled by alpha to normalize the peak at the threshold
        surr_grad = alpha / (1 + alpha * (input - threshold).abs()).pow(2)
        
        # Apply gain to boost signal flow
        grad_input = grad_output * surr_grad * gain
        
        # Gradient for threshold: d(v-threshold)/d(threshold) = -1
        grad_threshold = grad_output * (-surr_grad) * gain
        
        # Sum over broadcast dimensions for threshold
        while grad_threshold.ndim > threshold.ndim:
            grad_threshold = grad_threshold.sum(dim=0)
        for i in range(threshold.ndim):
            if threshold.shape[i] == 1 and grad_threshold.shape[i] > 1:
                grad_threshold = grad_threshold.sum(dim=i, keepdim=True)
        
        return grad_input, grad_threshold, None, None

# Leaky Integrate and Fire Neuron
class LIFNeuron(nn.Module):
    def __init__(self, d_model, alpha=5.0, gain=1.0):
        super(LIFNeuron, self).__init__()
        self.d_model = d_model
        self.alpha = alpha
        self.gain = gain
        self.threshold = nn.Parameter(torch.ones(d_model) * 1.0)
        self.tau = nn.Parameter(torch.ones(d_model) * 0.5)
        self.v_reset = nn.Parameter(torch.zeros(1, 1, d_model))
        self.register_buffer("v", torch.zeros(1, 1, d_model))
    
    def forward(self, x):
        # Initial state update with broadcasting
        # We MUST NOT modify v in-place after SurrogateSpike.apply
        v = self.tau * self.v + x
        
        # Spike using Surrogate Gradient for backprop
        # Reduced alpha for smoother gradients (5.0 is standard)
        spike = SurrogateSpike.apply(v, self.threshold, self.alpha, self.gain)
        
        # Reset mechanics - create a NEW version of v for the next state
        # Removed no_grad to allow v_reset to receive gradients
        mask = v >= self.threshold
        # Out-of-place reset to avoid a 'modified by inplace operation' error
        v_next = v.masked_fill(mask, 0.0) + self.v_reset * spike
        
        # Persistence (save for next time step)
        self.v = v_next[:, -1:, :].detach()
        
        # Normalization (Safer epsilon + clamp to prevent gradient explosion)
        # Gradient of 1/(x+eps) is -1/(x+eps)^2. If x=0, eps=1e-5, grad is -1e10!
        # Clamping to 1.0 ensures we don't amplify gradients when there are 0 or 1 spikes.
        denom = spike.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return spike / denom
    
    def reset(self):
        self.v = torch.zeros(1, 1, self.d_model).to(self.threshold.device)
    
    def init_weights(self):
        self.threshold.data.fill_(1.0)
        self.tau.data.fill_(0.5)
        self.v_reset.data.fill_(0.0)

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
    
    def reset_state(self):
        self.lif.reset()
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
        self.lif.init_weights()

class TestModel(nn.Module):
    def __init__(self, d_model, tree_depth=4):
        super().__init__()
        self.d_model = d_model
        self.tree_depth = tree_depth
        
        # We need projections for different phases
        # 1. Up-Sweep: S_parent = Merge(concat(S_left, S_right))
        self.up_projs = nn.ModuleList([nn.Linear(d_model * 2, d_model) for _ in range(tree_depth)])
        self.up_lifs = nn.ModuleList([LIFNeuron(d_model) for _ in range(tree_depth)])
        # Experiment with up-sweep normalization
        self.up_norms = nn.ModuleList([nn.RMSNorm(d_model) for _ in range(tree_depth)])

        # 2. Down-Sweep: C_right = Distribute(concat(C_parent, S_left))
        # C_left inherits C_parent directly
        self.down_projs = nn.ModuleList([nn.Linear(d_model * 2, d_model) for _ in range(tree_depth)])
        self.down_lifs = nn.ModuleList([LIFNeuron(d_model) for _ in range(tree_depth)])
        # Experiment with down-sweep normalization
        self.down_norms = nn.ModuleList([nn.RMSNorm(d_model) for _ in range(tree_depth)])

        # 3. Leaf mixing: output = Final(concat(x, C_leaf))
        self.leaf_proj = nn.Linear(d_model * 2, d_model)
        self.leaf_lif = LIFNeuron(d_model)
        # Experiment with final layer normalization
        self.leaf_norm = nn.RMSNorm(d_model)

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        num_levels = int(math.log2(T))
        
        # --- PHASE 1: UP-SWEEP (Reduction) ---
        # summaries[l] stores the nodes at level l
        # summaries[0] is the leaves (tokens x)
        summaries = [x]
        for l in range(num_levels):
            curr = summaries[-1]
            T_curr = curr.shape[1]
            if T_curr < 2: break
            
            # Efficient reshaping
            merged = curr.view(B, T_curr // 2, 2 * D)
            
            # Tree Residual: Mean pool child summaries
            res = (merged[:, :, :D] + merged[:, :, D:]) * 0.5
            
            proj_idx = l if l < self.tree_depth else self.tree_depth - 1
            summary_next = self.up_lifs[proj_idx](self.up_norms[proj_idx](self.up_projs[proj_idx](merged)))
            summaries.append(summary_next + res)
        
        # --- PHASE 2: DOWN-SWEEP (Distribution) ---
        # contexts[l] stores the left-prefix context for nodes at level l
        # Start at the root (highest level) with 0 context
        contexts = [None] * (num_levels + 1)
        contexts[num_levels] = torch.zeros(B, 1, D, device=x.device) # root context
        
        for l in range(num_levels, 0, -1):
            parent_ctx = contexts[l]
            # child_summaries: [B, T/2^l, 2, D]
            # Sliced children to avoid view(..., 2, D) overhead if possible, 
            # but view is usually cheap. The key is reducing torch.cat.
            child_summaries = summaries[l-1].view(B, parent_ctx.shape[1], 2, D)
            
            # Left inherits, Right = Proj(Cat(Parent, LeftSummaries))
            combined = torch.cat([parent_ctx, child_summaries[:, :, 0, :]], dim=-1)
            proj_idx = (l-1) if (l-1) < self.tree_depth else self.tree_depth - 1
            ctx_right = self.down_lifs[proj_idx](self.down_norms[proj_idx](self.down_projs[proj_idx](combined)))
            
            # Down-sweep Residual: Add parent context into the distributed right context
            ctx_right = ctx_right + parent_ctx
            
            # Interleave efficiently
            contexts[l-1] = torch.stack([parent_ctx, ctx_right], dim=2).view(B, -1, D)
            
        return self.leaf_lif(self.leaf_norm(self.leaf_proj(torch.cat([x, contexts[0]], dim=-1))))
    
    def reset_state(self):
        for lif in self.up_lifs:
            lif.reset()
        for lif in self.down_lifs:
            lif.reset()
        self.leaf_lif.reset()
    
    def init_weights(self):
        for proj in self.up_projs:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
        for proj in self.down_projs:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
        nn.init.xavier_uniform_(self.leaf_proj.weight)
        nn.init.zeros_(self.leaf_proj.bias)
        for lif in self.up_lifs:
            lif.init_weights()
        for lif in self.down_lifs:
            lif.init_weights()
        self.leaf_lif.init_weights()
        self.leaf_norm.weight.data.fill_(1.0)

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
        # Apply RMSNorm and Attention Residual
        attn_out = self.attention(self.pre_norm_attn(x))
        x = x + self.residual_scale_attn * self.post_norm_attn(attn_out)
        
        # Apply RMSNorm and FFN Residual
        ffn_out = self.feed_forward(self.pre_norm_ffn(x))
        x = x + self.residual_scale_ffn * self.post_norm_ffn(ffn_out)
        return x
    
    def reset_state(self):
        self.attention.reset_state()
        self.feed_forward.reset_state()
    
    def init_weights(self):
        self.pre_norm_attn.weight.data.fill_(1.0)
        self.post_norm_attn.weight.data.fill_(1.0)
        self.pre_norm_ffn.weight.data.fill_(1.0)
        self.post_norm_ffn.weight.data.fill_(1.0)
        self.attention.init_weights()
        self.feed_forward.init_weights()
        self.residual_scale_attn.data.fill_(0.01)
        self.residual_scale_ffn.data.fill_(0.01)

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
        return self.lm_head(self.ln_f(x))
    
    def reset_state(self):
        for layer in self.layers:
            layer.reset_state()

    def compile(self, fullgraph=False):
        """
        Optional: Compile layers for performance.
        Useful when seq_len is large and stable.
        """
        if hasattr(torch, "compile"):
            self.layers = nn.ModuleList([torch.compile(l, fullgraph=fullgraph) for l in self.layers])
            print("Model layers compiled.")

if __name__ == "__main__":
    # Optimize for Tensor Cores
    torch.set_float32_matmul_precision('high')
    # Enable anomaly detection to find the exact source of NaNs
    torch.autograd.set_detect_anomaly(True)
    
    B, T, D = 4, 1024, 256
    model = TestCrina(256, 256, 12, 4).to(device)
    x = torch.randint(0, 256, (B, T)).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Gradient Check (Verification)
    print("Verifying Gradient Flow for LIF Parameters...")
    model.train()
    x_grad = torch.randint(0, 256, (B, T)).to(device)
    y_grad = model(x_grad)
    
    # Check for NaNs early in forward pass
    if torch.isnan(y_grad).any():
        print("Warning: Forward pass already contains NaNs!")
    
    loss = y_grad.mean()
    loss.backward()
    
    # Check specific LIF parameters in the first attention layer
    first_attn = model.layers[0].attention
    params_to_check = {
        "Weight Proj (Tree Leaf)": first_attn.up_projs[0].weight,
        "Weight Proj (Tree Root)": first_attn.up_projs[-1].weight,
        "LIF Tau": first_attn.up_lifs[0].tau,
        "LIF Threshold": first_attn.up_lifs[0].threshold,
        "LIF V_Reset": first_attn.up_lifs[0].v_reset
    }
    
    for name, param in params_to_check.items():
        has_grad = param.grad is not None
        grad_norm = param.grad.norm().item() if has_grad else 0.0
        print(f"{name} -> Has Grad: {has_grad}, Grad Norm: {grad_norm:.6f}")
    
    # Check for NaNs in all gradients
    if torch.isnan(torch.tensor([p.grad.norm().item() for p in model.parameters() if p.grad is not None])).any():
        print("NaN detected in gradients!")
    
    # Baseline benchmark (Improved CPU/GPU code)
    print("Running Uncompiled Baseline...")
    model.eval()
    with torch.no_grad():
        for _ in range(2):
            y = model(x)
        
        start = time.time()
        for _ in range(10):
            y = model(x)
        end = time.time()
        print(f"Uncompiled average inference time: {(end-start)/10:.4f}s")

    # Compiled benchmark
    model.compile()
    with torch.no_grad():
        print("Compiling (First pass will be slow)...")
        y = model(x) # Cold start
        
        start = time.time()
        for _ in range(10):
            y = model(x)
        end = time.time()
    
    print(f"Output shape: {y.shape}")
    print(f"Compiled average inference time: {(end-start)/10:.4f}s")