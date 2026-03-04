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
        self.threshold = nn.Parameter(torch.ones(d_model))
        self.tau = nn.Parameter(torch.ones(d_model) * 0.5)
        self.v_reset = nn.Parameter(torch.zeros(1, 1, d_model))
        self.register_buffer("v", torch.zeros(1, 1, d_model))
    
    def forward(self, x):
        B, T, D = x.shape
        device = x.device
        dtype = x.dtype
        
        # 1. Parallel Integration (SRM)
        K = min(T, 64)
        # Using exp(log) for faster power calculation and better stability
        steps = torch.arange(K, device=device, dtype=dtype).flip(0).view(1, 1, K)
        log_tau = torch.log(self.tau.clamp(0.001, 0.999)).view(D, 1, 1)
        kappa = torch.exp(steps * log_tau)
        
        # Causal convolution (groups=D for channel-wise)
        x_padded = F.pad(x.transpose(1, 2), (K - 1, 0))
        v_integrated = F.conv1d(x_padded, kappa, groups=D).transpose(1, 2)
        
        # Persistence (decayed)
        p_steps = torch.arange(1, T + 1, device=device, dtype=dtype).view(1, -1, 1)
        decay = torch.exp(p_steps * log_tau.view(1, 1, D))
        v_integrated = v_integrated + (self.v * decay)

        # 2. Spiking (Vectorized Surrogate) - Surr gradient inline for speed
        diff = v_integrated - self.threshold
        hard_spike = (v_integrated >= self.threshold).float()
        surr = (self.alpha / (1 + self.alpha * diff.abs()).square()) * self.gain
        spike_seq = (hard_spike - surr).detach() + surr
        
        # Soft-Reset
        v_integrated = v_integrated - (spike_seq.detach() * self.threshold * 0.5)

        # Persistence for next batch
        # CRITICAL: Must clone() to avoid CUDA Graph overwriting errors in reduce-overhead mode
        self.v = v_integrated[:, -1:, :].detach().clone()
        
        # 3. Normalization (Per-token)
        denom = spike_seq.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return spike_seq / denom
        
    def reset(self):
        self.v.zero_()

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
        h = self.linear1(x)
        h = h + self.lif(h)  # Skip connection around LIF
        x = self.linear2(h)
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

        # Selection Mechanism (Gating)
        # We add "selectors" that allow the model to decide how much of the residual to keep
        self.up_selectors = nn.ModuleList([nn.Linear(d_model * 2, d_model) for _ in range(tree_depth)])
        self.down_selectors = nn.ModuleList([nn.Linear(d_model * 2, d_model) for _ in range(tree_depth)])
        self.leaf_selector = nn.Linear(d_model * 2, d_model)

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
            gate = torch.sigmoid(self.up_selectors[proj_idx](merged))
            x_proj = self.up_projs[proj_idx](merged)
            summary_next = x_proj + self.up_lifs[proj_idx](self.up_norms[proj_idx](x_proj))  # Skip around LIF
            summaries.append(gate * summary_next + (1 - gate) * res)
        
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
            
            proj_idx = (l-1) if (l-1) < self.tree_depth else self.tree_depth - 1
            combined = torch.cat([parent_ctx, child_summaries[:, :, 0, :]], dim=-1)
            gate = torch.sigmoid(self.down_selectors[proj_idx](combined))
            x_proj = self.down_projs[proj_idx](combined)
            ctx_right = x_proj + self.down_lifs[proj_idx](self.down_norms[proj_idx](x_proj))  # Skip around LIF
            
            # Down-sweep Selection residual
            ctx_right = gate * ctx_right + (1 - gate) * parent_ctx
            
            # Interleave efficiently
            contexts[l-1] = torch.stack([parent_ctx, ctx_right], dim=2).view(B, -1, D)
            
        leaf_combined = torch.cat([x, contexts[0]], dim=-1)
        gate = torch.sigmoid(self.leaf_selector(leaf_combined))
        x_proj = self.leaf_proj(leaf_combined)
        leaf_out = x_proj + self.leaf_lif(self.leaf_norm(x_proj))  # Skip around LIF
        return gate * leaf_out + (1 - gate) * x
    
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
        for sel in self.up_selectors:
            nn.init.xavier_uniform_(sel.weight)
            nn.init.constant_(sel.bias, -1.0) # Start biased towards residual
        for sel in self.down_selectors:
            nn.init.xavier_uniform_(sel.weight)
            nn.init.constant_(sel.bias, -1.0)
        nn.init.xavier_uniform_(self.leaf_selector.weight)
        nn.init.constant_(self.leaf_selector.bias, -1.0)

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

        self.residual_scale_attn = nn.Parameter(torch.tensor(0.1))
        self.residual_scale_ffn = nn.Parameter(torch.tensor(0.1))
    
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
        self.residual_scale_attn.data.fill_(0.1)
        self.residual_scale_ffn.data.fill_(0.1)

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
        Compile the whole model for peak performance.
        """
        if hasattr(torch, "compile"):
            # Compile the entire model to fuse the hierarchical tree loops
            print("Compiling model (Full)...")
            return torch.compile(self, fullgraph=fullgraph)
        return self

if __name__ == "__main__":
    # Optimize for Tensor Cores
    torch.set_float32_matmul_precision('high')
    
    B, T, D = 4, 1024, 256
    model = TestCrina(256, 256, 12, 4).to(device)
    x = torch.randint(0, 256, (B, T)).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Baseline benchmark
    print("Running Uncompiled Baseline...")
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            y = model(x)
        
        start = time.time()
        for _ in range(20):
            y = model(x)
        end = time.time()
        print(f"Uncompiled average inference time: {(end-start)/20:.4f}s")

    # Compiled benchmark
    compiled_model = model.compile()
    with torch.no_grad():
        print("Compiling (Warmup will be slow)...")
        # Compiler needs multiple steps to optimize
        for i in range(15):
            y = compiled_model(x)
            if i % 5 == 0: print(f"Warmup {i}/15...")
        
        start = time.time()
        for _ in range(20):
            y = compiled_model(x)
        end = time.time()
    
    print(f"Output shape: {y.shape}")
    print(f"Compiled average inference time: {(end-start)/20:.4f}s")
