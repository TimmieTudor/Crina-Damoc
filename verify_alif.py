import torch
import torch.nn as nn
import torch.nn.functional as F
import crina_cuda
from crina_tinyshakespeare import ALIFNeuron, ALIFFunction
import time
import numpy as np

def verify_and_benchmark_alif(B, N, D, iters=50, warmup=10):
    device = torch.device('cuda')
    
    # CRITICAL: Always reset grad_enabled at start of verification
    torch.set_grad_enabled(True)
    
    # Initialize inputs
    i_inj = torch.randn(B, N, D, device=device, requires_grad=True)
    v_state_init = torch.randn(B, 1, D, device=device).requires_grad_(True)
    a_state_init = torch.randn(B, 1, D, device=device).requires_grad_(True)
    
    threshold = 0.5
    tau_mem = 0.99
    tau_adapt = 0.95
    adapt_strength = 0.1
    slope = 25.0
    
    # --- PyTorch Baseline (Bit-Exact Manual Math) ---
    # We expand the initial states to match the pointwise snapshot logic
    v_ref = tau_mem * v_state_init.expand(B, N, D) + i_inj
    td_ref = 0.5 + a_state_init.expand(B, N, D)
    # PyTorch baseline for forward pass
    so_ref = torch.sigmoid(slope * (v_ref - td_ref))
    # Project logic: spike value is surrogate during training
    ref_out = so_ref + (v_ref >= td_ref).float() - (v_ref >= td_ref).float().detach()

    # --- Custom Kernel ---
    # Signature: i_inj, v_state, a_state, bt, as_w, tm, ta, slope, training
    with torch.enable_grad():
        custom_out, vf_c, af_c, vt_c, td_c = ALIFFunction.apply(
            i_inj, v_state_init.squeeze(1), a_state_init.squeeze(1),
            torch.ones(D, device=device)*0.5, torch.ones(D, device=device)*0.1,
            tau_mem, tau_adapt, slope,
            True # training flag
        )
    
    mae_fwd = torch.mean(torch.abs(ref_out - custom_out)).item()
    mae_vt = torch.mean(torch.abs(v_ref - vt_c)).item() if vt_c.numel() > 0 else 1e9
    mae_td = torch.mean(torch.abs(td_ref - td_c)).item() if td_c.numel() > 0 else 1e9
    
    # --- Backward Verification ---
    grad_output = torch.randn_like(ref_out)
    
    # Baseline Grad
    ref_out.backward(grad_output, retain_graph=True)
    ref_grad_in = i_inj.grad.clone()
    ref_grad_vs = v_state_init.grad.clone()
    ref_grad_as = a_state_init.grad.clone()
    
    # Custom Grad
    i_inj.grad.zero_()
    v_state_init.grad.zero_()
    a_state_init.grad.zero_()
    
    custom_out.backward(grad_output)
    
    mae_grad_in = torch.mean(torch.abs(ref_grad_in - i_inj.grad)).item()
    mae_grad_vs = torch.mean(torch.abs(ref_grad_vs - v_state_init.grad)).item()
    mae_grad_as = torch.mean(torch.abs(ref_grad_as - a_state_init.grad)).item()
    
    print(f"Shape: [{B}, {N}, {D}]")
    print(f"Forward MAE: {mae_fwd:.6e}")
    if vt_c.numel() > 0:
        print(f"Internal Activation MAE: VT={mae_vt:.6e}, TD={mae_td:.6e}")
    else:
        print("ERROR: Saved state tensors are empty!")

    print(f"Grad Input MAE: {mae_grad_in:.6e}")
    print(f"Grad Sample [0,0,0]: Ref={ref_grad_in[0,0,0]:.4f}, Custom={i_inj.grad[0,0,0]:.4f}")
    print(f"Initial State Grad MAE: VS={mae_grad_vs:.6e}, AS={mae_grad_as:.6e}")
    
    # --- Benchmark (Inference mode) ---
    torch.set_grad_enabled(False)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        v = tau_mem * v_state_init.expand(B, N, D) + i_inj
        td = 0.5 + a_state_init.expand(B, N, D)
        so = torch.sigmoid(slope * (v - td))
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) * 1000 / iters

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        custom_out_inf = ALIFFunction.apply(
            i_inj, v_state_init.squeeze(1), a_state_init.squeeze(1),
            torch.ones(D, device=device)*0.5, torch.ones(D, device=device)*0.1,
            tau_mem, tau_adapt, slope,
            False # training=False
        )
    torch.cuda.synchronize()
    cuda_time = (time.perf_counter() - start) * 1000 / iters
    
    print(f"Speedup: {pytorch_time / cuda_time:.2f}x ({pytorch_time:.3f}ms vs {cuda_time:.3f}ms)")
    print("-" * 30)

if __name__ == "__main__":
    print("Verifying Fused ALIF Neuron Kernel (Signature & State Fixed)...")
    verify_and_benchmark_alif(B=1, N=1, D=256)
    verify_and_benchmark_alif(B=1, N=64, D=256)
    verify_and_benchmark_alif(B=8, N=128, D=256)
