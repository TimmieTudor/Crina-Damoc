import torch
import torch.nn as nn
import torch.nn.functional as F
import crina_cuda
import time
import numpy as np

def verify_and_benchmark_rmsnorm(M, D, iters=100, warmup=10):
    device = torch.device('cuda')
    
    # Initialize inputs
    input_tensor = torch.randn(M, D, device=device, requires_grad=True)
    weight = torch.randn(D, device=device, requires_grad=True)
    eps = 1e-6
    
    # --- PyTorch Baseline ---
    norm_ref = nn.RMSNorm(D, eps=eps).to(device)
    with torch.no_grad():
        norm_ref.weight.copy_(weight)
    
    # Forward Warmup
    for _ in range(warmup):
        ref_out = norm_ref(input_tensor)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        ref_out = norm_ref(input_tensor)
    torch.cuda.synchronize()
    pytorch_time_fwd = (time.perf_counter() - start) * 1000 / iters
    
    # Backward Baseline
    grad_output = torch.randn_like(ref_out)
    for _ in range(warmup):
        ref_out.backward(grad_output, retain_graph=True)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        input_tensor.grad = None
        norm_ref.weight.grad = None
        ref_out = norm_ref(input_tensor)
        ref_out.backward(grad_output, retain_graph=True)
    torch.cuda.synchronize()
    pytorch_time_bwd = (time.perf_counter() - start) * 1000 / iters
    
    ref_grad_in = input_tensor.grad.clone()
    ref_grad_w = norm_ref.weight.grad.clone()

    # --- Custom CUDA Kernel ---
    # Forward Custom
    for _ in range(warmup):
        custom_out, inv_rms = crina_cuda.rmsnorm_forward(input_tensor, weight, eps)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        custom_out, inv_rms = crina_cuda.rmsnorm_forward(input_tensor, weight, eps)
    torch.cuda.synchronize()
    cuda_time_fwd = (time.perf_counter() - start) * 1000 / iters
    
    # Backward Custom
    for _ in range(warmup):
        custom_grad_in, custom_grad_w = crina_cuda.rmsnorm_backward(grad_output, input_tensor, weight, inv_rms)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        custom_grad_in, custom_grad_w = crina_cuda.rmsnorm_backward(grad_output, input_tensor, weight, inv_rms)
    torch.cuda.synchronize()
    cuda_time_bwd = (time.perf_counter() - start) * 1000 / iters
    
    # --- Numerical Parity ---
    mae_fwd = torch.mean(torch.abs(ref_out - custom_out)).item()
    mae_grad_in = torch.mean(torch.abs(ref_grad_in - custom_grad_in)).item()
    mae_grad_w = torch.mean(torch.abs(ref_grad_w - custom_grad_w)).item()
    
    print(f"Shape: [{M}, {D}]")
    print(f"Forward MAE: {mae_fwd:.6e}")
    print(f"Grad Input MAE: {mae_grad_in:.6e}")
    print(f"Grad Weight MAE: {mae_grad_w:.6e}")
    print(f"Forward Speedup: {pytorch_time_fwd / cuda_time_fwd:.2f}x ({pytorch_time_fwd:.3f}ms vs {cuda_time_fwd:.3f}ms)")
    print(f"Backward Speedup: {pytorch_time_bwd / cuda_time_bwd:.2f}x ({pytorch_time_bwd:.3f}ms vs {cuda_time_bwd:.3f}ms)")
    print("-" * 30)

if __name__ == "__main__":
    print("Verifying Fused RMSNorm Kernel...")
    verify_and_benchmark_rmsnorm(M=1024, D=256)
    verify_and_benchmark_rmsnorm(M=2048, D=512)
    verify_and_benchmark_rmsnorm(M=128, D=2048)
