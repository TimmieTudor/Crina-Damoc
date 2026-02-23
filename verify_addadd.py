import torch
import torch.nn as nn
import torch.nn.functional as F
import crina_cuda
import time
import numpy as np

def verify_and_benchmark(D, batch_dims=(), iters=100, warmup=10):
    device = torch.device('cuda')
    
    # Input shapes
    input_shape = batch_dims + (D,)
    
    # Initialize inputs
    a = torch.randn(D, device=device, requires_grad=True)
    b = torch.randn(D, device=device, requires_grad=True)
    c = torch.randn(D, device=device, requires_grad=True)
    
    # --- PyTorch Baseline ---
    def pytorch_baseline(a, b, c):
        return a + b + c
    
    # Warmup
    for _ in range(warmup):
        ref_out = pytorch_baseline(a, b, c)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        ref_out = pytorch_baseline(a, b, c)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) * 1000 / iters

    # Backward Baseline
    grad_output = torch.randn_like(ref_out)
    for _ in range(warmup):
        ref_out.backward(grad_output, retain_graph=True)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        ref_out = pytorch_baseline(a, b, c)
        ref_out.backward(grad_output, retain_graph=True)
    torch.cuda.synchronize()
    pytorch_time_bwd = (time.perf_counter() - start) * 1000 / iters
    
    ref_grad_a = a.grad.clone()
    ref_grad_b = b.grad.clone()
    ref_grad_c = c.grad.clone()
    
    # --- Custom CUDA Kernel ---
    # The kernel supports broadcasting via flattening leading dims
    def cuda_custom(a, b, c):
        return crina_cuda.addadd_forward(a, b, c)
    
    # Warmup
    for _ in range(warmup):
        custom_out = cuda_custom(a, b, c)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        custom_out = cuda_custom(a, b, c)
    torch.cuda.synchronize()
    cuda_time = (time.perf_counter() - start) * 1000 / iters

    grad_output = torch.randn_like(custom_out)
    grad_output.zero_()
    grad_a = torch.randn_like(a)
    grad_b = torch.randn_like(b)
    grad_c = torch.randn_like(c)

    grad_a.zero_()
    grad_b.zero_()
    grad_c.zero_()

    # Backward Custom
    for _ in range(warmup):
        grad_a, grad_b, grad_c = crina_cuda.addadd_backward(grad_output, a, b, c)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        grad_a, grad_b, grad_c = crina_cuda.addadd_backward(grad_output, a, b, c)
    torch.cuda.synchronize()
    cuda_time_bwd = (time.perf_counter() - start) * 1000 / iters
    
    custom_grad_a = a.grad.clone()
    custom_grad_b = b.grad.clone()
    custom_grad_c = c.grad.clone()
    
    # --- Numerical Parity ---
    mae = torch.mean(torch.abs(ref_out - custom_out)).item()
    max_err = torch.max(torch.abs(ref_out - custom_out)).item()
    
    mae_grad_a = torch.mean(torch.abs(ref_grad_a - custom_grad_a)).item()
    mae_grad_b = torch.mean(torch.abs(ref_grad_b - custom_grad_b)).item()
    mae_grad_c = torch.mean(torch.abs(ref_grad_c - custom_grad_c)).item()
    
    print(f"Shape: {input_shape} -> {custom_out.shape}")
    print(f"MAE: {mae:.6e}")
    print(f"Max Error: {max_err:.6e}")
    #print(f"PyTorch Time: {pytorch_time:.3f} ms")
    #print(f"CUDA Custom Time: {cuda_time:.3f} ms")
    print(f"Forward Speedup: {pytorch_time / cuda_time:.2f}x")
    print(f"Backward Speedup: {pytorch_time_bwd / cuda_time_bwd:.2f}x")
    print(f"Grad A MAE: {mae_grad_a:.6e}")
    print(f"Grad B MAE: {mae_grad_b:.6e}")
    print(f"Grad C MAE: {mae_grad_c:.6e}")
    print("-" * 30)
    
    return mae, pytorch_time, cuda_time

if __name__ == "__main__":
    print("Verifying Fused Linear Sigmoid Kernel...")
    
    # Standard 2D case
    print("\nCase 1: Standard 2D [256]")
    verify_and_benchmark(D=256)
    
    # Broadcasting case (3D)
    print("\nCase 2: 3D Broadcasting [8, 128]")
    verify_and_benchmark(D=128, batch_dims=(8,))
    
    # Small shapes (Stress tiling boundaries)
    print("\nCase 3: Small Bounds [7]")
    verify_and_benchmark(D=7)
    
    # Large case
    print("\nCase 4: Large [1024]")
    verify_and_benchmark(D=1024)