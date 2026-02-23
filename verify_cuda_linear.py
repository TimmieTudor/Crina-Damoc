import torch
import torch.nn as nn
import torch.nn.functional as F
import crina_cuda
import time
import numpy as np

def verify_and_benchmark(M, N, K, batch_dims=(), iters=100, warmup=10):
    device = torch.device('cuda')
    
    # Input shapes
    input_shape = batch_dims + (M, K)
    
    # Initialize inputs
    input_tensor = torch.randn(input_shape, device=device)
    weight = torch.randn(N, K, device=device)
    bias = torch.randn(N, device=device)
    
    # --- PyTorch Baseline ---
    def pytorch_baseline(x, w, b):
        return torch.sigmoid(F.linear(x, w, b))
    
    # Warmup
    for _ in range(warmup):
        ref_out = pytorch_baseline(input_tensor, weight, bias)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        ref_out = pytorch_baseline(input_tensor, weight, bias)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) * 1000 / iters
    
    # --- Custom CUDA Kernel ---
    # The kernel supports broadcasting via flattening leading dims
    def cuda_custom(x, w, b):
        return crina_cuda.linear_sigmoid_forward(x, w, b)
    
    # Warmup
    for _ in range(warmup):
        custom_out = cuda_custom(input_tensor, weight, bias)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        custom_out = cuda_custom(input_tensor, weight, bias)
    torch.cuda.synchronize()
    cuda_time = (time.perf_counter() - start) * 1000 / iters
    
    # --- Numerical Parity ---
    mae = torch.mean(torch.abs(ref_out - custom_out)).item()
    max_err = torch.max(torch.abs(ref_out - custom_out)).item()
    
    print(f"Shape: {input_shape} -> {custom_out.shape}")
    print(f"MAE: {mae:.6e}")
    print(f"Max Error: {max_err:.6e}")
    print(f"PyTorch Time: {pytorch_time:.3f} ms")
    print(f"CUDA Custom Time: {cuda_time:.3f} ms")
    print(f"Speedup: {pytorch_time / cuda_time:.2f}x")
    print("-" * 30)
    
    return mae, pytorch_time, cuda_time

if __name__ == "__main__":
    print("Verifying Fused Linear Sigmoid Kernel...")
    
    # Standard 2D case
    print("\nCase 1: Standard 2D [256, 512] x [512, 512]")
    verify_and_benchmark(M=256, N=512, K=512)
    
    # Broadcasting case (3D)
    print("\nCase 2: 3D Broadcasting [8, 128, 256] x [512, 256]")
    verify_and_benchmark(M=128, N=512, K=256, batch_dims=(8,))
    
    # Small shapes (Stress tiling boundaries)
    print("\nCase 3: Small Bounds [7, 13] x [19, 13]")
    verify_and_benchmark(M=7, N=19, K=13)
    
    # Large case
    print("\nCase 4: Large [1024, 1024] x [1024, 1024]")
    verify_and_benchmark(M=1024, N=1024, K=1024)

    # Test without bias
    print("\nCase 5: Without Bias")
    device = torch.device('cuda')
    input_tensor = torch.randn(128, 256, device=device)
    weight = torch.randn(512, 256, device=device)
    
    ref_out = torch.sigmoid(F.linear(input_tensor, weight, None))
    custom_out = crina_cuda.linear_sigmoid_forward(input_tensor, weight, None)
    
    mae = torch.mean(torch.abs(ref_out - custom_out)).item()
    print(f"MAE (No Bias): {mae:.6e}")
