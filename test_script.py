import torch
import torch.nn as nn
from torch.nn import functional as F
#from crina_tinyshakespeare import CrinaSynapse
from test_attention import TestCrina
from benchmark_crina_vs_llama import Model
import matplotlib.pyplot as plt
import time
import math

def get_model_params(model):
    """Returns the number of trainable parameters in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6

def benchmark_model(model, input_length, device, iters=10, warmup=3):
    """
    Measures the average inference time for a model at a given input length.
    Includes warmup and CUDA synchronization for accurate measurement.
    """
    model.eval()
    dummy_input = torch.randint(0, 256, (1, input_length), device=device)
    
    # Reset state (Very important for Crina's Temporal Memory)
    if hasattr(model, 'reset_state'):
        model.reset_state()
    elif hasattr(model, 'reset'):
        model.reset()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    if device == 'cuda':
        torch.cuda.synchronize()
        
    # Latency measurement
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(dummy_input)
            
    if device == 'cuda':
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    avg_time = (end_time - start_time) / iters
    return avg_time


def run_comparison():
    # Configuration
    torch._dynamo.config.recompile_limit = 22
    config = {
        "d_model": 256,
        "n_layers": 16,
        "vocab_size": 256,
        "tree_depth": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    #    "device": "cpu",
        "min_power": 2, # 2^2 = 4
        "max_power": 11, # 2^11 = 2048
        "iters": 12,
        "warmup": 5
    }
    
    print(f"Benchmarking on: {config['device'].upper()}")
    
    # Speed up for CUDA
    if config['device'] == 'cuda':
        torch.set_float32_matmul_precision('high')
        
    # Initialize models
    crina = TestCrina(
        vocab_size=config['vocab_size'], 
        d_model=config['d_model'], 
        num_layers=config['n_layers'], 
        tree_depth=config['tree_depth']
    ).to(config['device'])
    
    llama = Model(
        vocab_size=config['vocab_size'], 
        d_model=config['d_model'], 
        n_layers=config['n_layers'], 
        use_tree=False
    ).to(config['device'])
    
    # Compile models (fuses kernels and removes dispatcher overhead)
    if config['device'] == 'cuda':
        print("Compiling models...")
        crina = torch.compile(crina)
        llama = torch.compile(llama)
    
    print(f"Crina Parameters: {get_model_params(crina):.2f}M")
    print(f"Llama Parameters: {get_model_params(llama):.2f}M")
    
    lengths = [2**i for i in range(config['min_power'], config['max_power'] + 1)]
    llama_times = []
    crina_times = []
    
    print(f"{'Length':<10} | {'Llama (ms)':<12} | {'Crina (ms)':<12}")
    print("-" * 40)
    
    for length in lengths:
        t_llama = benchmark_model(llama, length, config['device'], config['iters'], config['warmup'])
        t_crina = benchmark_model(crina, length, config['device'], config['iters'], config['warmup'])
        
        llama_times.append(t_llama * 1000) # Convert to ms
        crina_times.append(t_crina * 1000)
        
        print(f"{length:<10} | {llama_times[-1]:<12.2f} | {crina_times[-1]:<12.2f}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(lengths, llama_times, marker='o', label='Standard Transformer (O(N²))', linewidth=2)
    plt.plot(lengths, crina_times, marker='s', label='Crina SNN Tree (O(N log N))', linewidth=2)
    
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel('Sequence Length (log scale)')
    plt.ylabel('Inference Latency (ms, log scale)')
    plt.title('Complexity Scaling: Crina vs. Transformer')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    # Save the plot
    plt.savefig('complexity_scaling.png')
    print("\nBenchmark complete. Plot saved as 'complexity_scaling.png'.")
    plt.show()

if __name__ == "__main__":
    run_comparison()