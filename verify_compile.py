import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from train_crina_pc import PCCrina
from test_attention import TestCrina

if __name__ == "__main__":
    torch._dynamo.config.recompile_limit = 22
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def get_model():
        base_model = TestCrina(vocab_size=256, d_model=256, num_layers=12, tree_depth=4).to(device)
        pc_model = PCCrina(base_model, lr_state=0.01, inference_steps=15, task_weight=5.0).to(device)
        pc_model.initialize_from_model()
        pc_model.set_layer_lrs(sensory_lr=0.02, abstract_lr=0.005)
        return pc_model

    print("Benchmarking without compilation...")
    pc_model = get_model()
    times_no_compile = []
    for _ in range(10):
        start_time = time.time()
        pc_model.pc_train_step(torch.randint(0, 256, (1, 128)).to(device), torch.randint(0, 256, (1, 128)).to(device))
        times_no_compile.append(time.time() - start_time)
    avg_no_compile = sum(times_no_compile) / len(times_no_compile)

    print("Benchmarking with compilation (default)...")
    pc_model = get_model()
    pc_model.pc_train_step = torch.compile(pc_model.pc_train_step)
    # Warmup
    for _ in range(5):
        pc_model.pc_train_step(torch.randint(0, 256, (1, 128)).to(device), torch.randint(0, 256, (1, 128)).to(device))
    
    start_time = time.time()
    for _ in range(10):
        pc_model.pc_train_step(torch.randint(0, 256, (1, 128)).to(device), torch.randint(0, 256, (1, 128)).to(device))
    time_default = (time.time() - start_time) / 10
    
    print("Benchmarking with compilation (reduce-overhead)...")
    pc_model = get_model()
    pc_model.pc_train_step = torch.compile(pc_model.pc_train_step, mode="reduce-overhead")
    # Warmup
    for _ in range(5):
        pc_model.pc_train_step(torch.randint(0, 256, (1, 128)).to(device), torch.randint(0, 256, (1, 128)).to(device))
        
    start_time = time.time()
    for _ in range(10):
        pc_model.pc_train_step(torch.randint(0, 256, (1, 128)).to(device), torch.randint(0, 256, (1, 128)).to(device))
    time_ro = (time.time() - start_time) / 10

    print("Benchmarking with nested compilation (layers + training)...")
    pc_model = get_model()
    pc_model.model.compile() 
    pc_model.pc_train_step = torch.compile(pc_model.pc_train_step, mode="reduce-overhead")
    # Warmup
    for _ in range(5):
        pc_model.pc_train_step(torch.randint(0, 256, (1, 128)).to(device), torch.randint(0, 256, (1, 128)).to(device))
        
    start_time = time.time()
    for _ in range(10):
        pc_model.pc_train_step(torch.randint(0, 256, (1, 128)).to(device), torch.randint(0, 256, (1, 128)).to(device))
    time_nested = (time.time() - start_time) / 10
    
    print(f"\nFinal Results (Average per step):")
    print(f"Without compilation: {avg_no_compile:.4f}s")
    print(f"Default Compilation: {time_default:.4f}s (Speedup: {avg_no_compile / time_default:.2f}x)")
    print(f"Reduce-Overhead:     {time_ro:.4f}s (Speedup: {avg_no_compile / time_ro:.2f}x)")
    print(f"Nested/Layered RO:   {time_nested:.4f}s (Speedup: {avg_no_compile / time_nested:.2f}x)")