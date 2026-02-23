import torch
import torch.nn as nn
from crina_tinyshakespeare import CrinaSynapse
import math

def register_nan_hooks(model):
    """Register forward and backward hooks to detect NaN sources"""
    
    def forward_hook(module, input, output):
        module_name = module.__class__.__name__
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any():
                print(f"!!! NaN detected in FORWARD of {module_name}")
                print(f"    Output shape: {output.shape}")
                print(f"    NaN count: {torch.isnan(output).sum().item()}")
                print(f"    Max value: {output[~torch.isnan(output)].max().item() if (~torch.isnan(output)).any() else 'all NaN'}")
                raise ValueError(f"NaN in forward pass of {module_name}")
        elif isinstance(output, (list, tuple)):
            for i, o in enumerate(output):
                if isinstance(o, torch.Tensor) and torch.isnan(o).any():
                    print(f"!!! NaN detected in FORWARD of {module_name} (output {i})")
                    raise ValueError(f"NaN in forward pass of {module_name}")
    
    def backward_hook(module, grad_input, grad_output):
        module_name = module.__class__.__name__
        if grad_output is not None:
            for i, go in enumerate(grad_output):
                if go is not None and isinstance(go, torch.Tensor) and torch.isnan(go).any():
                    print(f"!!! NaN detected in BACKWARD (grad_output) of {module_name}")
                    raise ValueError(f"NaN in backward pass of {module_name}")
    
    hooks = []
    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(forward_hook))
        hooks.append(module.register_full_backward_hook(backward_hook))
    
    return hooks

def check_parameters(model, iteration):
    """Check for NaN or extreme values in parameters"""
    for name, param in model.named_parameters():
        if param is None:
            continue
        if torch.isnan(param).any():
            print(f"ITER {iteration}: NaN in parameter {name}")
            return False
        if torch.isinf(param).any():
            print(f"ITER {iteration}: Inf in parameter {name}")
            return False
        max_val = param.abs().max().item()
        if max_val > 1e6:
            print(f"ITER {iteration}: Extreme value in parameter {name}: {max_val}")
    return True

def diagnose():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Diagnosing on {device}...")

    # Configuration matching user's setup
    vocab_size = 256
    d_model = 256
    n_layers = 8
    tree_depth = 4
    B, T = 2, 1024

    model = CrinaSynapse(
        vocab_size=vocab_size, 
        d_model=d_model, 
        n_layers=n_layers, 
        tree_depth=tree_depth
    ).to(device)

    model.train()
    
    # Register hooks
    print("Registering NaN detection hooks...")
    hooks = register_nan_hooks(model)

    # Optimizer with gradient clipping
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Mock Data
    xs = [torch.randint(0, vocab_size, (B, T), device=device) for _ in range(32)]
    ys = [torch.randint(0, vocab_size, (B, T), device=device) for _ in range(32)]

    print("Starting diagnostic loop...")
    try:
        for epoch in range(10):
            for i in range(32):
                iteration = epoch * 32 + i
                
                optimizer.zero_grad()
                
                # Reset state
                if hasattr(model, 'reset'):
                    model.reset()

                x = xs[i]
                y = ys[i]

                # Check parameters before forward
                if not check_parameters(model, iteration):
                    print(f"Parameter check failed at iteration {iteration}")
                    return

                # Forward
                logits = model(x)
                
                # Loss
                loss = nn.functional.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
                
                if iteration % 10 == 0:
                    print(f"ITER {iteration}: Loss = {loss.item():.4f}")

                # Backward
                loss.backward()

                # Check gradients
                max_grad = 0.0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        max_grad = max(max_grad, param.grad.abs().max().item())
                
                if iteration % 10 == 0:
                    print(f"ITER {iteration}: Max gradient = {max_grad:.4e}")
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()

    except Exception as e:
        print(f"\n!!! Exception at iteration {iteration}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()

if __name__ == "__main__":
    diagnose()
