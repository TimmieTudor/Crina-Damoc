
import torch
import torch.nn as nn
from crina_tinyshakespeare import CrinaSynapse
import math

def verify_training():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Verifying training on {device}...")

    # Mini configuration
    vocab_size = 256
    d_model = 256
    n_layers = 8
    tree_depth = 4
    B, T = 8, 1024

    model = CrinaSynapse(
        vocab_size=vocab_size, 
        d_model=d_model, 
        n_layers=n_layers, 
        tree_depth=tree_depth
    ).to(device)

    # Enable training mode (important for ALIF surrogate grads)
    model.train()

    # Compile
    print("Compiling model...")
    backend = "inductor" # Default
    model = torch.compile(model, backend=backend)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Mock Data
    xs = [torch.randint(0, vocab_size, (B, T), device=device) for _ in range(50)]
    ys = [torch.randint(0, vocab_size, (B, T), device=device) for _ in range(50)]

    last_grad = dict()
    for name, param in model.named_parameters():
        last_grad[name] = torch.zeros_like(param)

    # Training loop
    print("Starting training loop...")
    for epoch in range(10):
        for i in range(50):
            optimizer.zero_grad()
            
            # Reset state if needed (usually handled in model or forward, but check API)
            if hasattr(model, 'reset'):
                model.reset()
            elif hasattr(model, 'orig_mod') and hasattr(model.orig_mod, 'reset'):
                model.orig_mod.reset()

            x = xs[i]
            y = ys[i]

            # Forward
            logits = model(x)
            
            # Check output for NaNs
            if torch.isnan(logits).any():
                print(f"EPOCH {epoch}, ITER {i}: NaNs detected in Forward Output!")
                # Check the gradients. Are they vanishing? Are they exploding?
                for name, grad in last_grad.items():
                    if grad.abs().sum() <= 1e-10:
                        print(f"EPOCH {epoch}, ITER {i}: Gradient of {name} is vanishingly small!")
                    elif grad.abs().max() > 1e10:
                        print(f"EPOCH {epoch}, ITER {i}: Gradient of {name} is exploding!")
                # Print last gradient
                print("Last Gradient:")
                print(last_grad)
                return False

            # Loss
            loss = nn.functional.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            print(f"EPOCH {epoch}, ITER {i}: Loss = {loss.item():.4f}")

            # Backward
            loss.backward()

            # Check gradients
            has_nans = False
            all_zeros = True
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"EPOCH {epoch}, ITER {i}: NaNs detected in gradient of {name}")
                        has_nans = True
                    if param.grad.abs().sum() > 0:
                        all_zeros = False
                    last_grad[name] = param.grad.detach().clone()
            
            if has_nans:
                print("!!! VERIFICATION FAILED: NaNs in gradients !!!")
                return False
            
            if all_zeros:
                # Note: Sparse spikes might produce zero gradients sometimes, 
                # but usually at least weights should move.
                print(f"EPOCH {epoch}, ITER {i}: Warning - All gradients are zero (could be dead neurons or disconnected graph)")
                # Don't fail immediately, maybe valid for step 0
            
            # CRITICAL: Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
    
    print("Training loop completed successfully!")

    print("\nVerification Passed: 10 epochs of 50 iterations completed with no NaNs.")
    return True

if __name__ == "__main__":
    try:
        success = verify_training()
        if not success:
            exit(1)
    except Exception as e:
        print(f"CRASHED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
