
import torch
import torch._dynamo
from torch._subclasses.fake_tensor import is_fake
import crina_cuda

# Mocking the pattern found in crina_tinyshakespeare.py
# We will use the actual bindings if available, or just a dummy op if we want to isolate the framework behavior.
# But let's verify with the actual Linear wrapper first.

def linear_sigmoid_forward_wrapper(input, weight, bias=None):
    if is_fake(input):
        print("[Forward] Wrapped helper called with Fake Tensor. Returning empty.")
        output_shape = input.shape[:-1] + (weight.shape[0],)
        return input.new_empty(output_shape)
    print("[Forward] Wrapped helper called with Real Tensor.")
    return crina_cuda.linear_sigmoid_forward(input, weight, bias)

def linear_sigmoid_backward_wrapper(grad_output, output, input, weight):
    if is_fake(grad_output):
        print("[Backward] Wrapped helper called with Fake Tensor. Returning empty.")
        grad_input = torch.empty_like(input)
        grad_weight = torch.empty_like(weight)
        grad_bias = torch.empty((weight.shape[0],), device=input.device, dtype=input.dtype)
        # Initialize with garbage/NaNs explicitly to prove the point if empty doesn't
        # But 'empty' should be enough to show non-determinism or zeroes if lucky, but often garbage.
        return grad_input, grad_weight, grad_bias
    print("[Backward] Wrapped helper called with Real Tensor.")
    # Convert list to tuple if necessary
    results = crina_cuda.linear_sigmoid_backward(grad_output, output, input, weight)
    if isinstance(results, list):
        return tuple(results)
    return results

# We need to replicate the 'allow_in_graph' behavior?
# In crina_tinyshakespeare.py, the wrappers are marked allow_in_graph.
torch.compiler.allow_in_graph(linear_sigmoid_forward_wrapper)
torch.compiler.allow_in_graph(linear_sigmoid_backward_wrapper)

class FusedLinearSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        output = linear_sigmoid_forward_wrapper(input, weight, bias)
        ctx.save_for_backward(output, input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, input, weight = ctx.saved_tensors
        return linear_sigmoid_backward_wrapper(grad_output, output, input, weight)

def test_fn(x, w, b):
    return FusedLinearSigmoid.apply(x, w, b)

def main():
    torch.manual_seed(42)
    device = "cuda"
    
    # Inputs
    B, T, D = 2, 16, 64
    x = torch.randn(B, T, D, device=device, requires_grad=True)
    w = torch.randn(D, D, device=device, requires_grad=True)
    b = torch.randn(D, device=device, requires_grad=True)

    print("\n--- Eager Execution ---")
    y_eager = test_fn(x, w, b)
    loss = y_eager.sum()
    loss.backward()
    grad_x_eager = x.grad.clone()
    print("Eager grad_x mean:", grad_x_eager.mean().item())
    
    x.grad = None
    w.grad = None
    b.grad = None

    print("\n--- Compiled Execution ---")
    opt_fn = torch.compile(test_fn, backend="inductor")
    
    # First run (compilation)
    y_comp = opt_fn(x, w, b)
    loss_comp = y_comp.sum()
    loss_comp.backward()
    
    grad_x_comp = x.grad.clone()
    print("Compiled grad_x mean:", grad_x_comp.mean().item())
    
    # Check for NaNs
    if torch.isnan(grad_x_comp).any():
        print("!!! DETECTED NANS IN COMPILED GRADIENTS !!!")
    else:
        print("No NaNs detected in compiled gradients (might be lucky with empty memory).")
        
    # Check match
    if not torch.allclose(grad_x_eager, grad_x_comp, atol=1e-5):
        print("!!! MISMATCH BETWEEN EAGER AND COMPILED !!!")
        print("Max diff:", (grad_x_eager - grad_x_comp).abs().max().item())
    else:
        print("Gradients match.")

if __name__ == "__main__":
    main()
