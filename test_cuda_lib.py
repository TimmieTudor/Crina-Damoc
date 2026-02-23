import torch
import square_cuda

class FusedLinearSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        output = square_cuda.linear_sigmoid_forward(input, weight, bias)
        ctx.save_for_backward(output, input, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, input, weight, bias = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = square_cuda.linear_sigmoid_backward(grad_output, output, input, weight)
        return grad_input, grad_weight, grad_bias

# Helper module
class FusedLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features).cuda())
        self.bias = torch.nn.Parameter(torch.randn(out_features).cuda())
        
    def forward(self, x):
        return FusedLinearSigmoid.apply(x, self.weight, self.bias)

# Test
device = 'cuda'
x = torch.randn(32, 128, device=device, requires_grad=True)
model = FusedLinear(128, 64).to(device)

# Forward
y = model(x)
print("Output shape:", y.shape)

# Backward
loss = y.sum()
loss.backward()
print("Gradient shape:", x.grad.shape)