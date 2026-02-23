import torch
from torch.utils.cpp_extension import load_inline
import time

# 1. Define the CUDA Kernel (C++ source code)
# This string contains just the CUDA kernel and a simple C-style launcher function.
# It does not include any PyTorch-specific headers, making it pure C++/CUDA.
cuda_source = """
#include <cuda.h>
#include <cuda_runtime.h>

// The CUDA Kernel itself
__global__ void sigmoid_kernel(const float* __restrict__ input, float* __restrict__ output, int size) {
    // Calculate global thread index
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (index < size) {
        output[index] = 1.0f / (1.0f + expf(-input[index]));
    }
}

// The Backward Kernel (computes gradients)
// dy/dx = y * (1 - y)
// grad_input = grad_output * y * (1 - y)
__global__ void sigmoid_backward_kernel(const float* __restrict__ grad_output, const float* __restrict__ output, float* __restrict__ grad_input, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < size) {
        float y = output[index];
        grad_input[index] = grad_output[index] * y * (1.0f - y);
    }
}

// C-style launcher
extern "C" void sigmoid_launcher(const float* input, float* output, int size) {
    const int threads = 1024;
    const int blocks = (size + threads - 1) / threads;
    sigmoid_kernel<<<blocks, threads>>>(input, output, size);
}

extern "C" void sigmoid_backward_launcher(const float* grad_output, const float* output, float* grad_input, int size) {
    const int threads = 1024;
    const int blocks = (size + threads - 1) / threads;
    sigmoid_backward_kernel<<<blocks, threads>>>(grad_output, output, grad_input, size);
}
"""

# 2. Define the C++ wrapper using the PyTorch C++ API.
# This function will be compiled by the host C++ compiler (e.g., cl.exe).
cpp_source = """
#include <torch/extension.h>

// Forward declaration of the CUDA launcher
extern "C" void sigmoid_launcher(const float* input, float* output, int size);
extern "C" void sigmoid_backward_launcher(const float* grad_output, const float* output, float* grad_input, int size);

torch::Tensor sigmoid_forward(torch::Tensor input) {
    TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on CUDA");
    input = input.contiguous();
    auto output = torch::empty_like(input);
    sigmoid_launcher(input.data_ptr<float>(), output.data_ptr<float>(), input.numel());
    return output;
}

torch::Tensor sigmoid_backward(torch::Tensor grad_output, torch::Tensor output) {
    TORCH_CHECK(grad_output.device().is_cuda(), "grad_output must be on CUDA");
    TORCH_CHECK(output.device().is_cuda(), "output must be on CUDA");
    
    grad_output = grad_output.contiguous();
    output = output.contiguous();
    
    auto grad_input = torch::empty_like(output);
    sigmoid_backward_launcher(grad_output.data_ptr<float>(), output.data_ptr<float>(), grad_input.data_ptr<float>(), output.numel());
    return grad_input;
}
"""

def main():
    if not torch.cuda.is_available():
        print("CUDA is not available on this machine.")
        return

    print("Compiling CUDA kernel...")
    # 3. Compile and Load
    try:
        square_cuda = load_inline(
            name='square_cuda',
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=['sigmoid_forward', 'sigmoid_backward'],
            with_cuda=True,
            # Add flag to allow newer, unsupported MSVC versions.
            extra_cuda_cflags=["-O2"],
            verbose=True
        )
    except Exception as e:
        print("\n❌ Compilation failed!")
        error_str = str(e).lower()
        if "cl" in error_str and "where" in error_str:
            print("Error: The Microsoft Visual C++ compiler (cl.exe) was not found.")
            print("Tip: Run this script from the 'x64 Native Tools Command Prompt for VS 20xx'.")
            print("     Ensure 'Desktop development with C++' is installed in Visual Studio.")
        elif "unsupported microsoft visual studio version" in error_str:
            print("Error: Your MSVC version is not officially supported by your CUDA toolkit.")
            print("Tip: The '-allow-unsupported-compiler' flag was used, but compilation still failed.")
            print("     Consider downgrading Visual Studio to a supported version (e.g., 2022, 2019) or upgrading your CUDA Toolkit.")
        raise e
    print("Compilation complete.\n")

    # 4. Test the Kernel
    
    # Define the Autograd Function to link Forward and Backward
    class SigmoidFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            output = square_cuda.sigmoid_forward(input)
            # Save OUTPUT for backward pass (optimization)
            ctx.save_for_backward(output)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            # Retrieve saved output
            output, = ctx.saved_tensors
            # Compute gradient using our custom backward kernel
            return square_cuda.sigmoid_backward(grad_output, output)

    # Wrapper function to make it look like a normal PyTorch function
    def sigmoid(input):
        return SigmoidFunction.apply(input)

    size = 10_000_000
    x = torch.randn(size, device='cuda', requires_grad=True)

    print(f"Benchmarking on vector of size {size:_}...")
    
    # Warmup
    sigmoid(x)
    torch.cuda.synchronize()

    # Run Custom Kernel
    start_time = time.perf_counter()
    y_custom = sigmoid(x)
    # Trigger backward pass to test gradients
    loss = y_custom.sum()
    loss.backward()
    torch.cuda.synchronize()
    print(f"Custom Kernel (Forward + Backward) time:  {(time.perf_counter() - start_time) * 1000:.4f} ms")

    # Run PyTorch Native
    start_time = time.perf_counter()
    y_native = torch.sigmoid(x)
    torch.cuda.synchronize()
    print(f"PyTorch Native time: {(time.perf_counter() - start_time) * 1000:.4f} ms")

    # Verify
    if torch.allclose(y_custom, y_native):
        print("\n✅ Success! Results match.")
    else:
        print("\n❌ Error: Results do not match.")
        
    # Verify Gradients
    expected_grad = y_native * (1 - y_native)
    if torch.allclose(x.grad, expected_grad):
        print("✅ Success! Gradients match.")
    else:
        print("❌ Error: Gradients do not match.")

if __name__ == "__main__":
    main()