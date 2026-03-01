#![recursion_limit = "256"]
mod model;

use burn::backend::Wgpu;
use burn::tensor::Tensor;
use model::TestCrina;

fn main() {
    // Use WGPU backend (Metal/Vulkan/DX12)
    type Backend = Wgpu;
    let device = Default::default();

    // Hyperparameters matching Python implementation
    let vocab_size = 256;
    let d_model = 256;
    let num_layers = 12;
    let tree_depth = 4;

    // Initialize Model
    let model = TestCrina::<Backend>::new(vocab_size, d_model, num_layers, tree_depth, &device);
    
    // Initialize Persistent State explicitly for batch size 2
    let state = model.init_state(2, &device);

    // Dummy Input [Batch=2, Seq=128] of type Int
    let input = Tensor::<Backend, 2, burn::tensor::Int>::random(
        [2, 128], 
        burn::tensor::Distribution::Uniform(0.0, 255.0), 
        &device
    );

    // Forward pass with explicit state handling
    let (output, _new_state) = model.forward(input, state);
    
    println!("Output shape: {:?}", output.dims());
    println!("Success! State updated and gradients are enabled.");
}