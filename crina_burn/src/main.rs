#![recursion_limit = "256"]
mod model;

use burn::backend::Wgpu;
use burn::tensor::Tensor;
use model::CrinaSynapse;

fn main() {
    // Use WGPU backend (Metal/Vulkan/DX12)
    type Backend = Wgpu;
    let device = Default::default();

    // Hyperparameters
    let d_model = 256;
    let tree_depth = 4;

    // Initialize Model
    let model = CrinaSynapse::<Backend>::new(d_model, tree_depth, &device);
    
    // Initialize Persistent State explicitly
    let mut state = model.init_state(&device);

    // Dummy Input [Batch=2, Seq=128, Dim=256]
    let input = Tensor::<Backend, 3>::random([2, 128, 256], burn::tensor::Distribution::Default, &device);

    // Forward pass with explicit state handling
    let (output, new_state) = model.forward(input, state);
    
    println!("Output shape: {:?}", output.dims());
    println!("State updated successfully. Old state consumed, new state returned.");
    
    // In the next iteration, you pass `new_state` back in.
    // state = new_state; 
}