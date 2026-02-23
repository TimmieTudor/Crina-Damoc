use burn::{
    config::Config,
    module::{Module, Param},
    nn::{Linear, LinearConfig, LayerNorm, LayerNormConfig},
    tensor::{backend::Backend, Tensor, Distribution, activation},
};

// --- ALIF Neuron ---

#[derive(Config, Debug)]
pub struct ALIFConfig {
    pub size: usize,
    #[config(default = 0.5)]
    pub threshold: f32,
    #[config(default = 0.99)]
    pub tau_mem: f32,
    #[config(default = 0.95)]
    pub tau_adapt: f32,
    #[config(default = 0.1)]
    pub adapt_strength: f32,
}

#[derive(Module, Debug)]
pub struct ALIFNeuron<B: Backend> {
    pub size: usize,
    pub base_threshold: Param<Tensor<B, 1>>,
    pub adapt_strength: Param<Tensor<B, 1>>,
    pub tau_mem: f32,
    pub tau_adapt: f32,
}

// Explicit state struct: No hidden overwrites!
#[derive(Debug, Clone)]
pub struct ALIFState<B: Backend> {
    pub v: Tensor<B, 3>,     // [1, 1, size]
    pub adapt: Tensor<B, 3>, // [1, 1, size]
}

impl<B: Backend> ALIFNeuron<B> {
    pub fn new(config: &ALIFConfig, device: &B::Device) -> Self {
        let base_threshold = Tensor::ones([config.size], device) * config.threshold;
        let adapt_strength = Tensor::ones([config.size], device) * config.adapt_strength;

        Self {
            size: config.size,
            base_threshold: Param::from_tensor(base_threshold),
            adapt_strength: Param::from_tensor(adapt_strength),
            tau_mem: config.tau_mem,
            tau_adapt: config.tau_adapt,
        }
    }

    // Initialize state (zeros)
    pub fn init_state(&self, device: &B::Device) -> ALIFState<B> {
        ALIFState {
            v: Tensor::zeros([1, 1, self.size], device),
            adapt: Tensor::zeros([1, 1, self.size], device),
        }
    }

    // Forward returns (Spikes, NextState)
    pub fn forward(&self, input: Tensor<B, 3>, state: ALIFState<B>) -> (Tensor<B, 3>, ALIFState<B>) {
        // Expand state to match input batch [B, N, D]
        let [batch_size, num_nodes, _] = input.dims();
        let v_prev = state.v.clone().expand([batch_size, num_nodes, self.size]);
        let adapt_prev = state.adapt.clone().expand([batch_size, num_nodes, self.size]);

        // Dynamics
        let v_next = v_prev.mul_scalar(self.tau_mem).add(input);
        
        let threshold = self.base_threshold.val().unsqueeze::<2>().unsqueeze::<3>() 
            + adapt_prev.clone();

        // Hard spike
        let spike = v_next.clone().greater_equal(threshold.clone()).float();

        // Reset membrane potential
        let v_reset = v_next.sub(spike.clone().mul(threshold));

        // Update adaptation
        let adapt_next = adapt_prev.mul_scalar(self.tau_adapt) 
            + spike.clone().mul(self.adapt_strength.val().unsqueeze::<2>().unsqueeze::<3>());

        // State update logic: Preserve Batch and Node dimensions!
        // Do NOT mean-pool here, otherwise we lose per-sample history.
        let v_next_state = v_reset.clone().detach();
        let adapt_next_state = adapt_next.clone().detach();

        let next_state = ALIFState {
            v: v_next_state,
            adapt: adapt_next_state,
        };

        (spike, next_state)
    }
}

// --- Tree Self Attention ---

#[derive(Config, Debug)]
pub struct TreeAttentionConfig {
    pub d_model: usize,
    pub tree_depth: usize,
}

#[derive(Module, Debug)]
pub struct TreeSelfAttention<B: Backend> {
    d_model: usize,
    num_nodes: usize,
    tree_depth: usize,
    // Optimization: Single large linear layer for all nodes instead of Vec<Linear>
    node_proj: Linear<B>, 
    lif: ALIFNeuron<B>,
    node_weights: Param<Tensor<B, 2>>, // [num_nodes, d_model]
    norm: LayerNorm<B>,
}

impl<B: Backend> TreeSelfAttention<B> {
    pub fn new(config: &TreeAttentionConfig, device: &B::Device) -> Self {
        let num_nodes = (1 << config.tree_depth) - 1;
        
        // One projection for all nodes: Input 2*D -> Output num_nodes * D
        // We will reshape output to [B, num_nodes, D]
        let node_proj = LinearConfig::new(config.d_model * 2, config.d_model)
            .init(device);
            
        let lif = ALIFNeuron::new(&ALIFConfig::new(config.d_model), device);
        
        let node_weights = Tensor::random(
            [num_nodes, config.d_model], 
            Distribution::Normal(0.0, 0.02), 
            device
        );

        let norm = LayerNormConfig::new(config.d_model).init(device);

        Self {
            d_model: config.d_model,
            num_nodes,
            tree_depth: config.tree_depth,
            node_proj,
            lif,
            node_weights: Param::from_tensor(node_weights),
            norm,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, state: ALIFState<B>) -> (Tensor<B, 3>, ALIFState<B>) {
        let [batch, seq_len, d_model] = x.dims();
        let device = x.device();

        // 1. Pad sequence length to power of 2
        let padded_len = 1 << (seq_len as f32).log2().ceil() as usize;
        let x_padded = if padded_len > seq_len {
            // Simple zero padding for demo (Burn has specific pad ops or cat)
            let padding = Tensor::zeros([batch, padded_len - seq_len, d_model], &device);
            Tensor::cat(vec![x.clone(), padding], 1)
        } else {
            x.clone()
        };

        // 2. Leaves
        let eff_depth = usize::min(self.tree_depth, (padded_len as f32).log2() as usize);
        // Leaves are at the bottom level (eff_depth - 1). Capacity is 2^(eff_depth - 1).
        let num_leaves = 1 << (eff_depth - 1);
        let sub_seq = padded_len / num_leaves;

        // Mean pool to get leaves [B, num_leaves, D]
        let leaves = x_padded
            .reshape([batch, num_leaves, sub_seq, d_model])
            .mean_dim(2)
            .reshape([batch, num_leaves, d_model]);

        // 3. Tree Computation (Vectorized)
        // We'll store all node states in a tensor [B, num_nodes, D]
        // Initialize with zeros
        let mut node_states: Tensor<B, 3> = Tensor::zeros([batch, self.num_nodes, d_model], &device);
        
        // Prepare next persistent state containers (initialized with current state, will be overwritten)
        let mut next_state_v = state.v.clone();
        let mut next_state_adapt = state.adapt.clone();
        
        // Fill leaves into the state tensor
        let leaf_start = (1 << (eff_depth - 1)) - 1;
        
        // 3a. Process Leaves
        // Input to leaf node is concatenation of leaf vector with itself (as per Python code)
        let leaves_input = Tensor::cat(vec![leaves.clone(), leaves.clone()], 2);
        let leaves_proj = self.node_proj.forward(leaves_input);
        
        // Extract state for leaves: [B, num_leaves, D]
        let leaves_state = ALIFState {
            v: state.v.clone().slice([0..batch, leaf_start..(leaf_start + num_leaves), 0..d_model]),
            adapt: state.adapt.clone().slice([0..batch, leaf_start..(leaf_start + num_leaves), 0..d_model]),
        };

        let (leaves_spikes, leaves_next_state) = self.lif.forward(leaves_proj.clone(), leaves_state);
        
        // Store processed leaves in the main tensor
        // node_states[leaf_start..leaf_start+num_leaves] = leaves_spikes * leaves_proj (simplified to just spikes for now or proj*spike)
        // The Python code stores `proj * spike`.
        let leaves_out = leaves_spikes.clone().mul(leaves_proj); // Re-calculate proj not needed if we kept it, but for clarity
        
        node_states = node_states.slice_assign(
            [0..batch, leaf_start..(leaf_start + num_leaves), 0..d_model], 
            leaves_out
        );

        // Update persistent state for leaves
        next_state_v = next_state_v.slice_assign(
            [0..batch, leaf_start..(leaf_start + num_leaves), 0..d_model], 
            leaves_next_state.v
        );
        next_state_adapt = next_state_adapt.slice_assign(
            [0..batch, leaf_start..(leaf_start + num_leaves), 0..d_model], 
            leaves_next_state.adapt
        );

        // 3b. Process Internal Nodes (Bottom-Up)
        // Iterate from level above leaves (eff_depth - 2) down to root (0)
        if eff_depth > 1 {
            for level in (0..=(eff_depth - 2)).rev() {
                let level_start = (1 << level) - 1;
                let level_len = 1 << level;
                
                // Children are at the next level
                let children_start = (1 << (level + 1)) - 1;
                let children_len = 1 << (level + 1);
                
                // Extract children outputs [B, children_len, D]
                let children = node_states.clone().slice([0..batch, children_start..(children_start + children_len), 0..d_model]);
                
                // Reshape to separate left and right children: [B, level_len, 2, D]
                let children_paired = children.reshape([batch, level_len, 2, d_model]);
                
                // Split and concatenate [B, level_len, 2*D]
                let left = children_paired.clone().slice([0..batch, 0..level_len, 0..1, 0..d_model]).squeeze::<2>();
                let right = children_paired.slice([0..batch, 0..level_len, 1..2, 0..d_model]).squeeze::<2>();
                let fused = Tensor::cat(vec![left, right], 2);
                
                // Compute Node Logic
                let proj = self.node_proj.forward(fused).unsqueeze::<3>();
                
                // Extract state for this level
                let level_state = ALIFState {
                    v: state.v.clone().slice([0..batch, level_start..(level_start + level_len), 0..d_model]),
                    adapt: state.adapt.clone().slice([0..batch, level_start..(level_start + level_len), 0..d_model]),
                };

                let (spikes, level_next_state) = self.lif.forward(proj.clone(), level_state);
                
                let node_out = spikes.mul(proj);
                
                // Store in node_states
                node_states = node_states.slice_assign(
                    [0..batch, level_start..(level_start + level_len), 0..d_model], 
                    node_out
                );

                // Update persistent state for this level
                next_state_v = next_state_v.slice_assign(
                    [0..batch, level_start..(level_start + level_len), 0..d_model], 
                    level_next_state.v
                );
                next_state_adapt = next_state_adapt.slice_assign(
                    [0..batch, level_start..(level_start + level_len), 0..d_model], 
                    level_next_state.adapt
                );
            }
        }

        // 4. Mixture (Weighted Sum)
        // [B, num_leaves, D] * [num_leaves, D] (broadcast)
        // For the full tree, you would gather all nodes.
        
        // We use the active nodes (from 0 to leaf_end)
        let active_count = leaf_start + num_leaves;
        let active_states = node_states.slice([0..batch, 0..active_count, 0..d_model]);
        
        // Apply Softmax to weights (dim 0 = nodes)
        let weights = activation::softmax(self.node_weights.val().slice([0..active_count]), 0).unsqueeze::<3>(); // [1, N, D]
        
        let weighted = active_states.mul(weights); // [B, N, D]
        let mixture = weighted.sum_dim(1); // [B, 1, D]

        // 5. Residual + Norm
        let output = mixture.expand([batch, seq_len, d_model]);
        let output = self.norm.forward(output.add(x));

        (output, ALIFState { v: next_state_v, adapt: next_state_adapt })
    }
}

// --- Full Model Wrapper ---

#[derive(Module, Debug)]
pub struct CrinaSynapse<B: Backend> {
    attn: TreeSelfAttention<B>,
    // Add FeedForward, Embedding, etc.
}

impl<B: Backend> CrinaSynapse<B> {
    pub fn new(d_model: usize, tree_depth: usize, device: &B::Device) -> Self {
        Self {
            attn: TreeSelfAttention::new(&TreeAttentionConfig { d_model, tree_depth }, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, state: ALIFState<B>) -> (Tensor<B, 3>, ALIFState<B>) {
        self.attn.forward(x, state)
    }

    pub fn init_state(&self, device: &B::Device) -> ALIFState<B> {
        // Initialize state for ALL nodes [1, num_nodes, D]
        ALIFState {
            v: Tensor::zeros([1, self.attn.num_nodes, self.attn.d_model], device),
            adapt: Tensor::zeros([1, self.attn.num_nodes, self.attn.d_model], device),
        }
    }
}