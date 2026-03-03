use burn::{
    config::Config,
    module::{Module, Param},
    nn::{
        loss::CrossEntropyLossConfig,
        Embedding, EmbeddingConfig, Linear, LinearConfig,
    },
    tensor::{backend::Backend, Tensor, Int},
    train::{
        metric::{AccuracyMetric, LossMetric},
        TrainOutput, TrainStep, ValidStep, ClassificationOutput,
    },
};
use crate::data::OpenWebTextBatch;

// --- RMSNorm Implementation ---
#[derive(Config, Debug)]
pub struct RMSNormConfig {
    pub d_model: usize,
    #[config(default = 1e-5)]
    pub eps: f32,
}

#[derive(Module, Debug)]
pub struct RMSNorm<B: Backend> {
    weight: Param<Tensor<B, 1>>,
    eps: f32,
}

impl<B: Backend> RMSNorm<B> {
    pub fn new(config: &RMSNormConfig, device: &B::Device) -> Self {
        let weight = Tensor::ones([config.d_model], device);
        Self {
            weight: Param::from_tensor(weight),
            eps: config.eps,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // rms = sqrt(mean(x^2) + eps)
        let rms = x.clone().powf_scalar(2.0).mean_dim(2).add_scalar(self.eps).sqrt();
        let rms = rms.clamp_min(1e-4);

        // x / rms * weight
        let x_norm = x.div(rms);
        let weight = self.weight.val().unsqueeze_dims(&[0, 1]);
        x_norm.mul(weight)
    }
}

// --- LIF Neuron with Surrogate Gradients ---

#[derive(Config, Debug)]
pub struct LIFConfig {
    pub size: usize,
    #[config(default = 5.0)]
    pub alpha: f32,
}

#[derive(Module, Debug)]
pub struct LIFNeuron<B: Backend> {
    pub size: usize,
    pub threshold: Param<Tensor<B, 1>>,
    pub tau: Param<Tensor<B, 1>>,
    pub v_reset: Param<Tensor<B, 3>>,
    pub alpha: f32,
}


impl<B: Backend> LIFNeuron<B> {
    pub fn new(config: &LIFConfig, device: &B::Device) -> Self {
        let threshold = Tensor::ones([config.size], device);
        let tau = Tensor::ones([config.size], device) * 0.5;
        let v_reset = Tensor::zeros([1, 1, config.size], device);

        Self {
            size: config.size,
            threshold: Param::from_tensor(threshold),
            tau: Param::from_tensor(tau),
            v_reset: Param::from_tensor(v_reset),
            alpha: config.alpha,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, v_prev: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [batch, seq, _] = x.dims();
        
        // Implementation of LIF Dynamics: v = tau * v_prev + x
        let tau = self.tau.val().unsqueeze_dims(&[0, 1]);
        let v_next_raw = v_prev.mul(tau).add(x);
        
        let threshold = self.threshold.val().unsqueeze_dims(&[0, 1]);
        
        // Surrogate Gradient using Straight-Through Estimator trick
        // spike = (v >= threshold)
        // Backprop uses Fast Sigmoid: alpha / (1 + alpha * |v - threshold|)^2
        
        let diff = v_next_raw.clone().sub(threshold.clone());
        let hard_spike = v_next_raw.clone().greater_equal(threshold.clone()).float();
        
        // Simulated Fast Sigmoid (Normalized)
        let alpha = self.alpha;
        let surr = diff.clone().abs().mul_scalar(alpha).add_scalar(1.0).powf_scalar(-2.0).mul_scalar(alpha).clamp(0.0, 1.0);
        
        // Straight-through: spike = (hard_spike - surr).detach() + surr
        let spike = hard_spike.clone().sub(surr.clone()).detach().add(surr);
        
        // Reset: v_next = v.masked_fill(mask, 0) + v_reset * spike
        let mask = v_next_raw.clone().greater_equal(threshold).float();
        let v_next = v_next_raw.mul(mask.clone().neg().add_scalar(1.0))
            .add(spike.clone().mul(self.v_reset.val()));
            
        // Return (Spikes, v_persistent)
        // Like in Python: self.v = v_next[:, -1:, :].detach()
        let v_persistent = v_next.clone().slice([0..batch, (seq-1)..seq, 0..self.size]);
        
        // Normalize spikes to prevent unbounded activations (matches Python reference)
        // denom = spike.sum(dim=-1, keepdim=True).clamp(min=1.0)
        let denom = spike.clone().sum_dim(2).clamp_min(1.0);
        let spike = spike.div(denom);

        (spike, v_persistent)
    }
}

// --- Tree Attention with Residuals ---

#[derive(Module, Debug)]
pub struct TreeAttention<B: Backend> {
    d_model: usize,
    tree_depth: usize,
    up_projs: Vec<Linear<B>>,
    up_lifs: Vec<LIFNeuron<B>>,
    up_norms: Vec<RMSNorm<B>>,
    down_projs: Vec<Linear<B>>,
    down_lifs: Vec<LIFNeuron<B>>,
    down_norms: Vec<RMSNorm<B>>,
    leaf_proj: Linear<B>,
    leaf_lif: LIFNeuron<B>,
    leaf_norm: RMSNorm<B>,
}

#[derive(Debug, Clone)]
pub struct TreeState<B: Backend> {
    pub up_v: Vec<Tensor<B, 3>>,
    pub down_v: Vec<Tensor<B, 3>>,
    pub leaf_v: Tensor<B, 3>,
}

impl<B: Backend> TreeAttention<B> {
    pub fn new(d_model: usize, tree_depth: usize, device: &B::Device) -> Self {
        let mut up_projs = Vec::new();
        let mut up_lifs = Vec::new();
        let mut up_norms = Vec::new();
        let mut down_projs = Vec::new();
        let mut down_lifs = Vec::new();
        let mut down_norms = Vec::new();

        for _ in 0..tree_depth {
            up_projs.push(LinearConfig::new(d_model * 2, d_model).init(device));
            up_lifs.push(LIFNeuron::new(&LIFConfig::new(d_model), device));
            up_norms.push(RMSNorm::new(&RMSNormConfig::new(d_model), device));
            
            down_projs.push(LinearConfig::new(d_model * 2, d_model).init(device));
            down_lifs.push(LIFNeuron::new(&LIFConfig::new(d_model), device));
            down_norms.push(RMSNorm::new(&RMSNormConfig::new(d_model), device));
        }

        let leaf_proj = LinearConfig::new(d_model * 2, d_model).init(device);
        let leaf_lif = LIFNeuron::new(&LIFConfig::new(d_model), device);
        let leaf_norm = RMSNorm::new(&RMSNormConfig::new(d_model), device);

        Self {
            d_model,
            tree_depth,
            up_projs,
            up_lifs,
            up_norms,
            down_projs,
            down_lifs,
            down_norms,
            leaf_proj,
            leaf_lif,
            leaf_norm,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, state: TreeState<B>) -> (Tensor<B, 3>, TreeState<B>) {
        let [batch, seq, d_model] = x.dims();
        let num_levels = (seq as f32).log2() as usize;
        
        // --- PHASE 1: UP-SWEEP ---
        let mut summaries = vec![x.clone()];
        let mut next_up_v = state.up_v.clone();
        
        for l in 0..num_levels {
            let curr = summaries.last().unwrap();
            let [b, t, d] = curr.dims();
            if t < 2 { break; }
            
            let merged = curr.clone().reshape([b, t / 2, 2 * d]);
            let res = merged.clone().slice([0..b, 0..(t/2), 0..d])
                .add(merged.clone().slice([0..b, 0..(t/2), d..(2*d)]))
                .mul_scalar(0.5);
                
            let proj_idx = if l < self.tree_depth { l } else { self.tree_depth - 1 };
            
            let x_proj = self.up_projs[proj_idx].forward(merged);
            let x_norm = self.up_norms[proj_idx].forward(x_proj);
            let (spike, v_next) = self.up_lifs[proj_idx].forward(x_norm, state.up_v[proj_idx].clone());
            
            summaries.push(spike.add(res));
            next_up_v[proj_idx] = v_next; // Accumulate in the shared state slot
        }
        
        // --- PHASE 2: DOWN-SWEEP ---
        let mut contexts = vec![None; num_levels + 1];
        contexts[num_levels] = Some(Tensor::zeros([batch, 1, d_model], &x.device()));
        let mut next_down_v = state.down_v.clone();
        
        for l in (1..=num_levels).rev() {
            let parent_ctx = contexts[l].clone().unwrap();
            let child_summaries = summaries[l-1].clone().reshape([batch, parent_ctx.dims()[1], 2, d_model]);
            
            let left_child = child_summaries.clone().slice([0..batch, 0..parent_ctx.dims()[1], 0..1, 0..d_model])
                .reshape([batch, parent_ctx.dims()[1], d_model]);
            let combined = Tensor::cat(vec![parent_ctx.clone(), left_child], 2);
            
            let proj_idx = if (l-1) < self.tree_depth { l-1 } else { self.tree_depth - 1 };
            
            let x_proj = self.down_projs[proj_idx].forward(combined);
            let x_norm = self.down_norms[proj_idx].forward(x_proj);
            let (ctx_right, v_next) = self.down_lifs[proj_idx].forward(x_norm, state.down_v[proj_idx].clone());
            
            let ctx_right = ctx_right.add(parent_ctx.clone());
            
            // Interleave
            let ctx_interleaved = Tensor::cat(vec![parent_ctx.clone().unsqueeze_dim::<4>(2), ctx_right.unsqueeze_dim::<4>(2)], 2)
                .reshape([batch, parent_ctx.dims()[1] * 2, d_model]);
            contexts[l-1] = Some(ctx_interleaved);
            
            next_down_v[proj_idx] = v_next;
        }
        
        // --- PHASE 3: LEAF MIXING ---
        let combined_leaf = Tensor::cat(vec![x, contexts[0].clone().unwrap()], 2);
        let x_proj = self.leaf_proj.forward(combined_leaf);
        let x_norm = self.leaf_norm.forward(x_proj);
        let (output, next_leaf_v) = self.leaf_lif.forward(x_norm, state.leaf_v);
        
        let next_state = TreeState {
            up_v: next_up_v,
            down_v: next_down_v,
            leaf_v: next_leaf_v,
        };
        
        (output, next_state)
    }

    pub fn init_state(&self, batch: usize, device: &B::Device) -> TreeState<B> {
        TreeState {
            up_v: (0..self.tree_depth).map(|_| Tensor::zeros([batch, 1, self.d_model], device)).collect(),
            down_v: (0..self.tree_depth).map(|_| Tensor::zeros([batch, 1, self.d_model], device)).collect(),
            leaf_v: Tensor::zeros([batch, 1, self.d_model], device),
        }
    }
}

// --- FeedForward SNN ---

#[derive(Module, Debug)]
pub struct FeedForwardSNN<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    lif: LIFNeuron<B>,
}

impl<B: Backend> FeedForwardSNN<B> {
    pub fn new(d_model: usize, device: &B::Device) -> Self {
        Self {
            linear1: LinearConfig::new(d_model, d_model * 4).init(device),
            linear2: LinearConfig::new(d_model * 4, d_model).init(device),
            lif: LIFNeuron::new(&LIFConfig::new(d_model * 4), device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, v_prev: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let x = self.linear1.forward(x);
        let (spike, v_next) = self.lif.forward(x, v_prev);
        let x = self.linear2.forward(spike);
        (x, v_next)
    }
}

// --- Test Layer (Total Transformer-style block) ---

#[derive(Module, Debug)]
pub struct TestLayer<B: Backend> {
    pre_norm_attn: RMSNorm<B>,
    attention: TreeAttention<B>,
    post_norm_attn: RMSNorm<B>,
    pre_norm_ffn: RMSNorm<B>,
    feed_forward: FeedForwardSNN<B>,
    post_norm_ffn: RMSNorm<B>,
    residual_scale_attn: Param<Tensor<B, 1>>,
    residual_scale_ffn: Param<Tensor<B, 1>>,
}

#[derive(Debug, Clone)]
pub struct LayerState<B: Backend> {
    pub attn_state: TreeState<B>,
    pub ffn_v: Tensor<B, 3>,
}

impl<B: Backend> TestLayer<B> {
    pub fn new(d_model: usize, tree_depth: usize, device: &B::Device) -> Self {
        Self {
            pre_norm_attn: RMSNorm::new(&RMSNormConfig::new(d_model), device),
            attention: TreeAttention::new(d_model, tree_depth, device),
            post_norm_attn: RMSNorm::new(&RMSNormConfig::new(d_model), device),
            pre_norm_ffn: RMSNorm::new(&RMSNormConfig::new(d_model), device),
            feed_forward: FeedForwardSNN::new(d_model, device),
            post_norm_ffn: RMSNorm::new(&RMSNormConfig::new(d_model), device),
            residual_scale_attn: Param::from_tensor(Tensor::ones([1], device) * 0.01),
            residual_scale_ffn: Param::from_tensor(Tensor::ones([1], device) * 0.01),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, state: LayerState<B>) -> (Tensor<B, 3>, LayerState<B>) {
        // Attention Branch
        let x_attn = self.pre_norm_attn.forward(x.clone());
        let (attn_out, next_attn_state) = self.attention.forward(x_attn, state.attn_state);
        let x = x.add(self.post_norm_attn.forward(attn_out).mul(self.residual_scale_attn.val().unsqueeze_dims(&[0, 1])));

        // FFN Branch
        let x_ffn = self.pre_norm_ffn.forward(x.clone());
        let (ffn_out, next_ffn_v) = self.feed_forward.forward(x_ffn, state.ffn_v);
        let x = x.add(self.post_norm_ffn.forward(ffn_out).mul(self.residual_scale_ffn.val().unsqueeze_dims(&[0, 1])));

        (x, LayerState { attn_state: next_attn_state, ffn_v: next_ffn_v })
    }
}

// --- Main Model: TestCrina ---

#[derive(Config, Debug)]
pub struct TestCrinaConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_layers: usize,
    pub tree_depth: usize,
}

impl TestCrinaConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TestCrina<B> {
        let layers = (0..self.num_layers)
            .map(|_| TestLayer::new(self.d_model, self.tree_depth, device))
            .collect();

        TestCrina {
            embed: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            layers,
            ln_f: RMSNorm::new(&RMSNormConfig::new(self.d_model), device),
            lm_head: LinearConfig::new(self.d_model, self.vocab_size).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct TestCrina<B: Backend> {
    embed: Embedding<B>,
    layers: Vec<TestLayer<B>>,
    ln_f: RMSNorm<B>,
    lm_head: Linear<B>,
}

#[derive(Debug, Clone)]
pub struct CrinaState<B: Backend> {
    pub layer_states: Vec<LayerState<B>>,
}

impl<B: Backend> TestCrina<B> {
    pub fn new(vocab_size: usize, d_model: usize, num_layers: usize, tree_depth: usize, device: &B::Device) -> Self {
        let mut layers = Vec::new();
        for _ in 0..num_layers {
            layers.push(TestLayer::new(d_model, tree_depth, device));
        }

        Self {
            embed: EmbeddingConfig::new(vocab_size, d_model).init(device),
            layers,
            ln_f: RMSNorm::new(&RMSNormConfig::new(d_model), device),
            lm_head: LinearConfig::new(d_model, vocab_size).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2, Int>, state: CrinaState<B>) -> (Tensor<B, 3>, CrinaState<B>) {
        let mut x = self.embed.forward(x);
        let mut next_layer_states = Vec::new();
        
        for (i, layer) in self.layers.iter().enumerate() {
            let (out, next_s) = layer.forward(x, state.layer_states[i].clone());
            x = out;
            next_layer_states.push(next_s);
        }
        
        let x = self.lm_head.forward(self.ln_f.forward(x));
        (x, CrinaState { layer_states: next_layer_states })
    }

    pub fn init_state(&self, batch: usize, device: &B::Device) -> CrinaState<B> {
        let layer_states = self.layers.iter().map(|l| LayerState {
            attn_state: l.attention.init_state(batch, device),
            ffn_v: Tensor::zeros([batch, 1, l.attention.d_model * 4], device),
        }).collect();
        
        CrinaState { layer_states }
    }
}

// --- Training ---

impl<B: Backend> TestCrina<B> {
    pub fn forward_classification(&self, item: OpenWebTextBatch<B>) -> ClassificationOutput<B> {
        let [batch, seq] = item.inputs.dims();
        let state = self.init_state(batch, &item.inputs.device());
        
        let (logits, _) = self.forward(item.inputs.clone(), state); // [B, T, V]
        
        // Flatten for CrossEntropy: [B * T, V]
        let logits_flat = logits.reshape([batch * seq, 256]);
        let targets_flat = item.targets.clone().reshape([batch * seq]);
        
        let loss = CrossEntropyLossConfig::new()
            .init(&item.targets.device())
            .forward(logits_flat.clone(), targets_flat.clone());
            
        ClassificationOutput::new(loss, logits_flat, targets_flat)
    }
}

impl<B: burn::tensor::backend::AutodiffBackend> TrainStep<OpenWebTextBatch<B>, ClassificationOutput<B>> for TestCrina<B> {
    fn step(&self, item: OpenWebTextBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let output = self.forward_classification(item);
        TrainOutput::new(self, output.loss.backward(), output)
    }
}

impl<B: Backend> ValidStep<OpenWebTextBatch<B>, ClassificationOutput<B>> for TestCrina<B> {
    fn step(&self, item: OpenWebTextBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(item)
    }
}