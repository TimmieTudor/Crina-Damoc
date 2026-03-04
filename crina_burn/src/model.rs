use burn::{
    config::Config,
    module::{Module, Param},
    nn::{
        loss::CrossEntropyLossConfig,
        Embedding, EmbeddingConfig, Linear, LinearConfig,
    },
    tensor::{backend::Backend, activation::sigmoid, Tensor, Int},
    train::{
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
        // Optimization: use mul instead of powf_scalar for speed
        let x_sq = x.clone().mul(x.clone());
        let rms = x_sq.mean_dim(2).add_scalar(self.eps).sqrt();
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
    #[config(default = 1.0)]
    pub gain: f32,
}

#[derive(Module, Debug)]
pub struct LIFNeuron<B: Backend> {
    pub size: usize,
    pub threshold: Param<Tensor<B, 1>>,
    pub tau: Param<Tensor<B, 1>>,
    pub v_reset: Param<Tensor<B, 3>>,
    pub alpha: f32,
    pub gain: f32,
    pub surr_scale: f32, // alpha * gain
    pub steps_const: Tensor<B, 3>,
    pub persist_const: Tensor<B, 3>,
}


impl<B: Backend> LIFNeuron<B> {
    pub fn new(config: &LIFConfig, device: &B::Device) -> Self {
        let threshold = Tensor::ones([config.size], device);
        let tau = Tensor::ones([config.size], device) * 0.5;
        let v_reset = Tensor::zeros([1, 1, config.size], device);

        // Pre-compute constant tensors to reduce overhead
        let steps_const = Tensor::<B, 1, Int>::arange(0..32, device).float().reshape([1, 32, 1]);
        let persist_const = Tensor::<B, 1, Int>::arange(1..(512 + 1) as i64, device).float().reshape([1, 512, 1]);

        Self {
            size: config.size,
            threshold: Param::from_tensor(threshold),
            tau: Param::from_tensor(tau),
            v_reset: Param::from_tensor(v_reset),
            alpha: config.alpha,
            gain: config.gain,
            surr_scale: config.alpha * config.gain,
            steps_const,
            persist_const,
        }
    }

    #[tracing::instrument(level = "info", skip_all)]
    pub fn forward(&self, x: Tensor<B, 3>, v_prev: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [batch, seq, d_model] = x.dims();
        let threshold = self.threshold.val().unsqueeze_dims(&[0, 1]);
        
        // 1. Parallel Integration (SRM - Spike Response Model)
        let k = 32;
        let k_actual = if seq < k { seq } else { k };
        
        // steps_flip = (k_actual - 1) - steps
        let steps_flip = self.steps_const.clone()
            .slice([0..1, 0..k_actual, 0..1])
            .mul_scalar(-1.0)
            .add_scalar((k_actual - 1) as f32);
        
        // Kernel: tau^steps. Shape: [D, 1, K]
        // Optimization: avoid redundant unsqueezes
        let tau_vec = self.tau.val().reshape([d_model, 1, 1]);
        let log_tau = tau_vec.clone().clamp(0.001, 0.999).log();
        let kernel = log_tau.mul(steps_flip.reshape([1, 1, k_actual])).exp();
        
        let x_trans = x.transpose();
        let conv_options = burn::tensor::ops::ConvOptions::new([1], [k_actual - 1], [1], d_model);
        let v_integrated_raw = burn::tensor::module::conv1d(x_trans, kernel, None, conv_options);
        
        let v_integrated = v_integrated_raw.slice([0..batch, 0..d_model, 0..seq]).transpose();

        // Add persistence (decayed) using pre-computed persist_const
        let persist_steps = self.persist_const.clone().slice([0..1, 0..seq, 0..1]);
        let tau_unsqueezed = self.tau.val().unsqueeze_dims(&[0, 1]).clamp(0.001, 0.999);
        let persist_decay = tau_unsqueezed.log().mul(persist_steps).exp();
        let v_persist = v_prev.mul(persist_decay);
        let v_integrated = v_integrated.add(v_persist);

        // 2. Vectorized Spiking & Soft Reset
        let threshold_v = threshold.clone();
        let diff = v_integrated.clone().sub(threshold_v.clone());
        let hard_spike = v_integrated.clone().greater_equal(threshold_v.clone()).float();
        
        // Optimization: avoid powf_scalar(-2.0), use mul and recip
        let surr_base = diff.clone().abs().mul_scalar(self.alpha).add_scalar(1.0);
        let surr = surr_base.clone().mul(surr_base).recip().mul_scalar(self.surr_scale).clamp(0.0, 1.0);
        let spike_seq = hard_spike.sub(surr.clone()).detach().add(surr);
        
        // Soft Reset approximation
        let v_integrated = v_integrated.sub(spike_seq.clone().detach().mul(threshold_v).mul_scalar(0.5));
        
        // Normalize spikes
        let output = spike_seq.clone().div(spike_seq.sum_dim(2).clamp_min(1.0));
        
        // Persistence
        let v_final = v_integrated.slice([0..batch, (seq - 1)..seq, 0..d_model]);

        (output, v_final)
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
    up_selectors: Vec<Linear<B>>,
    down_selectors: Vec<Linear<B>>,
    leaf_selector: Linear<B>,
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

        let mut up_selectors = Vec::new();
        let mut down_selectors = Vec::new();

        for _ in 0..tree_depth {
            up_projs.push(LinearConfig::new(d_model * 2, d_model).init(device));
            up_lifs.push(LIFNeuron::new(&LIFConfig::new(d_model), device));
            up_norms.push(RMSNorm::new(&RMSNormConfig::new(d_model), device));
            up_selectors.push(LinearConfig::new(d_model * 2, d_model).init(device));
            
            down_projs.push(LinearConfig::new(d_model * 2, d_model).init(device));
            down_lifs.push(LIFNeuron::new(&LIFConfig::new(d_model), device));
            down_norms.push(RMSNorm::new(&RMSNormConfig::new(d_model), device));
            down_selectors.push(LinearConfig::new(d_model * 2, d_model).init(device));
        }

        let leaf_selector = LinearConfig::new(d_model * 2, d_model).init(device);

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
            up_selectors,
            down_selectors,
            leaf_selector,
            leaf_proj,
            leaf_lif,
            leaf_norm,
        }
    }

    #[tracing::instrument(level = "info", skip_all)]
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
            
            let gate = sigmoid(self.up_selectors[proj_idx].forward(merged.clone()));
            let x_proj = self.up_projs[proj_idx].forward(merged);
            let x_norm = self.up_norms[proj_idx].forward(x_proj.clone());
            let (spike, v_next) = self.up_lifs[proj_idx].forward(x_norm, state.up_v[proj_idx].clone());
            let summary_with_skip = x_proj.add(spike); // Skip connection around LIF
            
            // Selection: res + gate * (summary_with_skip - res)
            let summary_next = res.clone().add(gate.mul(summary_with_skip.sub(res)));
            summaries.push(summary_next);
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
            
            let gate = sigmoid(self.down_selectors[proj_idx].forward(combined.clone()));
            let x_proj = self.down_projs[proj_idx].forward(combined);
            let x_norm = self.down_norms[proj_idx].forward(x_proj.clone());
            let (ctx_right_raw, v_next) = self.down_lifs[proj_idx].forward(x_norm, state.down_v[proj_idx].clone());
            let ctx_with_skip = x_proj.add(ctx_right_raw); // Skip connection around LIF
            
            // Selection: parent_ctx + gate * (ctx_with_skip - parent_ctx)
            let ctx_right = parent_ctx.clone().add(gate.mul(ctx_with_skip.sub(parent_ctx.clone())));
            
            // Interleave: [P0, C0, P1, C1, ...]
            // Optimization: use stack + reshape instead of unsqueeze + cat
            let ctx_interleaved = Tensor::stack::<4>(vec![parent_ctx.clone(), ctx_right], 2)
                .reshape([batch, parent_ctx.dims()[1] * 2, d_model]);
            contexts[l-1] = Some(ctx_interleaved);
            
            next_down_v[proj_idx] = v_next;
        }
        
        // --- PHASE 3: LEAF MIXING ---
        let combined_leaf = Tensor::cat(vec![x.clone(), contexts[0].clone().unwrap()], 2);
        let gate = sigmoid(self.leaf_selector.forward(combined_leaf.clone()));
        let x_proj = self.leaf_proj.forward(combined_leaf);
        let x_norm = self.leaf_norm.forward(x_proj.clone());
        let (leaf_output, next_leaf_v) = self.leaf_lif.forward(x_norm, state.leaf_v);
        let leaf_with_skip = x_proj.add(leaf_output); // Skip connection around LIF
        
        // Final Selection: x + gate * (leaf_with_skip - x)
        let output = x.clone().add(gate.mul(leaf_with_skip.sub(x)));
        
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

    #[tracing::instrument(level = "info", skip_all)]
    pub fn forward(&self, x: Tensor<B, 3>, v_prev: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let h = self.linear1.forward(x);
        let (spike, v_next) = self.lif.forward(h.clone(), v_prev);
        let h = h.add(spike); // Skip connection around LIF
        let x = self.linear2.forward(h);
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
            residual_scale_attn: Param::from_tensor(Tensor::ones([1], device) * 0.1),
            residual_scale_ffn: Param::from_tensor(Tensor::ones([1], device) * 0.1),
        }
    }

    #[tracing::instrument(level = "info", skip_all)]
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

    #[tracing::instrument(level = "info", skip_all)]
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
    #[tracing::instrument(level = "info", skip_all)]
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