#![recursion_limit = "256"]
mod model;
mod data;

#[cfg(feature = "wgpu")]
use burn::backend::{Wgpu, wgpu::{WgpuDevice}};
#[cfg(feature = "cuda")]
use burn::backend::{Cuda, cuda::CudaDevice};

use burn::backend::Autodiff;
use burn::optim::AdamWConfig;
use burn::grad_clipping::GradientClippingConfig;
use burn::train::LearnerBuilder;
use burn::train::metric::{LossMetric, AccuracyMetric};
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;
use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::record::CompactRecorder;
use crate::data::{OpenWebTextBatcher, OpenWebTextDataset};

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: model::TestCrinaConfig,
    pub optimizer: AdamWConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 8)]
    pub batch_size: usize,
    #[config(default = 0)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn train<B: burn::tensor::backend::AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    std::fs::create_dir_all(artifact_dir).ok();
    config.save(format!("{}/config.json", artifact_dir)).unwrap();

    B::seed(&device, config.seed);

    let batcher_train = OpenWebTextBatcher::<B>::new(device.clone(), 128);
    let batcher_valid = OpenWebTextBatcher::<B::InnerBackend>::new(device.clone(), 128);

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(OpenWebTextDataset::new(0));

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(OpenWebTextDataset::new(1));

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        //.devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let _model_trained = learner.fit(dataloader_train, dataloader_test);
}

fn run<B: burn::tensor::backend::AutodiffBackend>(device: B::Device, artifact_dir: &str) {
    let config = TrainingConfig::new(
        model::TestCrinaConfig::new(256, 256, 12, 4), // vocab, d_model, layers, depth
        AdamWConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(1.0))),
    );

    train::<B>(
        artifact_dir,
        config,
        device,
    );
}

fn main() {
    let (chrome_layer, _guard) = tracing_chrome::ChromeLayerBuilder::new().build();
    let filter = EnvFilter::from_default_env()
        .add_directive("crina_burn=trace".parse().unwrap());
    tracing_subscriber::registry().with(filter).with(chrome_layer).init();
    println!("Tracing enabled. A trace-*.json file will be saved in the current directory.");

    let artifact_dir = "crina-checkpoints";

    #[cfg(feature = "cuda")]
    {
        println!("CUDA is available, using Cuda backend");
        run::<Autodiff<Cuda>>(CudaDevice::default(), artifact_dir);
        return;
    }

    #[cfg(feature = "wgpu")]
    {
        println!("Using Wgpu backend (default fallback)");
        run::<Autodiff<Wgpu>>(WgpuDevice::default(), artifact_dir);
    }

    #[cfg(not(any(feature = "cuda", feature = "wgpu")))]
    {
        compile_error!("Please enable at least one of the following features: cuda, wgpu");
    }
}