use burn::data::dataset::Dataset;
use hf_hub::api::sync::Api;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use std::fs::File;
use burn::tensor::backend::Backend;
use burn::data::dataloader::batcher::Batcher;
use burn::tensor::{Int, TensorData, Tensor};

#[derive(Clone, Debug)]
pub struct OpenWebTextItem {
    pub text: String,
}

pub struct OpenWebTextDataset {
    texts: Vec<String>,
}

impl OpenWebTextDataset {
    /// Loads a specific shard of OpenWebText (0-79)
    pub fn new(shard_idx: usize) -> Self {
        // 1. Initialize HF Hub API
        let api = Api::new().expect("Failed to create HF API");
        let repo = api.dataset("Skylion007/openwebtext".to_string());
        
        // 2. Format the shard name: plain_text/train-000XX-of-00080.parquet
        let filename = format!("plain_text/train-{:05}-of-00080.parquet", shard_idx);
        
        println!("Downloading OpenWebText shard: {}...", filename);
        let path = repo.get(&filename).expect("Failed to download shard");
        
        // 3. Read Parquet file
        let file = File::open(path).expect("Failed to open parquet file");
        let reader = SerializedFileReader::new(file).expect("Failed to create parquet reader");
        
        let mut texts = Vec::new();
        
        // Iterate through rows (OpenWebText parquet usually has one column: "text")
        for row in reader.get_row_iter(None).expect("Failed to get row iterator") {
            if let Ok(row) = row {
                // Get the "text" column
                if let Some(text) = row.get_string(0).ok() {
                    texts.push(text.to_string());
                }
            }
        }
        
        println!("Loaded {} documents from shard {}.", texts.len(), shard_idx);
        
        Self { texts }
    }
}

impl Dataset<OpenWebTextItem> for OpenWebTextDataset {
    fn get(&self, index: usize) -> Option<OpenWebTextItem> {
        self.texts.get(index).map(|t| OpenWebTextItem { text: t.clone() })
    }

    fn len(&self) -> usize {
        self.texts.len()
    }
}

#[derive(Debug, Clone)]
pub struct OpenWebTextBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> OpenWebTextBatch<B> {
    pub fn new(inputs: Tensor<B, 2, Int>, targets: Tensor<B, 2, Int>) -> Self {
        Self { inputs, targets }
    }
}

#[derive(Clone)]
pub struct OpenWebTextBatcher<B: Backend> {
    device: B::Device,
    max_seq_len: usize,
}

impl<B: Backend> OpenWebTextBatcher<B> {
    pub fn new(device: B::Device, max_seq_len: usize) -> Self {
        Self { device, max_seq_len }
    }
}

impl<B: Backend> Batcher<B, OpenWebTextItem, OpenWebTextBatch<B>> for OpenWebTextBatcher<B> {
    #[tracing::instrument(level = "info", skip_all)]
    fn batch(&self, items: Vec<OpenWebTextItem>, device: &B::Device) -> OpenWebTextBatch<B> {
        let batch_size = items.len();
        
        // Flattened buffers for the whole batch
        let mut inputs_flat = vec![0i32; batch_size * self.max_seq_len];
        let mut targets_flat = vec![0i32; batch_size * self.max_seq_len];

        for (b, item) in items.into_iter().enumerate() {
            let bytes = item.text.as_bytes();
            // Take up to max_seq_len + 1 to yield max_seq_len inputs and 1-shifted targets
            let len = bytes.len().min(self.max_seq_len + 1);
            let truncated = &bytes[..len];

            // If we have at least 2 tokens, we can form a pair
            if len >= 2 {
                let actual_len = len - 1;
                for i in 0..actual_len {
                    inputs_flat[b * self.max_seq_len + i] = truncated[i] as i32;
                    targets_flat[b * self.max_seq_len + i] = truncated[i + 1] as i32;
                }
            }
        }

        // Create 2D tensors directly from the flat buffers
        // Shape: [batch_size, max_seq_len]
        let inputs = Tensor::<B, 2, Int>::from_data(
            TensorData::new(inputs_flat, [batch_size, self.max_seq_len]),
            device,
        );
        let targets = Tensor::<B, 2, Int>::from_data(
            TensorData::new(targets_flat, [batch_size, self.max_seq_len]),
            device,
        );

        OpenWebTextBatch { inputs, targets }
    }
}
