import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import json
import math
from pathlib import Path
from tqdm import tqdm
import wandb

# Import the model components from your existing script
from crina_tinyshakespeare import CrinaSynapse, reset_all_lif_neurons

class ARCDataset(Dataset):
    def __init__(self, data_dir, num_demos=3, max_seq_len=1024):
        self.tasks = []
        self.num_demos = num_demos
        self.max_seq_len = max_seq_len
        self.SEP = 10
        
        path = Path(data_dir)
        for file in path.glob("*.json"):
            with open(file, 'r') as f:
                self.tasks.append(json.load(f))

    def grid_to_seq(self, grid):
        return [pixel for row in grid for pixel in row]

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        train_pairs = task['train']
        test_pair = task['test'][0]
        
        # Construct sequence: [demo1_in, SEP, demo1_out, SEP, ..., test_in, SEP, test_out]
        full_seq = []
        
        # Add demos
        for i in range(min(len(train_pairs), self.num_demos)):
            full_seq += self.grid_to_seq(train_pairs[i]['input'])
            full_seq.append(self.SEP)
            full_seq += self.grid_to_seq(train_pairs[i]['output'])
            full_seq.append(self.SEP)
            
        # Add test input
        full_seq += self.grid_to_seq(test_pair['input'])
        full_seq.append(self.SEP)
        
        # Mark where the "answer" starts for loss masking
        input_len = len(full_seq)
        
        # Add test output (target)
        full_seq += self.grid_to_seq(test_pair['output'])
        
        # Truncate if too long (keeping the end)
        if len(full_seq) > self.max_seq_len:
            full_seq = full_seq[-self.max_seq_len:]
            input_len = max(0, input_len - (len(full_seq) - self.max_seq_len))

        # Pad to max_seq_len
        actual_len = len(full_seq)
        padding = [0] * (self.max_seq_len - actual_len)
        full_seq = full_seq + padding
        
        x = torch.tensor(full_seq[:-1], dtype=torch.long)
        y = torch.tensor(full_seq[1:], dtype=torch.long)
        
        # Mask: we only care about the loss on the test_output part
        mask = torch.zeros(self.max_seq_len - 1)
        # The test output tokens in 'y' start at index input_len-1 and end at actual_len-1
        # But wait, y[i] is the target for x[i]. 
        # So for x[input_len-1] (the SEP after test_in), the target is y[input_len-1] (the first pixel of test_out).
        mask[input_len-1 : actual_len-1] = 1.0
        
        return x, y, mask

def transfer_weights(src_path, dest_model):
    """Transfer weights from a Shakespeare model to an ARC model."""
    print(f"Transferring weights from {src_path}...")
    src_state = torch.load(src_path, map_location='cpu')
    dest_state = dest_model.state_dict()
    
    transferred = 0
    for name, param in src_state.items():
        if name in dest_state and param.shape == dest_state[name].shape:
            dest_state[name].copy_(param)
            transferred += 1
            
    dest_model.load_state_dict(dest_state)
    print(f"Successfully transferred {transferred} parameter groups.")
    return dest_model

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Hyperparams
    batch_size = 4
    lr = 1e-4
    epochs = 50
    d_model = 256
    n_layers = 8
    tree_depth = 4
    max_seq_len = 1025
    
    # 2. Dataset
    train_dir = "./ARC/data/training"
    if not os.path.exists(train_dir):
        print(f"Error: {train_dir} not found. Please ensure ARC data is present.")
        return
        
    ds = ARCDataset(train_dir, num_demos=2, max_seq_len=max_seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    
    # 3. Model
    model = CrinaSynapse(vocab_size=11, d_model=d_model, n_layers=n_layers, tree_depth=tree_depth).to(device)
    
    # Optional: Load from pre-trained Shakespeare
    #if os.path.exists("crina_tinyshakespeare.pth"):
        #transfer_weights("crina_tinyshakespeare.pth", model)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    wandb.init(project="crina_arc_finetune", name="arc_experiment_1")
    
    for epoch in range(epochs):
        model.train()
        reset_all_lif_neurons(model)
        epoch_loss = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for x, y, mask in pbar:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            
            logits = model(x) # [B, T, V]
            
            # Weighted loss
            loss = F.cross_entropy(logits.permute(0, 2, 1), y, reduction='none')
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
        wandb.log({"loss": avg_loss})
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"crina_arc_epoch_{epoch}.pth")

    torch.save(model.state_dict(), "crina_arc_final.pth")
    wandb.finish()

if __name__ == "__main__":
    train()
