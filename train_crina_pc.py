import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from test_attention import TestCrina, TestLayer
import matplotlib.pyplot as plt
import time
import os
import numpy as np

from datasets import load_dataset

from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv
import atexit
load_dotenv()

block_size = 1024
batch_size = 4
#initial_lr_state = 0.02
#final_lr_state = 0.01

# Predictive Coding Wrapper for TestCrina
class PCCrina(nn.Module):
    def __init__(self, model, lr_state=0.01, inference_steps=10):
        super().__init__()
        self.model = model
        self.lr_state = lr_state
        self.inference_steps = inference_steps
        
        self.d_model = model.embed.embedding_dim
        self.num_layers = len(model.layers)
        
        # Predictive layers: Level l+1 predicts Level l
        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.LayerNorm(self.d_model)
            ) for _ in range(self.num_layers)
        ])

        # NEW: Layer-wise Inference Learning Rates
        # Higher layers often need more/less aggressive updates to match sensory dynamics.
        self.lr_state_layers = nn.Parameter(torch.ones(self.num_layers + 1) * lr_state, requires_grad=False)
        
        # NEW: Error Precision (Trainable)
        # Scales the contribution of each layer's error to the total energy.
        self.precisions = nn.Parameter(torch.ones(self.num_layers) * 1.0)
        
        # optimization: state persistence across chunks
        self.states_cache = None
        self.batch_size = 0
        
    def reset_states(self, B, device):
        self.states_cache = [torch.zeros(B, block_size, self.d_model, device=device) for _ in range(self.num_layers + 1)]
        self.batch_size = B

    def pc_train_step(self, x_tokens, target_tokens, is_new_doc=False):
        """
        One step of Predictive Coding training.
        """
        B, T = x_tokens.shape
        device = x_tokens.device
        
        if self.states_cache is None or is_new_doc or self.batch_size != B:
            self.reset_states(B, device)
            with torch.no_grad():
                h = [self.model.embed(x_tokens)]
                for layer in self.model.layers:
                    h.append(layer(h[-1]))
                self.states_cache = h
            
        states = [p.clone().detach().requires_grad_(True) for p in self.states_cache]
        
        # 2. INFERENCE (E-Step): Minimize Energy
        # Using Adam for better stability. We pass per-layer LRs by creating different param groups.
        param_groups = []
        for l in range(self.num_layers + 1):
            param_groups.append({'params': [states[l]], 'lr': self.lr_state_layers[l].item()})
        
        optimizer_states = torch.optim.Adam(param_groups)
        
        for _ in range(self.inference_steps):
            optimizer_states.zero_grad()
            energy = 0
            
            # Prediction Errors with Precision Weighting
            for l in range(self.num_layers):
                pred = self.predictors[l](states[l+1])
                error = states[l] - pred
                # Scale by precision weight (softplus ensures positivity)
                p = F.softplus(self.precisions[l])
                energy += p * torch.mean(error**2) 
            
            # Top-level error
            logits = self.model.lm_head(self.model.ln_f(states[-1]))
            target_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_tokens.view(-1))
            
            total_energy = energy + target_loss
            
            total_energy.backward()
            optimizer_states.step()
            
        # Update cache with settled states
        with torch.no_grad():
            self.states_cache = [p.detach().clone() for p in states]

        # 3. LEARNING (M-Step): Update Weights
        # Settled errors for top-down predictors
        self.zero_grad()
        weight_update_energy = 0
        for l in range(self.num_layers):
            pred = self.predictors[l](states[l+1].detach())
            error = states[l].detach() - pred
            weight_update_energy += torch.mean(error**2)
            
        # Settled errors for bottom-up model layers (Recognition pass)
        for l in range(self.num_layers):
            target_h = states[l+1].detach()
            input_h = states[l].detach()
            pred_up = self.model.layers[l](input_h)
            error_up = target_h - pred_up
            weight_update_energy += torch.mean(error_up**2)

        weight_update_energy.backward()
        
        return weight_update_energy.item()

    def pc_val_step(self, x_tokens, target_tokens, is_new_doc=False):
        """
        Validation step: settling latent states without weight updates.
        Returns the settled Energy and Cross-Entropy loss.
        """
        B, T = x_tokens.shape
        device = x_tokens.device
        
        # Validation uses its own separate persistence logic (or just resets)
        # For simplicity in validation, we can just reset states or use a transient cache.
        v_h = [self.model.embed(x_tokens)]
        for layer in self.model.layers:
            v_h.append(layer(v_h[-1]))
            
        states = [p.clone().detach().requires_grad_(True) for p in v_h]
        
        param_groups = []
        for l in range(self.num_layers + 1):
            param_groups.append({'params': [states[l]], 'lr': self.lr_state_layers[l].item()})
        optimizer_states = torch.optim.Adam(param_groups)
        
        with torch.enable_grad():
            for _ in range(self.inference_steps):
                optimizer_states.zero_grad()
                energy = 0
                for l in range(self.num_layers):
                    pred = self.predictors[l](states[l+1])
                    error = states[l] - pred
                    p = F.softplus(self.precisions[l])
                    energy += p * torch.mean(error**2)
                
                logits = self.model.lm_head(self.model.ln_f(states[-1]))
                target_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_tokens.view(-1))
                
                total_energy = energy + target_loss
                total_energy.backward()
                optimizer_states.step()
            
        return total_energy.item(), target_loss.item()

    def initialize_from_model(self):
        """
        Initialize top-down predictors with Xavier initialization.
        Feedback Alignment (weight transpose) is skipped as the current 
        architecture uses a complex non-linear attention block.
        """
        with torch.no_grad():
            for l in range(self.num_layers):
                # Predictor is Sequential[Linear, LN]. Sequential[0] is Linear.
                nn.init.xavier_uniform_(self.predictors[l][0].weight)
                self.predictors[l][0].bias.data.zero_()

    def set_layer_lrs(self, sensory_lr=0.02, abstract_lr=0.005):
        """
        Decay inference learning rates as we go higher in the hierarchy.
        Sensory levels (0, 1) usually need faster updates.
        """
        lrs = torch.linspace(sensory_lr, abstract_lr, self.num_layers + 1)
        self.lr_state_layers.data.copy_(lrs)

def get_next_pair(idx):
    """
    Returns (x_chunk, y_chunk, needs_reset) for a specific batch index.
    If the current document stream is empty, fetches a new document.
    """
    global shared_it
    try:
        return next(doc_streams[idx]), False
    except StopIteration:
        while True:
            try:
                doc = next(shared_it)['text'].encode('utf-8')
                if len(doc) > block_size + 1:
                    # Create sequence of chunks for this document
                    chunks = []
                    for i in range(0, len(doc) - block_size, block_size):
                        cx = torch.from_numpy(np.frombuffer(doc[i:i+block_size], dtype=np.uint8).copy()).long()
                        cy = torch.from_numpy(np.frombuffer(doc[i+1:i+block_size+1], dtype=np.uint8).copy()).long()
                        chunks.append((cx, cy))
                    doc_streams[idx] = iter(chunks)
                    return next(doc_streams[idx]), True
            except StopIteration:
                # Dataset exhausted, re-initialize if needed or exit
                print("Dataset exhausted.")
                return None, False

current_batch = 0
current_train_iter = 0
current_val_iter = 0
total_loss = 0.0
val_total_loss = 0.0

# --- TRAINING SCRIPT ---
if __name__ == "__main__":
    ds = load_dataset("vietgpt/openwebtext_en", split="train", streaming=True)
    val_ds = ds.take(550) # Use a small portion for periodic validation
    train_ds = ds.skip(550)
    
    # Stateful Batch Loading
    shared_it = iter(train_ds)
    val_it = iter(val_ds)
    # Each batch index has its own current document iterator
    doc_streams = [iter([]) for _ in range(batch_size)]
    val_doc_streams = [iter([]) for _ in range(batch_size)]

    def get_next_pair(idx, iterator, streams):
        """
        Returns (x_chunk, y_chunk, needs_reset) for a specific batch index.
        If the current document stream is empty, fetches a new document.
        """
        global shared_it
        try:
            return next(streams[idx]), False
        except StopIteration:
            while True:
                try:
                    doc = next(iterator)['text'].encode('utf-8')
                    if len(doc) > block_size + 1:
                        # Create sequence of chunks for this document
                        chunks = []
                        for i in range(0, len(doc) - block_size, block_size):
                            cx = torch.from_numpy(np.frombuffer(doc[i:i+block_size], dtype=np.uint8).copy()).long()
                            cy = torch.from_numpy(np.frombuffer(doc[i+1:i+block_size+1], dtype=np.uint8).copy()).long()
                            chunks.append((cx, cy))
                        streams[idx] = iter(chunks)
                        return next(streams[idx]), True
                except StopIteration:
                    # Dataset exhausted, re-initialize if needed or exit
                    print("Dataset exhausted.")
                    return None, False
    
    writer = SummaryWriter("runs/pc_crina_training/experiment_deep_1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Disable multiprocessing for downloads to avoid socket errors on Windows
    os.environ["HF_HUB_DISABLE_MP_DOWNLOAD"] = "1"

    # Initialize basic TestCrina
    # Scaling to 12 layers for "Deep" demonstration
    base_model = TestCrina(vocab_size=256, d_model=256, num_layers=12, tree_depth=4).to(device)
    pc_model = PCCrina(base_model, lr_state=0.01, inference_steps=15).to(device)
    
    # NEW: Apply deep-scaling optimizations
    pc_model.initialize_from_model()
    pc_model.set_layer_lrs(sensory_lr=0.02, abstract_lr=0.005)

    def save_model():
        if current_train_iter > 0:
            torch.save(base_model.state_dict(), "crina_pc_openwebtext.pth")
            print(f"\nModel saved successfully! (Iter {current_train_iter})")
        writer.close()
    
    atexit.register(save_model)

    optimizer_weights = torch.optim.AdamW(pc_model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer_weights, mode='min', factor=0.75, patience=3)
    
    # Synthetic Task: Predict copy of input (Simple for demo)
    print("Starting Predictive Coding Training on TestCrina...")
    losses = []
    while current_train_iter < 10000:
        batch_x = []
        batch_y = []
        batch_resets = []
        
        # 1. PREPARE BATCH
        for i in range(batch_size):
            pair, needs_reset = get_next_pair(i, shared_it, doc_streams)
            if pair is None: 
                save_model()
                exit()
            
            batch_x.append(pair[0])
            batch_y.append(pair[1])
            batch_resets.append(needs_reset)
            
        x_batch = torch.stack(batch_x).to(device)
        y_batch = torch.stack(batch_y).to(device)
        is_new_doc_batch = any(batch_resets) # If any sequence in batch is new

        # PC Step
        loss = pc_model.pc_train_step(x_batch, y_batch, is_new_doc=is_new_doc_batch)
        
        optimizer_weights.step()
        optimizer_weights.zero_grad()

        losses.append(loss)
        if current_train_iter % 10 == 0:
            print(f"Step {current_train_iter} | PC Energy: {loss:.4f}")
            writer.add_scalar("Train/PC Energy", loss, current_train_iter)
            
        # 2. VALIDATION
        if current_train_iter % 100 == 0:
            pc_model.eval()
            val_energy_acc = 0.0
            val_loss_acc = 0.0
            val_steps = 5
            
            with torch.no_grad():
                for v_step in range(val_steps):
                    v_batch_x = []
                    v_batch_y = []
                    for i in range(batch_size):
                        pair, _ = get_next_pair(i, val_it, val_doc_streams)
                        if pair is None: 
                            # If validation set is exhausted, reset it and retry
                            val_it = iter(val_ds)
                            pair, _ = get_next_pair(i, val_it, val_doc_streams)
                        
                        if pair is not None:
                            v_batch_x.append(pair[0])
                            v_batch_y.append(pair[1])
                    
                    if len(v_batch_x) == batch_size:
                        vx = torch.stack(v_batch_x).to(device)
                        vy = torch.stack(v_batch_y).to(device)
                        v_energy, v_loss = pc_model.pc_val_step(vx, vy)
                        val_energy_acc += v_energy
                        val_loss_acc += v_loss
            
            avg_val_energy = val_energy_acc / val_steps
            avg_val_loss = val_loss_acc / val_steps
            print(f"--- VALIDATION | Step {current_train_iter} | Energy: {avg_val_energy:.4f} | Loss: {avg_val_loss:.4f} ---")
            writer.add_scalar("Val/PC Energy", avg_val_energy, current_train_iter)
            writer.add_scalar("Val/Cross Entropy", avg_val_loss, current_train_iter)
            
            # Step the scheduler based on validation loss
            scheduler.step(avg_val_loss)
            writer.add_scalar("Train/Learning Rate", optimizer_weights.param_groups[0]['lr'], current_train_iter)
            writer.add_scalar("Train/State Learning Rate", pc_model.lr_state, current_train_iter)
            
            pc_model.train()

        current_train_iter += 1
            
    #plt.figure(figsize=(8, 5))
    #plt.plot(losses)
    #plt.title("Predictive Coding Energy Minimization (TestCrina)")
    #plt.xlabel("Training Steps")
    #plt.ylabel("Local Prediction Error (Energy)")
    #plt.grid(True)
    #plt.savefig("pc_crina_training.png")
    #print("Training complete. Results saved to pc_crina_training.png")
    save_model()
