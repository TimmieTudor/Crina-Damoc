import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau
from test_attention import TestCrina, TestLayer, FeedForwardSNN
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
    def __init__(self, model, lr_state=0.01, inference_steps=10, task_weight=5.0):
        super().__init__()
        self.model = model
        self.lr_state = lr_state
        self.inference_steps = inference_steps
        self.task_weight = task_weight
        self.d_model = model.embed.embedding_dim
        self.tree_depth = model.layers[0].attention.tree_depth
        self.num_layers = len(model.layers)
        
        # Predictive layers: Level l+1 predicts Level l
        # Upgrading to 2-layer MLPs for higher generative capacity.
        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model * 2),
                nn.GELU(),
                nn.Linear(self.d_model * 2, self.d_model),
                nn.LayerNorm(self.d_model)
            ) for _ in range(self.num_layers)
        ])

        # NEW: Layer-wise Inference Learning Rates
        # Higher layers often need more/less aggressive updates to match sensory dynamics.
        self.lr_state_layers = nn.Parameter(torch.ones(self.num_layers + 1) * lr_state, requires_grad=False)
        
        # NEW: Error Precision (Trainable)
        # Scales the contribution of each layer's error to the total energy.
        self.precisions = nn.Parameter(torch.ones(self.num_layers) * 1.0)
        
        # NEW: Cache LRs as Python list to avoid .item() syncs in hot loop
        self.lr_list = [lr_state] * (self.num_layers + 1)
        
        # optimization: state persistence across chunks
        self.states_cache = None
        self.batch_size = 0
        
    def reset_states(self, B, device):
        self.states_cache = [torch.zeros(B, block_size, self.d_model, device=device) for _ in range(self.num_layers + 1)]
        self.batch_size = B

    def pc_train_step(self, x_tokens, target_tokens, is_new_doc=False, use_lookahead=True, return_metrics=False):
        """
        One step of Predictive Coding training using Variational Dual-Inference.
        """
        B, T = x_tokens.shape
        device = x_tokens.device
        
        # 1. INITIALIZATION
        if self.states_cache is None or is_new_doc or self.batch_size != B or use_lookahead:
            with torch.no_grad():
                h = [self.model.embed(x_tokens)]
                for layer in self.model.layers:
                    h.append(layer(h[-1]))
                self.states_cache = h
            self.batch_size = B
            
        states = [p.clone().detach().requires_grad_(True) for p in self.states_cache]
        
        # 2. INFERENCE (E-Step): Minimize Variational Energy
        p_weights = [F.softplus(self.precisions[l]) for l in range(self.num_layers)]
        # Variational term: -ln(precision)
        prec_logs = [torch.log(p + 1e-6) for l, p in enumerate(p_weights)]
        
        for _ in range(self.inference_steps):
            energy = 0
            # A. Prediction Errors (Top-down)
            for l in range(self.num_layers):
                pred = self.predictors[l](states[l+1])
                error = states[l] - pred
                energy += p_weights[l] * torch.mean(error**2) 
            
            # B. NEW: Recognition Errors (Bottom-up term in E-step)
            # This forces states to stay close to what the model's layers produce
            for l in range(self.num_layers):
                pred_up = self.model.layers[l](states[l])
                error_up = states[l+1] - pred_up
                # Sharing precision for BU/TD for simplicity, or could have separate
                energy += p_weights[l] * torch.mean(error_up**2)

            # C. Top-level task error (Weighted)
            logits = self.model.lm_head(self.model.ln_f(states[-1]))
            target_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_tokens.view(-1))
            
            # Total Variational Energy
            total_energy = energy + self.task_weight * target_loss
            
            state_grads = torch.autograd.grad(total_energy, states)
            with torch.no_grad():
                for l in range(self.num_layers + 1):
                    states[l] -= self.lr_list[l] * state_grads[l]
            
        with torch.no_grad():
            self.states_cache = [p.detach().clone() for p in states]

        # 3. LEARNING (M-Step): Update Weights and Precisions
        layer_metrics = {}
        self.zero_grad()
        weight_update_energy = 0
        total_mse_energy = 0 # Purely positive MSE for intuitive logging
        
        # A. Prediction Learning (Top-down)
        for l in range(self.num_layers):
            pred = self.predictors[l](states[l+1].detach())
            error = states[l].detach() - pred
            e_val = torch.mean(error**2)
            # Variational Objective for Precision: pi * e^2 - ln(pi)
            # This term can be negative if error is very small and precision is high.
            weight_update_energy += p_weights[l] * e_val - prec_logs[l]
            total_mse_energy += e_val.item()
            if return_metrics:
                layer_metrics[f"Layer_{l}/Inference_Error"] = e_val.item()
            
        # B. Model Learning (Recognition pass / Bottom-up)
        total_rec_err = 0
        for l in range(self.num_layers):
            target_h = states[l+1].detach()
            input_h = states[l].detach()
            pred_up = self.model.layers[l](input_h)
            error_up = target_h - pred_up
            e_up_val = torch.mean(error_up**2)
            weight_update_energy += e_up_val 
            total_rec_err += e_up_val.item()
            if return_metrics:
                layer_metrics[f"Layer_{l}/Recognition_Error"] = e_up_val.item()

        # C. NEW: Task Learning (LM Head & classification weights)
        # We must add this to the M-step so the weights actually learn the task!
        logits = self.model.lm_head(self.model.ln_f(states[-1].detach()))
        target_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_tokens.view(-1))
        weight_update_energy += target_loss
        
        if return_metrics:
            layer_metrics["Total/Recognition_Energy"] = total_rec_err
            layer_metrics["Total/MSE_Energy"] = total_mse_energy
            layer_metrics["Total/Variational_Energy"] = weight_update_energy.item()

        weight_update_energy.backward()
        
        with torch.no_grad():
            self.precisions.clamp_(min=-3.0, max=5.0)
            if return_metrics:
                for l in range(self.num_layers):
                    layer_metrics[f"Layer_{l}/Precision"] = p_weights[l].item()
        
        # Return the purely positive MSE energy for simpler console output
        return total_mse_energy, layer_metrics

    def bp_train_step(self, x_tokens, target_tokens):
        """
        Standard Backpropagation step for performance comparison.
        """
        self.zero_grad()
        logits = self.model(x_tokens)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_tokens.view(-1))
        loss.backward()
        grad_metrics = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_metrics[f"GradNorm/{name}"] = param.grad.norm().item()
        return loss.item(), grad_metrics

    def pc_val_step(self, x_tokens, target_tokens, is_new_doc=False):
        """
        Validation step using Variational Dual-Inference.
        """
        B, T = x_tokens.shape
        device = x_tokens.device
        
        v_h = [self.model.embed(x_tokens)]
        for layer in self.model.layers:
            v_h.append(layer(v_h[-1]))
            
        states = [p.clone().detach().requires_grad_(True) for p in v_h]
        p_weights = [F.softplus(self.precisions[l]) for l in range(self.num_layers)]
        
        with torch.enable_grad():
            for _ in range(self.inference_steps):
                energy = 0
                for l in range(self.num_layers):
                    pred = self.predictors[l](states[l+1])
                    error = states[l] - pred
                    energy += p_weights[l] * torch.mean(error**2)

                for l in range(self.num_layers):
                    pred_up = self.model.layers[l](states[l])
                    error_up = states[l+1] - pred_up
                    energy += p_weights[l] * torch.mean(error_up**2)
                
                logits = self.model.lm_head(self.model.ln_f(states[-1]))
                target_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_tokens.view(-1))
                
                total_energy = energy + self.task_weight * target_loss
                state_grads = torch.autograd.grad(total_energy, states)
                
                with torch.no_grad():
                    for l in range(self.num_layers + 1):
                        states[l] -= self.lr_list[l] * state_grads[l]
            
        # Return purely positive MSE + Loss part for validation tracking
        with torch.no_grad():
            val_mse = 0
            for l in range(self.num_layers):
                pred = self.predictors[l](states[l+1])
                val_mse += torch.mean((states[l] - pred)**2).item()
                
        return val_mse, target_loss.item()
    
    def bp_val_step(self, x_tokens, target_tokens):
        """
        Validation step using standard Backpropagation.
        """
        self.zero_grad()
        logits = self.model(x_tokens)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_tokens.view(-1))
        return loss.item()
    
    def initialize_from_model(self):
        """
        Initialize top-down predictors with the same initialization as the layers themselves
        """
        with torch.no_grad():
            for l in range(self.num_layers):
                # Predictor is Sequential[Linear, GELU, Linear, LayerNorm].
                # We initialize the two Linear layers.
                nn.init.xavier_uniform_(self.predictors[l][0].weight)
                self.predictors[l][0].bias.data.zero_()
                nn.init.xavier_uniform_(self.predictors[l][2].weight)
                self.predictors[l][2].bias.data.zero_()

    def set_layer_lrs(self, sensory_lr=0.02, abstract_lr=0.005):
        """
        Decay inference learning rates as we go higher in the hierarchy.
        Sensory levels (0, 1) usually need faster updates.
        """
        lrs = torch.linspace(sensory_lr, abstract_lr, self.num_layers + 1)
        self.lr_state_layers.data.copy_(lrs)
        self.lr_list = lrs.tolist() # Store as python list for sync-free access

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
    
    # Mode selection: PC or BACKPROP
    mode = "pc"
    if len(sys.argv) > 1 and sys.argv[1] in ["pc", "bp"]:
        mode = sys.argv[1]
    print(f"Running in {mode.upper()} mode.")

    experiment_idx = 1
    experiment_name = f"experiment_bridge_{mode}"
    if os.path.exists("runs/pc_crina_training"):
        for folder in os.listdir("runs/pc_crina_training"):
            if folder.startswith(experiment_name):
                experiment_idx += 1
    experiment_name = f"{experiment_name}_{experiment_idx}"

    writer = SummaryWriter(f"runs/pc_crina_training/{experiment_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Enable TensorFloat32 for performance on NVIDIA GPUs
    torch.set_float32_matmul_precision('high')
    #os.environ["HF_HUB_DISABLE_MP_DOWNLOAD"] = "1"

    # Initialize basic TestCrina
    # Scaling to 12 layers for "Deep" demonstration
    base_model = TestCrina(vocab_size=256, d_model=256, num_layers=12, tree_depth=4).to(device)
    # Variational PC with Task Weighting
    pc_model = PCCrina(base_model, lr_state=0.01, inference_steps=15, task_weight=5.0).to(device)
    # NEW: Apply deep-scaling optimizations
    pc_model.initialize_from_model()
    pc_model.set_layer_lrs(sensory_lr=0.02, abstract_lr=0.005)

    # Compile the specific training methods AFTER initialization
    #if hasattr(torch, "compile") and mode == "pc":
    #    print("Compiling PC training and validation steps...")
    #    # We compile the methods directly because torch.compile(model) only targets model.forward()
    #    pc_model.model = torch.compile(pc_model.model)
    #    pc_model.pc_train_step = torch.compile(pc_model.pc_train_step)
    #    pc_model.pc_val_step = torch.compile(pc_model.pc_val_step)

    def save_model():
        if current_train_iter > 0:
            torch.save(pc_model.model.state_dict(), "crina_pc_openwebtext.pth")
            print(f"\nModel saved successfully! (Iter {current_train_iter})")
        writer.close()
    
    atexit.register(save_model)

    optimizer_weights = torch.optim.AdamW(pc_model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer_weights, mode='min', factor=0.75, patience=5)
    
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

        # Training Step
        if mode == "pc":
            # Performance Optimization: Only compute per-layer diagnostics every 10 steps to avoid GPU-CPU sync overhead
            want_diagnostics = (current_train_iter % 10 == 0)
            loss, metrics = pc_model.pc_train_step(x_batch, y_batch, is_new_doc=is_new_doc_batch, return_metrics=want_diagnostics)
        else:
            loss, grad_metrics = pc_model.bp_train_step(x_batch, y_batch)
        
        # Check for NaN/Inf Divergence
        if not np.isfinite(loss):
            print(f"\n[PANIC] Divergence detected at step {current_train_iter} (Loss: {loss})")
            save_model()
            exit(1)

        # NEW: Gradient Clipping & Monitoring
        grad_norm = torch.nn.utils.clip_grad_norm_(pc_model.parameters(), max_norm=1.0)
        
        optimizer_weights.step()
        optimizer_weights.zero_grad()

        losses.append(loss)
        if current_train_iter % 10 == 0:
            print(f"Step {current_train_iter} | PC Energy: {loss:.4f} | Grad Norm: {grad_norm:.4f}")
            writer.add_scalar("Train/PC Energy", loss, current_train_iter)
            writer.add_scalar("Train/Grad Norm", grad_norm, current_train_iter)
            
            # Log per-layer diagnostics
            if mode == "pc":
                for k, v in metrics.items():
                    writer.add_scalar(f"Diagnostics/{k}", v, current_train_iter)
            else:
                for k, v in grad_metrics.items():
                    writer.add_scalar(f"Diagnostics/{k}", v, current_train_iter)
            
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
                        if mode == "pc":
                            v_energy, v_loss = pc_model.pc_val_step(vx, vy)
                        else:
                            v_loss = pc_model.bp_val_step(vx, vy)
                            v_energy = 0
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
