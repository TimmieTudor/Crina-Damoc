from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import math
import time
import os
import requests
from tqdm import tqdm
#from crina_tinyshakespeare import CrinaSynapse
from test_attention import TestCrina

#import wandb
from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv
import atexit
load_dotenv()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
print(f"Using device: {device}")

# Disable multiprocessing for downloads to avoid socket errors on Windows
os.environ["HF_HUB_DISABLE_MP_DOWNLOAD"] = "1"

ds = load_dataset("vietgpt/openwebtext_en", split="train", streaming=True)

block_size = 1024
batch_size = 8

#model = CrinaSynapse(vocab_size=256, d_model=256, n_layers=8, tree_depth=4).to(device)
model = TestCrina(vocab_size=256, d_model=256, num_layers=8, tree_depth=4).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
#scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=500)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=300_000, eta_min=5e-6)

#dummy_input = torch.randint(0, 256, (batch_size, block_size)).to(device)

# Analyze graph breaks before compilation
#print("Analyzing graph breaks...")
#import torch._dynamo
#explanation = torch._dynamo.explain(model, dummy_input)
#print(explanation)
#model.reset_state() # Reset state after analysis run

#exit()

"""try:
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    torch._dynamo.config.capture_scalar_outputs = True
    #model = torch.compile(model, fullgraph=True)
    model = torch.compile(model)
    print("Model compiled successfully!")
except Exception as e:
    print("Failed to compile model. Skipping...")
    print(e)"""


print(f"Model parameters: {sum([p.numel() for p in model.parameters()]):_}")

accumulation_steps = 1
optimizer.zero_grad()

#wandb.init(project="crina-damoc")
# SummaryWriter will be initialized below


# Global state will be initialized in the main loop block below


def save_model():
    if current_train_iter > 0:
        torch.save(model.state_dict(), "crina_openwebtext.pth")
        print(f"\nModel saved successfully! (Iter {current_train_iter})")
    writer.close()

atexit.register(save_model)

# Stateful Batch Loading
shared_it = iter(ds)
# Each batch index has its own current document iterator
doc_streams = [iter([]) for _ in range(batch_size)]

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


accumulation_steps = 1
optimizer.zero_grad()
writer = SummaryWriter('runs/crina_stateful_batching_2')

current_batch = 0
current_train_iter = 0
current_val_iter = 0
total_loss = 0.0
val_total_loss = 0.0

model.train()
#model.detach_state() # Ensure clean state before loop starts
try:
    while True:
        batch_x = []
        batch_y = []
        
        # 1. PREPARE BATCH
        for i in range(batch_size):
            pair, needs_reset = get_next_pair(i)
            if pair is None: 
                save_model()
                exit()
            
            #if needs_reset:
            #    model.reset_state(i)
            
            batch_x.append(pair[0])
            batch_y.append(pair[1])
            
        x_batch = torch.stack(batch_x).to(device)
        y_batch = torch.stack(batch_y).to(device)

        # 2. VALIDATION CHECK (Isolation via State Saving)
        if current_train_iter % 50 == 0:
            model.eval()
            #saved_state = model.collect_state()
            with torch.no_grad():
                logits = model(x_batch)
                v_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_batch.view(-1))
                val_total_loss += v_loss.item()
                current_val_iter += 1
                avg_val = val_total_loss / current_val_iter
                bpb = avg_val / math.log(2)
                
                print(f"\n[VAL] Iter {current_train_iter} | Loss: {v_loss.item():.4f} | Avg: {avg_val:.4f} | BPB: {bpb:.4f}")
                writer.add_scalar("val_loss", v_loss.item(), current_train_iter)
                writer.add_scalar("val_bpb", bpb, current_train_iter)
                
                if current_train_iter > 0:
                    avg_train = total_loss / current_train_iter
                    writer.add_scalar("train_val_gap", avg_val - avg_train, current_train_iter)
            
            #model.load_state(saved_state)
            model.train()

        # 3. TRAIN STEP
        logits = model(x_batch)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_batch.view(-1))
        
        (loss / accumulation_steps).backward()
        #model.detach_state() # Truncate BPTT


        current_batch += 1
        if current_batch % accumulation_steps == 0:
            # Clip gradients and check for NaNs
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            if torch.isnan(grad_norm):
                print(f"\n[WARNING] NaN gradient detected (norm={grad_norm}). Skipping step.")
                optimizer.zero_grad()
                continue

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            current_train_iter += 1
            total_loss += loss.item()
            
            if current_train_iter % 5 == 0:
                avg_train = total_loss / current_train_iter
                print(f"Iter {current_train_iter} | Loss: {loss.item():.4f} | Avg: {avg_train:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
                writer.add_scalar("train_loss", loss.item(), current_train_iter)
                writer.add_scalar("lr", optimizer.param_groups[0]['lr'], current_train_iter)
                writer.flush()

except KeyboardInterrupt:
    print("\nTraining interrupted.")
    save_model()
