import optuna
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
from crina_tinyshakespeare import TinyShakespeareDataset, CrinaSynapse, CosineWarmupScheduler, reset_all_lif_neurons

def objective(trial):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists('tiny_shakespeare.txt'):
        print('Downloading TinyShakespeare dataset...')
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        response = requests.get(url)
        with open('tiny_shakespeare.txt', 'w') as f:
            f.write(response.text)

    # Load TinyShakespeare
    with open('tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    train_ds = TinyShakespeareDataset(train_data, block_size=128)
    val_ds = TinyShakespeareDataset(val_data, block_size=128)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    num_epochs = 1
    max_train_iters = num_epochs * len(train_loader)
    warmup_iters = 100
    d_model = trial.suggest_int("d_model", 128, 384)
    n_layers = trial.suggest_int("n_layers", 4, 8)
    tree_depth = trial.suggest_int("tree_depth", 2, 6)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    model = CrinaSynapse(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, tree_depth=tree_depth).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineWarmupScheduler(optimizer, warmup_start_lr=1e-4, warmup_epochs=warmup_iters, max_epochs=max_train_iters)

    torch.set_float32_matmul_precision('high')
    model = torch.compile(model, mode="default")
    print(f"Model parameters: {sum([p.numel() for p in model.parameters()]):_}")
    #wandb.init(project="crina_tinyshakespeare", name="crina_tinyshakespeare", config={
    #    "model": "CrinaSynapse",
    #    "vocab_size": vocab_size,
    #    "d_model": 256,
    #    "n_layers": 8,
    #    "tree_depth": 4,
    #    "learning_rate": 3e-4,
    #    "batch_size": 64,
    #    "num_epochs": 1,
    #    "warmup_iters": warmup_iters,
    #    "max_train_iters": max_train_iters
    #})
    current_iter = 0
    for epoch in range(num_epochs):
        reset_all_lif_neurons(model)
        model.train()
        #print(list(enumerate(train_loader))[0])
        pbar = tqdm(train_loader, desc="Training...")
        for x, y in pbar:
            x, y = x.cuda(), y.cuda()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})
            scheduler.step(current_iter)
            current_iter += 1
            #wandb.log({"train_loss": loss.item()})
        print(f"Epoch {epoch} | Train loss: {loss.item():.4f}")

        # Validation
        #model.eval()
        #with torch.no_grad():
        #    val_loss = 0
        #    pbar2 = tqdm(val_loader, desc="Validating...")
        #    for x, y in pbar2:
        #        x, y = x.cuda(), y.cuda()
        #        logits = model(x)
        #        loss_item = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item()
        #        val_loss += loss_item
        #        pbar2.set_postfix({'val_loss': loss_item})
        #    val_loss /= len(val_loader)
        #print(f"Epoch {epoch} | Val loss: {val_loss:.4f}")
        #wandb.log({"val_loss": val_loss})
    
    # Save the original model's state dict (uncompiled) to avoid _orig_mod prefix
    #raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    #torch.save(raw_model.state_dict(), "crina_tinyshakespeare.pth")
    #wandb.save("crina_tinyshakespeare.pth")
    #wandb.finish()
    return loss.item()

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    print(f"Best hyperparameters: {study.best_params}")
    optuna.visualization.plot_param_importances(study)