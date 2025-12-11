import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import prune
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
import os
import numpy as np
from transformers import BertTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

# Import your Crina model (adjust path to crina_model.py)
from crina_model import CrinaSynapse, LIFNeuron, SparseLinear  # Assuming crina_model.py is in the same dir

# Hyperparameters
d_model = 256
n_heads = 16
n_layers = 4
dropout = 0.1
sparsity_level = 0.3
time_steps = 10
batch_size = 32
epochs = 5
profile_epochs = 0
lr = 1.25e-04
dataset_dir = "multimodal_dataset_refined"  # Your dataset dir
output_dir = "crina_pretrain"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Input sizes
text_seq_len = 1024
image_channels, image_size = 3, 32
audio_len = 128
video_frames, video_channels, video_size = 10, 3, 32
#vocab_size = 30522
vocab_size = 255

def ord_or_zero(c):
    if c == '':
        return 0
    else:
        return ord(c)

# Custom Dataset class for .pt batches
class MultimodalDataset(Dataset):
    def __init__(self, dataset_dir, tokenizer_name='bert-base-uncased'):
        # Flatten all samples across .pt files into a single list
        self.samples = []
        batches = [os.path.join(dataset_dir, f) for f in sorted(os.listdir(dataset_dir)) if f.endswith('.pt')]
        for batch_file in batches:
            batch = torch.load(batch_file)
            self.samples.extend(batch)  # Add each sample to the flat list
        
        # Load tokenizer
        #self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = text_seq_len  # Match text_seq_len
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Handle tuple idx from Subset
        if isinstance(idx, tuple):
            idx = idx[0]
        sample = self.samples[idx]
        # Text tokenization
        tokens = sample[0]
        #print(tokens)
        #text_str = ' '.join(tokens)  # Join tokens to sentence
        #tokenized = self.tokenizer(text_str, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        #text_tensor = tokenized.input_ids.squeeze(0)  # [max_length]

        text_tensor = torch.tensor([ord_or_zero(c) for c in tokens])
        
        # Image
        image_tensor = sample[1]
        
        # Audio
        audio_tensor = sample[2]
        
        # Video
        video_tensor = sample[3]
        
        return text_tensor, image_tensor, audio_tensor, video_tensor
    

# Custom Dataset for flat samples
class FlatMultimodalDataset(Dataset):
    def __init__(self, samples, tokenizer_name='bert-base-uncased'):
        self.samples = samples
        # Load tokenizer
        #self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = text_seq_len  # Match text_seq_len
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Text encoding (hash to int)
        tokens = sample[0]
        #print(tokens)
        #text_str = ' '.join(tokens)  # Join tokens to sentence
        #tokenized = self.tokenizer(text_str, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        #text_tensor = tokenized.input_ids.squeeze(0)  # [max_length]
        text_tensor = torch.tensor([ord_or_zero(c) for c in tokens])
        image_tensor = sample[1]
        audio_tensor = sample[2]
        video_tensor = sample[3]
        return text_tensor, image_tensor, audio_tensor, video_tensor

# Training function
def train_crina():
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    dataset = MultimodalDataset(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Model, optimizer, criterion
    model = CrinaSynapse(d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print(f"Model parameters: {sum([p.numel() for p in model.parameters()])}")

    model.train()
    total_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (text_input, image_input, audio_input, video_input) in enumerate(dataloader):
            text_input, image_input, audio_input, video_input = text_input.to(device), image_input.to(device), audio_input.to(device), video_input.to(device)
            
            optimizer.zero_grad()
            doc_out = model(text_input, image_input, audio_input, video_input)
            text_out = doc_out.text
            image_out = doc_out.image
            audio_out = doc_out.audio
            video_out = doc_out.video
            
            # Reconstruction losses (autoencoder style)
            loss_text = criterion(text_out.float(), text_input.float())  # Cast to float for MSE
            loss_image = criterion(image_out, image_input)
            loss_audio = criterion(audio_out, audio_input)
            loss_video = criterion(video_out, video_input)
            loss = loss_text + loss_image + loss_audio + loss_video
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        total_loss += avg_epoch_loss
        print(f"Epoch {epoch+1}/{epochs} completed, Avg Loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(output_dir, f"crina_epoch_{epoch+1}.pt"))
    
    print(f"Training complete! Final Avg Loss: {total_loss / epochs:.4f}")

def pretrain_crina(dataset_dir="multimodal_dataset_refined", output_dir="crina_pretrain", epochs=100, lr=0.001, recursion_depth=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrinaSynapse(d_model=d_model, n_heads=n_heads, n_layers=n_layers, sparsity_level=sparsity_level, dropout=dropout).to(device)
    #model = torch.compile(model, mode='reduce-overhead')  # Or 'max-autotune' for GPU
    optimizer = optim.AdamW(model.parameters(), lr=lr, fused=True)
    criterion_mse = nn.MSELoss()
    criterion_contrast = nn.CosineEmbeddingLoss()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Tokenizer for text
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    criterion_ce = nn.CrossEntropyLoss()  # Ignore pads
    
    os.makedirs(output_dir, exist_ok=True)
    dataset = MultimodalDataset(dataset_dir)  # Your Dataset class
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # Full batch_size=32

    # Load all batches into a flat list of samples
    all_samples = []
    for batch_file in sorted(os.listdir(dataset_dir)):
        if batch_file.endswith('.pt'):
            batch = torch.load(os.path.join(dataset_dir, batch_file))
            all_samples.extend(batch)  # Flat list of (text, image, audio, video) tuples

    # Manual split (80/20)
    split_idx = int(0.8 * len(all_samples))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    train_dataset = FlatMultimodalDataset(train_samples)
    val_dataset = FlatMultimodalDataset(val_samples)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"Model parameters: {sum([p.numel() for p in model.parameters()]):_}")

    epoch_train_losses = []
    epoch_val_losses = []
    
    model.train()
    total_loss = 0.0
    for epoch in range(epochs):
        train_step_losses = []
        val_step_losses = []
        epoch_loss = 0.0

        if epoch < profile_epochs:
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True)

        """current_sparsity = 0.3 + 0.05 * (epoch / epochs)  # Slow ramp
        if epoch >= 10 and epoch % 5 == 0:  # Re-prune every 5 epochs
            model.to('cpu')  # Move to CPU for pruning
            for name, module in model.named_modules():
                if isinstance(module, SparseLinear):
                    prune.l1_unstructured(module.linear, name='weight', amount=current_sparsity)
            model.to(device)  # Move back to GPU
            print(f"Epoch {epoch+1}: Re-pruned to sparsity {current_sparsity:.2f}")"""
        for i, (text_input, image_input, audio_input, video_input) in enumerate(train_dataloader):
            if epoch < profile_epochs:
                with prof, record_function("batch_forward"):
                    text_input = text_input.to(device)
                    image_input = image_input.to(device)
                    audio_input = audio_input.to(device)
                    video_input = video_input.to(device)

                    image_input = (image_input - image_input.min()) / (image_input.max() - image_input.min() + 1e-8)  # 0-1 normalize
                    audio_input = (audio_input - audio_input.mean()) / (audio_input.std() + 1e-8)  # Z-score
                    video_input = (video_input - video_input.min()) / (video_input.max() - video_input.min() + 1e-8)
                    
                    # Masking for MMM (15% per modality)
                    mask_prob = 0.15
                    mask_text = torch.rand_like(text_input.float()) < mask_prob
                    text_masked = text_input.clone().float()
                    #masked_indices = torch.where(mask_text)[1]  # Positions to mask
                    #text_masked[mask_text] = tokenizer.mask_token_id  # Use [MASK] token (103)
                    text_masked[mask_text] = 0
                    mask_image = torch.rand_like(image_input) > 0.15
                    image_masked = image_input * mask_image.float()
                    mask_audio = torch.rand_like(audio_input) > 0.15
                    audio_masked = audio_input * mask_audio.float()
                    mask_video = torch.rand_like(video_input) > 0.15
                    video_masked = video_input * mask_video.float()
                    # Similar for audio, video (simplified)
                    
                    optimizer.zero_grad()
                    doc = model(text_masked, image_masked, audio_masked, video_masked, recursion_depth=recursion_depth)
            else:
                # The rest of the training loop
                text_input, image_input, audio_input, video_input = text_input.to(device), image_input.to(device), audio_input.to(device), video_input.to(device)
                mask_prob = 0.15
                mask_text = torch.rand_like(text_input.float()) < mask_prob
                text_masked = text_input.clone().float()
                text_masked[mask_text] = 0
                mask_image = torch.rand_like(image_input) > 0.15
                image_masked = image_input * mask_image.float()
                mask_audio = torch.rand_like(audio_input) > 0.15
                audio_masked = audio_input * mask_audio.float()
                mask_video = torch.rand_like(video_input) > 0.15
                video_masked = video_input * mask_video.float()
                
                optimizer.zero_grad()
                doc = model(text_masked, image_masked, audio_masked, video_masked, recursion_depth=recursion_depth)

            # --- Loss Calculation on Final Step ---
            # MMM Loss (reconstruction)
            loss_text = criterion_mse(doc.text.float(), text_input.float() / vocab_size)
            loss_image = criterion_mse(doc.image, image_input)
            loss_audio = criterion_mse(doc.audio, audio_input)
            loss_video = criterion_mse(doc.video, video_input)
            loss_mmm = loss_text + loss_image + loss_audio + loss_video

            # CHA Loss
            temperature = 0.07
            modal_weights = [0.4, 0.2, 0.2, 0.2]
            embs = [doc.text.mean(dim=1), doc.image.mean(dim=(1,2,3)), doc.audio.mean(dim=1), doc.video.mean(dim=(1,2,3,4))]
            for k in range(len(embs)):
                if embs[k].dim() == 1: embs[k] = embs[k].unsqueeze(0)
            weighted_emb = torch.zeros_like(embs[0])
            for w, emb in zip(modal_weights, embs):
                weighted_emb += w * emb
            target = torch.cat([torch.ones(weighted_emb.size(0)), -torch.ones(weighted_emb.size(0))]).to(device)
            pos_pairs = weighted_emb.unsqueeze(1).repeat(1, weighted_emb.size(0), 1).view(-1, weighted_emb.size(1))
            neg_pairs = pos_pairs.roll(1, dims=0)
            loss_cha = criterion_contrast(pos_pairs, neg_pairs, target)

            loss = 0.5 * loss_mmm + 0.5 * loss_cha
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {i}/{len(train_dataloader)}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
                train_step_losses.append(loss.item())
            
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        total_loss += avg_epoch_loss
        print(f"Epoch {epoch+1}/{epochs} completed, Avg Loss: {avg_epoch_loss:.4f}")
        #scheduler.step(avg_epoch_loss)

        epoch_train_losses.append(avg_epoch_loss)

        if epoch < profile_epochs:
            prof.step()  # End epoch profile
            prof.export_chrome_trace(f"trace_epoch_{epoch}.json")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        model.eval()  # No gradients
        val_loss = 0.0
        with torch.no_grad():
            for i, val_batch in enumerate(val_dataloader):
                text_input, image_input, audio_input, video_input = val_batch  # Adjust for your unpacking
                text_input = text_input.to(device)
                image_input = image_input.to(device)
                audio_input = audio_input.to(device)
                video_input = video_input.to(device)

                doc = model(text_input.float(), image_input, audio_input, video_input, recursion_depth=recursion_depth)
                
                # MMM Loss (reconstruction)
                loss_text = criterion_mse(doc.text.float(), text_input.float() / vocab_size)
                loss_image = criterion_mse(doc.image, image_input)
                loss_audio = criterion_mse(doc.audio, audio_input)
                loss_video = criterion_mse(doc.video, video_input)
                loss_mmm = loss_text + loss_image + loss_audio + loss_video
                
                # CHA Loss
                temperature = 0.07
                modal_weights = [0.4, 0.2, 0.2, 0.2]
                embs = [doc.text.mean(dim=1), doc.image.mean(dim=(1,2,3)), doc.audio.mean(dim=1), doc.video.mean(dim=(1,2,3,4))]
                for k in range(len(embs)):
                    if embs[k].dim() == 1: embs[k] = embs[k].unsqueeze(0)
                
                weighted_emb = torch.zeros_like(embs[0])
                for w, emb in zip(modal_weights, embs):
                    weighted_emb += w * emb
                
                target = torch.cat([torch.ones(weighted_emb.size(0)), -torch.ones(weighted_emb.size(0))]).to(device)
                pos_pairs = weighted_emb.unsqueeze(1).repeat(1, weighted_emb.size(0), 1).view(-1, weighted_emb.size(1))
                neg_pairs = pos_pairs.roll(1, dims=0)
                loss_cha = criterion_contrast(pos_pairs, neg_pairs, target)

                loss = 0.5 * loss_mmm + 0.5 * loss_cha
                val_loss += loss.item()
                if i % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {i}/{len(val_dataloader)}, Validation Loss: {loss.item():.4f}")
                    val_step_losses.append(loss.item())
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}/{epochs} completed, Validation Loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)
        epoch_val_losses.append(avg_val_loss)
        model.train()  # Back to training

        # Align val steps to train steps scale
        train_steps = len(train_step_losses)
        val_x = np.linspace(0, train_steps, len(val_step_losses))

        plt.figure(figsize=(10, 5))
        plt.plot(range(train_steps), train_step_losses, label='Train Loss per Step', alpha=0.7)
        plt.plot(val_x, val_step_losses, label='Val Loss per Step', marker='o', alpha=0.7)
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title(f'Epoch {epoch+1} Step Losses (Aligned)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'step_loss_epoch_{epoch+1}.png'))
        plt.close()

        # Clear for next epoch
        train_step_losses = []
        val_step_losses = []

        for module in model.modules():
            if isinstance(module, LIFNeuron):
                module.v = torch.zeros_like(module.v).detach()
        
        # Save checkpoint
        #torch.save(model.state_dict(), os.path.join(output_dir, f"crina_pretrain_epoch_{epoch+1}.pt"))
    
    print(f"Pretraining complete! Final Avg Loss: {total_loss / epochs:.4f}")
    return model, epoch_train_losses, epoch_val_losses

if __name__ == '__main__':
    #train_crina()
    print("Pretraining Crina (woinw)...")
    model, train_losses, val_losses = pretrain_crina(epochs=5, lr=lr)
    torch.save(model.state_dict(), "crina_pretrain/crina_pretrain_final.pt")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', marker='o')
    if val_losses:
        plt.plot(val_losses, label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Crina-Synapse Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'final_loss_curve.png'))
    plt.show()