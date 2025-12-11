import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset  # Hugging Face
from torchvision import datasets, transforms
import torchaudio
from PIL import Image
import random
from sklearn.model_selection import train_test_split
import json
import sys
import requests
import yt_dlp
import librosa
import tqdm

# Hyperparameters (match your model)
batch_size = 32
num_samples = 50000  # Scale to 100k+ as needed
text_seq_len = 1024
image_size = 32
audio_len = 128
video_frames = 10
video_size = 32
image_channels = 3
video_channels = 3
output_dir = "multimodal_dataset_large"
train_split = 0.8
os.makedirs(output_dir, exist_ok=True)

def download_audio(url, output_dir='.', filename=None):
    """
    Downloads the audio of a YouTube video as an MP3 file and returns the full path 
    to the downloaded file.
    
    Args:
        url (str): The URL of the YouTube video.
        output_dir (str): The directory to save the file.
        filename (str, optional): The base filename to use (without extension).
                                  If None, the video title is used.
                                  
    Returns:
        str or None: The full path to the downloaded MP3 file, or None on failure.
    """
    
    # 1. Define the output template
    # If a specific filename is provided, use it. Otherwise, use the title.
    file_template = f'{filename}.%(ext)s' if filename else '%(title)s.%(ext)s'
    outtmpl = os.path.join(output_dir, file_template)

    expected_path_prefix = os.path.join(output_dir, filename if filename else 'Video')
    if filename and os.path.exists(f"{expected_path_prefix}.mp3"):
        #print(f"✅ Skipping download: File '{filename}.mp3' already exists.")
        return f"{expected_path_prefix}.mp3"

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': outtmpl,
        'noplaylist': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'noprogress': True,
        #'progress_hooks': [lambda d: print(f"Status: {d['status']}") if d['status'] != 'finished' else None]
        #'progress_hooks': []
    }

    try:
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            #print(f"Attempting to download audio from: {url}")
            
            # extract_info performs the download when 'download=True'
            info = ydl.extract_info(url, download=True)
            
            # The 'info' dictionary contains the final file path under '_filepath'
            # after all post-processors (like MP3 conversion) have run.
            final_filepath = info.get('requested_downloads', [{}])[-1].get('filepath')

            if final_filepath and os.path.exists(final_filepath):
                #print(f"\nSuccessfully downloaded audio to: {final_filepath}")
                return final_filepath
            else:
                #print("\nDownload finished, but file path could not be determined.")
                return None
                
    except Exception as e:
        #print(f"\nAn error occurred: {e}")
        return None

def audio_to_numpy(audio_file_path, target_len=128, sample_rate=16000):
    """
    Convert audio file to NumPy array for Crina dataset.
    Args:
        audio_file_path (str): Path to .wav, .mp3, etc.
        target_len (int): Length to truncate/pad to (e.g., 128).
        sample_rate (int): Target SR (e.g., 16kHz).
    Returns:
        np.array: Normalized waveform [target_len].
    """
    # Load audio
    waveform, sr = librosa.load(audio_file_path, sr=sample_rate, mono=True)  # Mono for simplicity
    
    # Truncate or pad
    if len(waveform) > target_len:
        waveform = waveform[:target_len]
    else:
        waveform = np.pad(waveform, (0, target_len - len(waveform)), mode='constant')
    
    # Normalize to z-score
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)
    
    return waveform  # np.array [target_len]

# Step 1: Load Diverse Text+Image (MS COCO)
print("Loading MS COCO for text+image...")
coco_dataset = load_dataset("phiyodr/coco2017", split="train[:30000]")  # 30k pairs
coco_samples = []
for item in tqdm.tqdm(coco_dataset):
    text = item['captions'][0]  # Caption as text
    tokens = list(text.lower())[:text_seq_len]
    ids = [ord(token) for token in tokens] + [0] * (text_seq_len - len(tokens))
    text_tensor = torch.tensor(ids, dtype=torch.long)
    #print(item.keys())
    #sys.exit()

    image_path = os.path.join("coco_data", item['file_name'])
    if not os.path.exists(image_path):
        image_response = requests.get(item['coco_url'])
        if image_response.status_code == 200:
            if not os.path.exists("coco_data/train2017"):
                os.mkdir("coco_data/train2017")
            with open(image_path, 'wb') as f:
                f.write(image_response.content)
    image_bitmap = Image.open(image_path, 'r')
    image = np.array(image_bitmap)  # PIL Image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2)  # Augmentation
    ])
    image_tensor = transform(image).unsqueeze(0)  # [1, 3, 32, 32]
    coco_samples.append((text_tensor, image_tensor))
    #print(f"Processed COCO Sample {i+1}")

# Step 2: Load Audio+Text (AudioCaps)
print("Loading AudioCaps for audio+text...")
audiocaps_dataset = load_dataset("d0rj/audiocaps", split="train[:20000]")  # 20k pairs
audiocaps_samples = []
for item in tqdm.tqdm(audiocaps_dataset):
    text = item['caption'][:text_seq_len]
    tokens = list(text.lower())[:text_seq_len]
    ids = [ord(token) for token in tokens] + [0] * (text_seq_len - len(tokens))
    text_tensor = torch.tensor(ids, dtype=torch.long)

    if not os.path.exists("audiocaps_data"):
        os.mkdir("audiocaps_data")
    audio_filepath = download_audio(f"https://youtube.com/watch?v={item['youtube_id']}", "audiocaps_data", f"{item['audiocap_id']}")
    if audio_filepath is None: continue
    audio = audio_to_numpy(audio_filepath)
    if len(audio) < audio_len:
        audio = np.pad(audio, (0, audio_len - len(audio)))
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # [1, audio_len]
    #audio_tensor = (audio_tensor - audio_tensor.mean()) / (audio_tensor.std() + 1e-8)  # Z-score
    audiocaps_samples.append((text_tensor, audio_tensor))
    #print(f"Processed AudioCaps sample {i+1}")

# Step 3: Load Video+Text (VATEX)
print("Loading VATEX for video+text...")
vatex_dataset = load_dataset("vatex", split="train[:20000]")  # 20k pairs
vatex_samples = []
for item in vatex_dataset:
    text = item['caption'][:text_seq_len]
    tokens = list(text.lower())[:text_seq_len]
    ids = [ord(token) for token in tokens] + [0] * (text_seq_len - len(tokens))
    text_tensor = torch.tensor(ids, dtype=torch.long)
    
    # Simulate video frames (in practice, extract from video path)
    frames = torch.randn(video_frames, video_channels, video_size, video_size)  # Synthetic
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((video_size, video_size))])
    video_tensor = torch.stack([transform(Image.fromarray(frame.permute(1, 2, 0).numpy() * 255)) for frame in frames])  # [video_frames, 3, 32, 32]
    video_tensor = video_tensor.unsqueeze(0)  # [1, 10, 3, 32, 32]
    vatex_samples.append((text_tensor, video_tensor))

# Step 4: Load Additional Audio (LibriSpeech for diversity)
print("Loading LibriSpeech for audio...")
libri_dataset = load_dataset("librispeech_asr", "clean", split="train.100[:10000]")  # 10k clips
libri_samples = []
for item in libri_dataset:
    audio = item['audio']['array'][:audio_len]
    if len(audio) < audio_len:
        audio = np.pad(audio, (0, audio_len - len(audio)))
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # [1, audio_len]
    audio_tensor = (audio_tensor - audio_tensor.mean()) / (audio_tensor.std() + 1e-8)
    libri_samples.append(audio_tensor)

# Step 5: Align and Balance Dataset (Cycle indices for diversity)
print("Aligning dataset...")
dataset_samples = []
for i in range(num_samples):
    text = random.choice([c[0] for c in coco_samples + audiocaps_samples + vatex_samples])  # Random text
    image = random.choice([c[1] for c in coco_samples]) if random.random() > 0.5 else torch.randn(1, image_channels, image_size, image_size)  # 50% real, 50% synth
    audio = random.choice([c[1] for c in audiocaps_samples] + libri_samples)
    video = random.choice([c[1] for c in vatex_samples]) if random.random() > 0.5 else torch.randn(1, video_frames, video_channels, video_size, video_size)
    sample = (text.tolist(), image.squeeze(0).numpy(), audio.squeeze(0).numpy(), video.squeeze(0).numpy())
    dataset_samples.append(sample)

# Step 6: Split and Save Batched .pt Files
print("Splitting and saving...")
train_samples, val_samples = train_test_split(dataset_samples, test_size=0.2, random_state=42)

def save_batches(samples, split_name):
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        torch.save(torch.tensor(batch), os.path.join(split_dir, f"batch_{i//batch_size}.pt"))

save_batches(train_samples, "train")
save_batches(val_samples, "val")

# Step 7: Custom Dataset Class
class MultimodalDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        split_dir = os.path.join(data_dir, split)
        self.files = [os.path.join(split_dir, f) for f in sorted(os.listdir(split_dir)) if f.endswith('.pt')]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        batch = torch.load(self.files[idx])
        text, image, audio, video = batch[0]  # Unpack first sample
        return torch.tensor(text), torch.tensor(image), torch.tensor(audio), torch.tensor(video)

# Example Usage
train_dataset = MultimodalDataset(output_dir, "train")
val_dataset = MultimodalDataset(output_dir, "val")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("Dataset prepared! Sample shapes:")
for batch in train_dataloader:
    text, image, audio, video = batch
    print(f"Text: {text.shape}, Image: {image.shape}, Audio: {audio.shape}, Video: {video.shape}")
    break