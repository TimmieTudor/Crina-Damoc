import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset  # Hugging Face
from torchvision import datasets, transforms
import torchaudio
import json
import sys
import requests
import yt_dlp
import librosa

text_seq_len = 1024
audio_len = 128

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

# Step 2: Load Audio+Text (AudioCaps)
print("Loading AudioCaps for audio+text...")
audiocaps_dataset = load_dataset("d0rj/audiocaps", split="train[:20000]")  # 20k pairs
audiocaps_samples = []
for i, item in enumerate(audiocaps_dataset):
    text = item['caption'][:text_seq_len]
    tokens = list(text.lower())[:text_seq_len]
    ids = [ord(token) for token in tokens] + [0] * (text_seq_len - len(tokens))
    text_tensor = torch.tensor(ids, dtype=torch.long)

    if not os.path.exists("audiocaps_data"):
        os.mkdir("audiocaps_data")
    audio_filepath = download_audio(f"https://youtube.com/watch?v={item['youtube_id']}", "audiocaps_data", f"{item['audiocap_id']}")
    audio = audio_to_numpy(audio_filepath)
    if len(audio) < audio_len:
        audio = np.pad(audio, (0, audio_len - len(audio)))
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # [1, audio_len]
    audio_tensor = (audio_tensor - audio_tensor.mean()) / (audio_tensor.std() + 1e-8)  # Z-score
    audiocaps_samples.append((text_tensor, audio_tensor))
    print(f"Processed AudioCaps sample {i+1}")