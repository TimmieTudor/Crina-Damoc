import torch
import torchaudio
import torchvision
from datasets import load_dataset
import numpy as np
from PIL import Image
import os
import json
import zipfile
import rarfile  # For RAR extraction; install: pip install unrar
import sys

if __name__ == "__main__":
    # Define target dimensions
    text_seq_len = 1024
    image_channels, image_size = 3, 32
    audio_len = 128
    video_frames, video_channels, video_size = 10, 3, 32
    batch_size = 32
    output_dir = "multimodal_dataset_refined"
    metadata_file = "dataset_metadata.json"

    # Paths for UCF101
    ucf_root = "./data/ucf101/UCF-101"  # Where videos will be extracted
    ucf_annotation = "./data/ucf101/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist"  # Annotation dir
    ucf_rar = "UCF101.rar"  # RAR file to download
    ucf_zip = "UCF101TrainTestSplits-RecognitionTask.zip"  # ZIP for annotations

    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ucf_root, exist_ok=True)

    # Download and extract UCF101 if not present
    if not os.path.exists(os.path.join(ucf_root, "ApplyEyeMakeup")):  # Check for extracted videos
        print("Downloading UCF101 videos (~6.9GB)...")
        if not os.path.exists(ucf_rar):
            import requests
            url = "http://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
            r = requests.get(url, stream=True)
            with open(ucf_rar, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Extracting UCF101 videos...")
        with rarfile.RarFile(ucf_rar) as rf:
            rf.extractall(ucf_root)

    if not os.path.exists(ucf_annotation):
        print("Downloading UCF101 annotations (~0.1MB)...")
        if not os.path.exists(ucf_zip):
            import requests
            url = "http://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"
            r = requests.get(url, stream=True)
            with open(ucf_zip, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Extracting UCF101 annotations...")
        with zipfile.ZipFile(ucf_zip, 'r') as zf:
            zf.extractall(ucf_root)
        ucf_annotation = os.path.join(ucf_root, "UCF101TrainTestSplits-RecognitionTask")

    # Load datasets
    text_dataset = load_dataset("squad", split="train[:10080]")
    image_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
    audio_dataset = load_dataset("librispeech_asr", "clean", split="train.100[:10080]")
    video_dataset = torchvision.datasets.UCF101(root=ucf_root, annotation_path=ucf_annotation, train=True, frames_per_clip=video_frames)

    print(type(video_dataset), len(text_dataset), len(image_dataset), len(audio_dataset), len(video_dataset) if hasattr(video_dataset, '__len__') else "video_dataset not callable")
    # Preprocess functions
    def preprocess_text(text):
        #print(list(text.lower()))
        #print(len(list(text.lower())))
        tokens = list(text.lower())[:text_seq_len]
        tokens += ["" for _ in range(text_seq_len - len(tokens))]
        return tokens

    def preprocess_image(img, idx):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = img.resize((image_size, image_size), Image.LANCZOS)
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

    def preprocess_audio(audio, idx):
        if "array" not in audio:
            print(f"Warning: Missing 'array' key in audio at index {idx}, skipping")
            return None
        waveform, sr = audio["array"], audio["sampling_rate"]
        #print(f"Index {idx}: waveform shape {waveform.shape if hasattr(waveform, 'shape') else 'No shape'}, type {type(waveform)}, sr {sr}")  # Debug line
        if not isinstance(waveform, (list, np.ndarray, torch.Tensor)):
            print(f"Warning: Invalid 'array' type ({type(waveform)}) at index {idx}, skipping")
            return None
        waveform = torch.tensor(waveform) if not isinstance(waveform, torch.Tensor) else waveform
        if waveform.dim() != 2:
            waveform = waveform.unsqueeze(0)  # Ensure [1, time] if 1D
        sr = int(sr) if not isinstance(sr, int) else sr  # Ensure sr is int
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if waveform.size(1) > audio_len:
            waveform = waveform[:, :audio_len]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, audio_len - waveform.size(1)))
        return waveform.squeeze(0).float()

    def preprocess_video(video, idx):
        frames = video[0]  # [frames, H, W, C]
        resized_frames = []
        for f_idx, frame in enumerate(frames):
            # Validate shape
            if frame.dim() != 3:
                print(f"Warning: Invalid frame dim {frame.dim()} at frame {f_idx} of sample {idx}, skipping")
                return None
            h, w, c = frame.shape
            if c not in (1, 3):
                print(f"Warning: Invalid channels {c} at frame {f_idx} of sample {idx}, skipping")
                return None
            if h != 240 or w != 320:  # Standard UCF101
                print(f"Warning: Invalid H/W {h}x{w} at frame {f_idx} of sample {idx}, skipping")
                return None
            
            # Convert to RGB if 1-channel
            if c == 1:
                frame = frame.repeat(1, 1, 3)
            
            # Normalize and convert to uint8
            #frame_rgb = frame.permute(2, 0, 1).numpy()  # [C, H, W]
            frame_rgb = frame.numpy()
            frame_rgb = (frame_rgb * 255).clip(0, 255).astype(np.uint8)

            #print(frame_rgb.shape)
            #sys.exit()
            
            try:
                #print("setting up image")
                frame_img = Image.fromarray(frame_rgb).resize((video_size, video_size), Image.LANCZOS)
                #print("resizing frame")
                resized_frame = torch.from_numpy(np.array(frame_img)).permute(2, 0, 1).float() / 255.0
                #print("adding frame")
                resized_frames.append(resized_frame)
            except Exception as e:
                print(f"Warning: Failed to process frame {f_idx} of sample {idx}: {e}, skipping")
                sys.exit()
                return None
        
        if not resized_frames:
            return None
        frames = torch.stack(resized_frames)  # [frames, channels, video_size, video_size]
        
        if len(frames) > video_frames:
            frames = frames[:video_frames]
        else:
            pad_zeros = torch.zeros(video_frames - len(frames), video_channels, video_size, video_size)
            frames = torch.cat([frames, pad_zeros], dim=0)
        return frames.permute(1, 0, 2, 3)  # [channels, frames, H, W]
    
    # Alignment function
    def align_samples(text_idx, image_idx, audio_idx, video_idx):
        max_len = min(len(text_dataset), len(image_dataset), len(audio_dataset), len(video_dataset))
        aligned_idx = (text_idx % max_len, image_idx % max_len, audio_idx % max_len, video_idx % max_len)
        return aligned_idx

    # Prepare and save dataset with metadata
    dataset = []
    metadata = []
    for i in range(0, min(len(text_dataset), len(image_dataset), len(audio_dataset), len(video_dataset)), batch_size):
        batch_metadata = []
        for j in range(batch_size):
            try:
                idx = i + j
                text_idx, image_idx, audio_idx, video_idx = align_samples(idx, idx, idx, idx)
                text = preprocess_text(text_dataset[text_idx]["context"])
                #print("text preprocessed")
                image = preprocess_image(image_dataset[image_idx][0], image_idx)
                #print("images preprocessed")
                audio = preprocess_audio(audio_dataset[audio_idx]["audio"], audio_idx)
                #print("audio preprocessed")
                video = preprocess_video(video_dataset[video_idx], video_idx)
                #print("video preprocessed")
                sample = (text, image, audio, video)
                dataset.append(sample)
                batch_metadata.append({
                    "text_idx": text_idx,
                    "image_idx": image_idx,
                    "audio_idx": audio_idx,
                    "video_idx": video_idx,
                    "context": text_dataset[text_idx]["context"][:64]
                })
                print(f"Processed sample {idx}")
            except Exception as e:
                print(f"Skipped sample {idx} due to error: {e}")
                continue
        
        if dataset:
            batch_num = i // batch_size
            torch.save(dataset, os.path.join(output_dir, f"batch_{batch_num}.pt"))
            metadata.append(batch_metadata)
            dataset = []

    if dataset:
        batch_num = len(os.listdir(output_dir))
        torch.save(dataset, os.path.join(output_dir, f"batch_{batch_num}.pt"))
        metadata.append(batch_metadata)

    # Save metadata
    with open(os.path.join(output_dir, metadata_file), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset prepared with {len(os.listdir(output_dir))} batches in {output_dir} with metadata")