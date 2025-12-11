import torch
import torch.nn.functional as F
import json
import numpy as np
import os
import argparse
from pathlib import Path

# Assume your CrinaSynapse model class is imported
from crina_model import CrinaSynapse  # Replace with your import

def load_task(json_file):
    """Load a single ARC task from JSON."""
    with open(json_file, 'r') as f:
        task = json.load(f)
    return task

def grid_to_tensor(grid, normalize=True):
    """Convert ARC grid (list of lists, 0-9) to tensor [1, 1, H, W]."""
    grid = np.array(grid, dtype=np.float32)
    if normalize:
        grid = grid / 9.0  # Normalize to [0,1]
    tensor = torch.tensor(grid).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    return tensor

def tensor_to_grid(tensor):
    """Convert tensor [1, 1, H, W] to ARC grid (list of lists, 0-9)."""
    grid = tensor.squeeze().numpy().round().clip(0, 9).astype(int).tolist()
    return grid

def evaluate_task(model, task, device, num_attempts=2, temperature=0.1):
    """Evaluate on one task: 3 demos + test input, predict output."""
    #print(task)
    train_tasks = task['train']
    test_input = grid_to_tensor(task['test'][0]['input'])
    
    # Few-shot: Average embeds from demos as "context"
    demo_context = torch.zeros(1, model.d_model, device=device)
    for demo in train_tasks[:3]:  # Use first 3 demos
        demo_input = grid_to_tensor(demo['input'])
        demo_output = grid_to_tensor(demo['output'])
        # Adapt Crina forward for grid (use image_embed as proxy)
        demo_pred = model.image_embed(demo_input)  # [1, d_model//16, H, W]
        demo_pred = demo_pred.mean(dim=(2,3))  # Flatten to [1, d_model//16]
        demo_pred = F.interpolate(demo_pred.unsqueeze(-1).unsqueeze(-1), size=(model.d_model, 1)).squeeze(-1).squeeze(-1)  # Dummy expand
        demo_context += demo_pred  # Accumulate context
    demo_context /= len(train_tasks[:3])  # Average [1, d_model]
    
    # Predict on test input (2 attempts with temperature noise)
    predictions = []
    for attempt in range(num_attempts):
        # Add temperature noise to context
        noisy_context = demo_context + temperature * torch.randn_like(demo_context)
        # Run Crina (adapt forward for single image input)
        with torch.no_grad():
            pred = model.image_embed(test_input)  # [1, d_model//16, H, W]
            pred = pred.mean(dim=(2,3))  # [1, d_model//16]
            pred = F.interpolate(pred.unsqueeze(-1).unsqueeze(-1), size=(model.d_model, 1)).squeeze(-1).squeeze(-1)  # Expand
            pred = pred + noisy_context  # Fuse with demo context
            pred_grid = model.image_output(pred.view(1, model.d_model//16, 1, 1))  # Dummy spatial for output
            pred_grid = pred_grid.view(1, 1, int(np.sqrt(model.d_model//16)), int(np.sqrt(model.d_model//16)))  # Reshape to grid
            pred_grid = tensor_to_grid(pred_grid)  # [H, W] integers 0-9
        predictions.append(pred_grid)
    
    # Check if any attempt matches ground truth
    test_output = task['test']['output']
    success = any(pred == test_output for pred in predictions)
    return success

def main(args):
    model = CrinaSynapse(d_model=256, n_heads=16, n_layers=4, sparsity_level=0.3, dropout=0.1)  # Your model params
    model.load_state_dict(torch.load(args.model_path))
    model.to(args.device)
    model.eval()
    
    data_dir = Path(args.data_dir)
    tasks = []
    for file in data_dir.glob("*.json"):
        task = load_task(file)
        tasks.append(task)
    
    total_success = 0
    for task in tasks:
        success = evaluate_task(model, task, args.device, num_attempts=args.num_attempts, temperature=args.temperature)
        if success:
            total_success += 1
        print(f"Task {task['task_id']}: {'Success' if success else 'Fail'}")
    
    accuracy = total_success / len(tasks) * 100
    print(f"Pass@{args.num_attempts} Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to Crina model checkpoint")
    parser.add_argument("--data_dir", type=str, default="./data/eval", help="Dir with JSON files")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--num_attempts", type=int, default=2, help="Attempts per task")
    parser.add_argument("--temperature", type=float, default=0.1, help="Noise for attempts")
    args = parser.parse_args()
    main(args)