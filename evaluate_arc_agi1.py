import torch
import torch.nn.functional as F
import json
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm

# Import the model components
from crina_tinyshakespeare import CrinaSynapse, reset_all_lif_neurons

def load_task(json_file):
    """Load a single ARC task from JSON."""
    with open(json_file, 'r') as f:
        task = json.load(f)
    return task

def grid_to_sequence(grid):
    """Flatten ARC grid to a sequence of tokens [0-9]."""
    return [pixel for row in grid for pixel in row]

def sequence_to_grid(seq, h, w):
    """Reshape a sequence back into a grid of (h, w)."""
    grid = []
    for i in range(h):
        grid.append(seq[i*w : (i+1)*w])
    return grid

def evaluate_task(model, task, device, num_attempts=3, temperature=0.1):
    """Evaluate on one task: Serialize demos + test input into a single sequence."""
    train_tasks = task['train']
    test_info = task['test'][0]
    test_input = test_info['input']
    test_output = test_info['output']
    target_h, target_w = len(test_output), len(test_output[0])
    
    # Separator token
    SEP = 10
    
    # Construct few-shot prompt
    full_prompt = []
    for demo in train_tasks[:3]:  # Use up to 3 demos
        full_prompt += grid_to_sequence(demo['input'])
        full_prompt.append(SEP)
        full_prompt += grid_to_sequence(demo['output'])
        full_prompt.append(SEP)
    
    full_prompt += grid_to_sequence(test_input)
    full_prompt.append(SEP)
    
    prompt_tensor = torch.tensor(full_prompt, dtype=torch.long).unsqueeze(0).to(device)
    
    for attempt in range(num_attempts):
        reset_all_lif_neurons(model)
        generated = []
        current_seq = prompt_tensor.clone()
        
        with torch.no_grad():
            for _ in range(target_h * target_w):
                logits = model(current_seq)
                next_token_logits = logits[:, -1, :]
                
                if temperature > 0 and attempt > 0: # Add noise only after first failed attempt
                    probs = F.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated.append(next_token.item())
                current_seq = torch.cat([current_seq, next_token], dim=1)
                
                if current_seq.size(1) > 1025:
                    current_seq = current_seq[:, -1025:]
                    
        pred_grid = sequence_to_grid(generated, target_h, target_w)
        if pred_grid == test_output:
            return True, attempt + 1
            
    return False, num_attempts

def load_model_flexible(model_path, device):
    """Load a checkpoint and adjust vocab_size dynamically."""
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle torch.compile prefix '_orig_mod.'
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("_orig_mod.", "")
        new_state_dict[name] = v
    state_dict = new_state_dict

    # Detect vocab size from embedding weights
    checkpoint_vocab_size = state_dict['embed.weight'].shape[0]
    print(f"Detected checkpoint vocab_size: {checkpoint_vocab_size}")
    model = CrinaSynapse(vocab_size=checkpoint_vocab_size, d_model=256, n_layers=8, tree_depth=4)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def main(args):
    print(f"--- ARC-AGI-1 Evaluation (Path: {args.data_dir}) ---")
    model = load_model_flexible(args.model_path, args.device)
    
    data_dir = Path(args.data_dir)
    tasks = list(data_dir.glob("*.json"))
    
    if not tasks:
        print(f"No tasks found in {args.data_dir}. Checking fallback...")
        return

    total_success = 0
    pbar = tqdm(tasks)
    for file in pbar:
        task = load_task(file)
        success, attempts = evaluate_task(model, task, args.device, num_attempts=args.num_attempts, temperature=args.temperature)
        if success:
            total_success += 1
        pbar.set_postfix({'success': f"{total_success}/{len(tasks)}", 'last_attempts': attempts})
    
    accuracy = total_success / len(tasks) * 100
    print(f"\nFinal Result: {total_success}/{len(tasks)} ({accuracy:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./ARC/data/evaluation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_attempts", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
