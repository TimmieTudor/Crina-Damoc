import torch
import torch.nn as nn
from torch.nn import functional as F
from crina_model import CrinaSynapse
from thop import profile

if __name__ == "__main__":
    d_model = 256
    n_heads = 16
    n_layers = 4
    dropout = 0.1
    sparsity_level = 0.3
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    model = CrinaSynapse(d_model=d_model, n_heads=n_heads, n_layers=n_layers, sparsity_level=sparsity_level, dropout=dropout).to(device)
    #model.load_state_dict(torch.load("crina_pretrain/crina_pretrain_final.pt", weights_only=True))
    model.eval()

    # Dummy inputs matching your model (adjust shapes)
    dummy_text = torch.randn(1, 1024).to(device)  # [1, 1024]
    dummy_image = torch.randn(1, 3, 32, 32).to(device)  # [1, 3, 32, 32]
    dummy_audio = torch.randn(1, 128).to(device)  # [1, 128]
    dummy_video = torch.randn(1, 3, 10, 32, 32).to(device)  # [1, 3, 10, 32, 32]

    # Profile with dummy forward pass
    flops, params = profile(model, inputs=(dummy_text, dummy_image, dummy_audio, dummy_video, 3), verbose=False)
    print(f"Model FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Model Parameters: {params / 1e6:.2f} M")