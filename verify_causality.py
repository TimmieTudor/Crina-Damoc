import torch
import torch.nn as nn
from test_script_3 import TestTreeSelfAttention
from test_attention import TestModel

def verify_causality():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    D = 64
    T = 8
    #model = TestTreeSelfAttention(D, is_causal=True).to(device).eval()
    model = TestModel(D).to(device).eval()
    
    # Input 1
    x1 = torch.randn(1, T, D).to(device)
    
    # Input 2: same as x1 except for the last token
    x2 = x1.clone()
    x2[0, -1, :] += 1.0
    
    with torch.no_grad():
        out1 = model(x1)
        
        # RESET STATE to prevent leakage (Crucial for SNNs and Temporal Memory)
        #model.reset_state()
        for m in model.modules():
            if hasattr(m, 'reset'):
                m.reset()

                
        out2 = model(x2)

        
    # Check if all tokens except the last one are identical
    diff = (out1[:, :-1, :] - out2[:, :-1, :]).abs().max().item()
    
    print(f"Max difference in past tokens: {diff:.12f}")
    if diff < 1e-6:
        print("SUCCESS: Causality is intact.")
    else:
        print("FAILURE: Lookahead bias detected!")

if __name__ == "__main__":
    verify_causality()
