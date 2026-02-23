import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PCLayer(nn.Module):
    """
    A single layer in a Predictive Coding hierarchy.
    It receives top-down predictions and generates bottom-up errors.
    """
    def __init__(self, dim_bottom, dim_top):
        super().__init__()
        # Top-down weights: predicts bottom layer from top layer
        self.W = nn.Parameter(torch.randn(dim_bottom, dim_top) * 0.1)
        self.state = None # To be initialized during inference

    def forward(self, x_top):
        # Generates prediction for the layer below
        return torch.matmul(x_top, self.W.t())

class PCNetwork(nn.Module):
    def __init__(self, layer_dims):
        super().__init__()
        self.layers = nn.ModuleList([
            PCLayer(layer_dims[i], layer_dims[i+1]) 
            for i in range(len(layer_dims) - 1)
        ])
        self.dims = layer_dims

    def initialize_states(self, batch_size):
        """Initializes latent states for all layers except the sensory input."""
        for i, layer in enumerate(self.layers):
            # layer.state is the 'hidden' representation for the level i
            # (which predicts the level below it)
            layer.state = torch.zeros(batch_size, self.dims[i+1], 
                                    device=device, requires_grad=True)

    def get_total_energy(self, x_sensory):
        """Total Energy = Sum of squared prediction errors across all levels."""
        energy = 0
        current_bottom = x_sensory
        
        for i, layer in enumerate(self.layers):
            prediction = layer(layer.state)
            error = current_bottom - prediction
            energy += (error**2).sum()
            
            # The next level's 'bottom' is this level's state
            current_bottom = layer.state
            
        # Optional: regularization on the top-most state to prevent explosion
        energy += 0.01 * (current_bottom**2).sum() 
        return energy

def train_pc():
    # Model: Input (64) <- Hidden (32) <- Top (16)
    dims = [64, 32, 16]
    model = PCNetwork(dims).to(device)
    
    # Target pattern (sensory input) - a simple sine wave pattern
    t = torch.linspace(0, 1, 64).to(device)
    target = torch.sin(2 * np.pi * t).repeat(8, 1) # Batch of 8
    
    # Hyperparameters
    inference_steps = 100
    learning_epochs = 200
    lr_state = 0.1
    lr_weights = 0.01
    
    energy_history = []

    print(f"Training Predictive Coding model on {device}...")

    for epoch in range(learning_epochs):
        # 1. INFERENCE PHASE (Update States, Fixed Weights)
        model.initialize_states(batch_size=8)
        state_optimizer = optim.SGD([l.state for l in model.layers], lr=lr_state)
        
        for _ in range(inference_steps):
            state_optimizer.zero_grad()
            energy = model.get_total_energy(target)
            energy.backward()
            state_optimizer.step()
            
            # Keep states detached from weight updates later
            for l in model.layers:
                l.state.data.clamp_(-5, 5) # Stability

        # 2. LEARNING PHASE (Update Weights, Fixed States)
        # We use a new optimizer for weights or just reuse a global one
        weight_optimizer = optim.Adam(model.parameters(), lr=lr_weights)
        weight_optimizer.zero_grad()
        
        # Energy calculated with final states from inference
        final_energy = model.get_total_energy(target)
        final_energy.backward()
        weight_optimizer.step()
        
        energy_history.append(final_energy.item())
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{learning_epochs}, Energy: {final_energy.item():.4f}")

    # Visualization
    plt.figure(figsize=(10, 5))
    
    # Plot Energy
    plt.subplot(1, 2, 1)
    plt.plot(energy_history)
    plt.title("Total Prediction Error (Energy)")
    plt.xlabel("Epoch")
    plt.ylabel("Energy")
    
    # Plot Reconstruction
    plt.subplot(1, 2, 2)
    with torch.no_grad():
        reconstruction = model.layers[0](model.layers[0].state)[0].cpu().numpy()
        plt.plot(target[0].cpu().numpy(), label="Target")
        plt.plot(reconstruction, label="PC Reconstruction", linestyle="--")
        plt.title("Sensory Reconstruction")
        plt.legend()
    
    plt.tight_layout()
    plt.savefig("pc_training_result.png")
    print("Training complete. Plot saved as 'pc_training_result.png'.")

if __name__ == "__main__":
    train_pc()
