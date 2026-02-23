import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-using the LIFNeuron concept from the user's codebase
class LIFNeuron(nn.Module):
    def __init__(self, d_model):
        super(LIFNeuron, self).__init__()
        self.d_model = d_model
        # Using fixed params for the demo to focus on PC learning
        self.threshold = 1.0
        self.tau = 0.5
        self.v_reset = 0.0
        self.v = torch.zeros(d_model).to(device)

    def forward(self, x):
        # x is the input CURRENT (prediction error)
        # We run a single time step of LIF dynamics
        self.v = self.tau * self.v + x
        spike = (self.v >= self.threshold).float()
        self.v = self.v * (1 - spike) + self.v_reset * spike
        return spike

    def reset(self):
        self.v = torch.zeros(self.d_model, device=device)

class SpikingPCLayer(nn.Module):
    def __init__(self, dim_bottom, dim_top):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim_bottom, dim_top) * 0.1)
        self.lif = LIFNeuron(dim_top)
        self.dim_top = dim_top

    def predict(self, x_top_spikes):
        # x_top_spikes: (batch, dim_top) -> prediction: (batch, dim_bottom)
        return torch.matmul(x_top_spikes, self.W.t())

class SpikingPCNetwork(nn.Module):
    def __init__(self, layer_dims):
        super().__init__()
        self.layers = nn.ModuleList([
            SpikingPCLayer(layer_dims[i], layer_dims[i+1]) 
            for i in range(len(layer_dims) - 1)
        ])
        self.dims = layer_dims

    def run_snn_inference(self, x_sensory, time_steps=50):
        """
        Runs the SNN for multiple time steps.
        Latent layers integrate prediction errors into their membrane potentials.
        Returns the time-averaged spike rates.
        """
        batch_size = x_sensory.shape[0]
        
        # Reset LIF states
        for l in self.layers:
            l.lif.reset()
            l.avg_spikes = torch.zeros(batch_size, l.dim_top, device=device)

        # Record sensory error dynamics
        current_bottom_spikes = x_sensory # We treat input as 'spikes' for consistency

        for t in range(time_steps):
            # Top-down pass: layers generate predictions for layer below
            # For simplicity in this demo, we use the average spikes as 'state' for prediction
            # In a full temporal SNN, this would be a feedback loop of spikes.
            
            for i in range(len(self.layers)):
                layer = self.layers[i]
                
                # Each layer's LIF integrates the error from below
                # Error = sensory_input - prediction_from_above
                # Here we use a simplified error-to-current mapping
                if i == 0:
                    pred = layer.predict(layer.avg_spikes)
                    error = x_sensory - pred
                else:
                    pred = layer.predict(layer.avg_spikes)
                    error = self.layers[i-1].avg_spikes - pred
                
                # Integrate error CURRENT into LIF
                # If error is positive, V increases. If error is negative, V increases (absolute error)
                # or we can use two-sided error neurons. For this demo: absolute error.
                current = torch.abs(error).mean(dim=-1, keepdim=True).repeat(1, layer.dim_top)
                
                spikes = layer.lif(current * 2.0) # Scale current to cross threshold
                layer.avg_spikes = (layer.avg_spikes * t + spikes) / (t + 1)

    def get_total_energy(self, x_sensory):
        energy = 0
        current_bottom = x_sensory
        for i, layer in enumerate(self.layers):
            prediction = layer.predict(layer.avg_spikes)
            error = current_bottom - prediction
            energy += (error**2).sum()
            current_bottom = layer.avg_spikes
        return energy

def train_spiking_pc():
    dims = [64, 32, 16]
    model = SpikingPCNetwork(dims).to(device)
    
    # Target pattern (sensory input)
    t = torch.linspace(0, 1, 64).to(device)
    target = (torch.sin(2 * np.pi * t) > 0).float().repeat(8, 1) # Square wave spikes
    
    epochs = 100
    lr_weights = 0.01
    weight_optimizer = optim.Adam(model.parameters(), lr=lr_weights)
    
    energy_history = []

    print(f"Training Spiking Predictive Coding on {device}...")

    for epoch in range(epochs):
        # 1. INFERENCE PHASE (Temporal Spiking)
        with torch.no_grad():
            model.run_snn_inference(target, time_steps=40)

        # 2. LEARNING PHASE (Local Weight Updates)
        weight_optimizer.zero_grad()
        energy = model.get_total_energy(target)
        energy.backward()
        weight_optimizer.step()
        
        energy_history.append(energy.item())
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Energy: {energy.item():.4f}")

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(energy_history)
    plt.title("Spiking PC: Total Prediction Error")
    
    plt.subplot(1, 2, 2)
    with torch.no_grad():
        reconstruction = model.layers[0].predict(model.layers[0].avg_spikes)[0].cpu().numpy()
        plt.plot(target[0].cpu().numpy(), label="Target Spikes")
        plt.plot(reconstruction, label="Reconstruction", linestyle="--")
        plt.legend()
        plt.title("SNN-PC Reconstruction")
    
    plt.savefig("spiking_pc_results.png")
    print("Results saved to 'spiking_pc_results.png'.")

if __name__ == "__main__":
    train_spiking_pc()
