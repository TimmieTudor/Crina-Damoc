import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PCLayer(nn.Module):
    def __init__(self, dim_bottom, dim_top):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim_bottom, dim_top) * 0.1)

    def predict(self, x_top):
        # Using a non-linearity helps in learning more complex mappings
        return torch.tanh(torch.matmul(x_top, self.W.t()))

class PCNetwork(nn.Module):
    def __init__(self, layer_dims):
        super().__init__()
        self.layers = nn.ModuleList([
            PCLayer(layer_dims[i], layer_dims[i+1]) 
            for i in range(len(layer_dims) - 1)
        ])
        self.dims = layer_dims

    def initialize_states(self, batch_size):
        self.states = []
        for d in self.dims[1:]:
            s = torch.zeros(batch_size, d, device=device, requires_grad=True)
            self.states.append(s)

    def get_total_energy(self, x_sensory):
        energy = 0
        current_bottom = x_sensory
        
        for i, layer in enumerate(self.layers):
            prediction = layer.predict(self.states[i])
            error = current_bottom - prediction
            energy += (error**2).sum()
            current_bottom = self.states[i]
            
        # Regularization on top-most state
        energy += 0.01 * (self.states[-1]**2).sum()
        return energy

    def generate(self, latent_vector):
        """Pure top-down generation from a latent seed."""
        with torch.no_grad():
            x = latent_vector
            for layer in reversed(self.layers):
                x = layer.predict(x)
            return x

def get_data(batch_size=12, seq_len=64):
    """Generates a batch with sine, square, and triangle waves."""
    t = torch.linspace(0, 1, seq_len).to(device)
    data = []
    
    # Sine
    data.append(torch.sin(2 * np.pi * t))
    # Square
    data.append(torch.sign(torch.sin(2 * np.pi * t)))
    # Triangle
    data.append(2 * torch.abs(2 * (t - torch.floor(t + 0.5))) - 1)
    
    data = torch.stack(data).repeat(batch_size // 3, 1)
    return data

def train_generative_pc():
    dims = [64, 64, 32] # Increased capacity for multiple patterns
    model = PCNetwork(dims).to(device)
    
    # Hyperparameters
    batch_size = 12
    inference_steps = 150
    epochs = 400
    
    lr_state = 0.05   # Reduced for stability
    lr_weights = 0.002 # Reduced for stability

    
    weight_optimizer = optim.Adam(model.parameters(), lr=lr_weights)
    
    print(f"Training Generative PC model on {device}...")
    target = get_data(batch_size)

    for epoch in range(epochs):
        # 1. Inference
        model.initialize_states(batch_size)
        state_optimizer = optim.SGD(model.states, lr=lr_state)
        
        for _ in range(inference_steps):
            state_optimizer.zero_grad()
            energy = model.get_total_energy(target)
            energy.backward()
            state_optimizer.step()
        
        # 2. Learning
        weight_optimizer.zero_grad()
        final_energy = model.get_total_energy(target)
        final_energy.backward()
        weight_optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Energy: {final_energy.item():.4f}")

    # --- GENERATION & INTERPOLATION ---
    print("Generating new samples and interpolating latent space...")
    
    # Get latent vectors for Sine (index 0) and Triangle (index 2) from our trained states
    with torch.no_grad():
        latent_sine = model.states[-1][0:1] # Latent for sine
        latent_triangle = model.states[-1][2:3] # Latent for triangle
    
    # Generate from pure latent seeds
    gen_sine = model.generate(latent_sine)
    gen_triangle = model.generate(latent_triangle)
    
    # Interpolate
    alphas = [0.25, 0.5, 0.75]
    interpolated_gens = []
    for a in alphas:
        interp_latent = (1 - a) * latent_sine + a * latent_triangle
        interpolated_gens.append(model.generate(interp_latent))

    # Visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 5, 1)
    plt.plot(gen_sine[0].cpu().numpy())
    plt.title("Generated: Sine")
    
    for i, a in enumerate(alphas):
        plt.subplot(1, 5, i + 2)
        plt.plot(interpolated_gens[i][0].cpu().numpy())
        plt.title(f"Interp: {a:.2f}")
        
    plt.subplot(1, 5, 5)
    plt.plot(gen_triangle[0].cpu().numpy())
    plt.title("Generated: Triangle")
    
    plt.tight_layout()
    plt.savefig("generative_pc_results.png")
    print("Results saved as 'generative_pc_results.png'.")

if __name__ == "__main__":
    train_generative_pc()
