import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import time

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load the pre-saved MNIST subset (100 samples per class)
mnist_subset = torch.load("mnist_subset_100_per_class.pt")
images = torch.stack([item[0] for item in mnist_subset])  # Shape: [1000, 1, 28, 28]
labels = torch.tensor([item[1] for item in mnist_subset])

# Prepare the dataset (single batch for SKA forward learning)
inputs = images  # No mini-batches, full dataset used for forward-only updates

# Define the SKA model with 4 layers
class SKAModel(nn.Module):
    def __init__(self, input_size=784, layer_sizes=[256, 128, 64, 10], K=50):
        super(SKAModel, self).__init__()
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.K = K  # Number of forward steps

        # Initialize weights and biases as nn.ParameterList
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        prev_size = input_size
        for size in layer_sizes:
            self.weights.append(nn.Parameter(torch.randn(prev_size, size) * 0.01))
            self.biases.append(nn.Parameter(torch.zeros(size)))
            prev_size = size

        # Tracking tensors for knowledge accumulation and entropy computation
        self.Z = [None] * len(layer_sizes)  # Knowledge tensors per layer
        self.D = [None] * len(layer_sizes)  # Decision probability tensors
        self.D_prev = [None] * len(layer_sizes)  # Previous decisions for computing shifts
        self.delta_D = [None] * len(layer_sizes)  # Decision shifts per step
        self.entropy = [None] * len(layer_sizes)  # Layer-wise entropy storage

        # Store entropy, cosine, and output distribution history for visualization
        self.entropy_history = [[] for _ in range(len(layer_sizes))]
        self.cosine_history = [[] for _ in range(len(layer_sizes))]
        self.output_history = []  # New: Store mean output distribution (10 classes) per step
        
        # New: Store Frobenius norms for each layer per forward step
        self.frobenius_history = [[] for _ in range(len(layer_sizes))]

    def forward(self, x):
        """Computes SKA forward pass, storing knowledge and decisions."""
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  # Flatten images

        for l in range(len(self.layer_sizes)):
            # Compute knowledge tensor Z = Wx + b
            z = torch.mm(x, self.weights[l]) + self.biases[l]
            # New: Compute and store Frobenius norm of z
            frobenius_norm = torch.norm(z, p='fro')
            self.frobenius_history[l].append(frobenius_norm.item())
            # Apply sigmoid activation to get decision probabilities
            d = torch.sigmoid(z)
            # Store values for entropy computation
            self.Z[l] = z
            self.D[l] = d
            x = d  # Output becomes input for the next layer

        return x

    def calculate_entropy(self):
        """Computes entropy reduction and cos(theta) per layer."""
        total_entropy = 0
        for l in range(len(self.layer_sizes)):
            if self.Z[l] is not None and self.D_prev[l] is not None and self.D[l] is not None:
                # Compute decision shifts
                self.delta_D[l] = self.D[l] - self.D_prev[l]
                # Entropy reduction using SKA formula
                dot_product = torch.sum(self.Z[l] * self.delta_D[l])
                layer_entropy = -1 / np.log(2) * dot_product
                self.entropy[l] = layer_entropy.item()
                self.entropy_history[l].append(layer_entropy.item())

                # Compute cos(theta) for alignment
                z_norm = torch.norm(self.Z[l])
                delta_d_norm = torch.norm(self.delta_D[l])
                if z_norm > 0 and delta_d_norm > 0:
                    cos_theta = dot_product / (z_norm * delta_d_norm)
                    self.cosine_history[l].append(cos_theta.item())
                else:
                    self.cosine_history[l].append(0.0)  # Default if norms are zero

                total_entropy += layer_entropy
        return total_entropy

    def ska_update(self, inputs, learning_rate=0.01):
        """Updates weights using entropy-based learning without backpropagation."""
        for l in range(len(self.layer_sizes)):
            if self.delta_D[l] is not None:
                # Previous layer's output
                prev_output = inputs.view(inputs.shape[0], -1) if l == 0 else self.D_prev[l-1]
                # Compute sigmoid derivative: D * (1 - D)
                d_prime = self.D[l] * (1 - self.D[l])
                # Compute entropy gradient
                gradient = -1 / np.log(2) * (self.Z[l] * d_prime + self.delta_D[l])
                # Compute weight updates via outer product
                dW = torch.matmul(prev_output.t(), gradient) / prev_output.shape[0]
                # Update weights and biases
                self.weights[l] = self.weights[l] - learning_rate * dW
                self.biases[l] = self.biases[l] - learning_rate * gradient.mean(dim=0)

    def initialize_tensors(self, batch_size):
        """Resets decision tensors at the start of each training iteration."""
        for l in range(len(self.layer_sizes)):
            self.Z[l] = None         # Reset knowledge tensors
            self.D[l] = None         # Reset current decision probabilities
            self.D_prev[l] = None    # Reset previous decision probabilities
            self.delta_D[l] = None   # Reset decision shifts
            self.entropy[l] = None   # Reset entropy storage
            self.entropy_history[l] = []  # Reset entropy history
            self.cosine_history[l] = []   # Reset cosine history
        self.output_history = []  # Reset output history
        # New: Reset Frobenius norm history
        self.frobenius_history = [[] for _ in range(len(self.layer_sizes))]

    def visualize_entropy_heatmap(self, step):
        """Dynamically scales the heatmap range and visualizes entropy reduction."""
        entropy_data = np.array(self.entropy_history)
        vmin = np.min(entropy_data)  # Dynamically set minimum entropy value
        vmax = 0.0  # Keep 0 as the upper limit for standardization
        plt.figure(figsize=(12, 8))
        sns.heatmap(entropy_data, cmap="Blues_r", vmin=vmin, vmax=vmax,  
                    xticklabels=range(1, entropy_data.shape[1] + 1),
                    yticklabels=[f"Layer {i+1}" for i in range(len(self.layer_sizes))])
        plt.title(f"Layer-wise Entropy Heatmap (Step {step})")
        plt.xlabel("Step Index K")
        plt.ylabel("Network Layers")
        plt.tight_layout()
        plt.savefig(f"entropy_heatmap_step_{step}.png")
        plt.show()

    def visualize_cosine_heatmap(self, step):
        """Visualizes cos(theta) alignment heatmap with a diverging scale."""
        cosine_data = np.array(self.cosine_history)
        plt.figure(figsize=(12, 8))
        sns.heatmap(cosine_data, cmap="coolwarm_r", vmin=-1.0, vmax=1.0,  
                    xticklabels=range(1, cosine_data.shape[1] + 1),
                    yticklabels=[f"Layer {i+1}" for i in range(len(self.layer_sizes))])
        plt.title(f"Layer-wise Cos(\u03B8) Alignment Heatmap (Step {step})")
        plt.xlabel("Step Index K")
        plt.ylabel("Network Layers")
        plt.tight_layout()
        plt.savefig(f"cosine_heatmap_step_{step}.png")
        plt.show()

    def visualize_frobenius_heatmap(self, step):
        """Visualizes the Frobenius Norm heatmap for the knowledge tensor Z across layers."""
        frobenius_data = np.array(self.frobenius_history)
        vmin = np.min(frobenius_data) if frobenius_data.size > 0 else 0
        vmax = np.max(frobenius_data) if frobenius_data.size > 0 else 1
        plt.figure(figsize=(12, 8))
        sns.heatmap(frobenius_data, cmap="viridis", vmin=vmin, vmax=vmax,
                    xticklabels=range(1, frobenius_data.shape[1] + 1),
                    yticklabels=[f"Layer {i+1}" for i in range(len(self.layer_sizes))])
        plt.title(f"Layer-wise Frobenius Norm Heatmap (Step {step})")
        plt.xlabel("Step Index K")
        plt.ylabel("Network Layers")
        plt.tight_layout()
        plt.savefig(f"frobenius_heatmap_step_{step}.png")
        plt.show()

    def visualize_output_distribution(self):
        """Plots the evolution of the 10-class output distribution over K steps."""
        output_data = np.array(self.output_history)  # Shape: [K, 10]
        plt.figure(figsize=(10, 6))
        plt.plot(output_data)  # Plot each class as a line
        plt.title('Output Decision Probability Evolution Across Steps (Single Pass)')
        plt.xlabel('Step Index K')
        plt.ylabel('Mean Sigmoid Output')
        plt.legend([f"Class {i}" for i in range(10)], loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("output_distribution_single_pass.png")
        plt.show()

# Training parameters
model = SKAModel()
learning_rate = 0.01

# SKA training over multiple forward steps
total_entropy = 0
step_count = 0
start_time = time.time()

# Initialize tensors for first step
model.initialize_tensors(inputs.size(0))

# Process K forward steps (without backpropagation)
for k in range(model.K):
    outputs = model.forward(inputs)
    # Store mean output distribution for the final layer
    model.output_history.append(outputs.mean(dim=0).detach().cpu().numpy())  # [10] vector
    if k > 0:  # Compute entropy after first step
        batch_entropy = model.calculate_entropy()
        model.ska_update(inputs, learning_rate)
        total_entropy += batch_entropy
        step_count += 1
        print(f'Step: {k}, Total Steps: {step_count}, Entropy: {batch_entropy:.4f}')
        model.visualize_entropy_heatmap(step_count)
        model.visualize_cosine_heatmap(step_count)  # Add cosine heatmap
        # New: Visualize Frobenius norm heatmap
        model.visualize_frobenius_heatmap(step_count)
    # Update previous decision tensors
    model.D_prev = [d.clone().detach() if d is not None else None for d in model.D]

# Final statistics
total_time = time.time() - start_time
avg_entropy = total_entropy / step_count if step_count > 0 else 0
print(f"Training Complete: Avg Entropy={avg_entropy:.4f}, Steps={step_count}, Time={total_time:.2f}s")

# Plot entropy history across layers
plt.figure(figsize=(8, 6))
plt.plot(np.array(model.entropy_history).T)  # Transpose for layer-wise visualization
plt.title('Entropy Evolution Across Layers (Single Pass)')
plt.xlabel('Step Index K')
plt.ylabel('Entropy')
plt.legend([f"Layer {i+1}" for i in range(len(model.layer_sizes))])
plt.grid(True)
plt.savefig("entropy_history_single_pass.png")
plt.show()

# Plot cosine history across layers (single pass)
plt.figure(figsize=(8, 6))
plt.plot(np.array(model.cosine_history).T)  # Transpose for layer-wise visualization
plt.title('Cos(\u03B8) Alignment Evolution Across Layers (Single Pass)')
plt.xlabel('Step Index K')
plt.ylabel('Cos(\u03B8)')
plt.legend([f"Layer {i+1}" for i in range(len(model.layer_sizes))])
plt.grid(True)
plt.savefig("cosine_history_single_pass.png")
plt.show()

# New: Plot Frobenius history across layers (single pass)
plt.figure(figsize=(8, 6))
plt.plot(np.array(model.frobenius_history).T)  # Transpose for layer-wise visualization
plt.title('Z Tensor Frobenius Norm Evolution Across Layers (Single Pass)')
plt.xlabel('Step Index K')
plt.ylabel('Z Tensor Frobenius Norm')
plt.legend([f"Layer {i+1}" for i in range(len(model.layer_sizes))])
plt.grid(True)
plt.savefig("frobenius_history_single_pass.png")
plt.show()

# Plot output distribution history
model.visualize_output_distribution()

print("Training complete. Visualizations generated.")
