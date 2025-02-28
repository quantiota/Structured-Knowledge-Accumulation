import torch
import torch.nn as nn
import numpy as np

class SKAModel(nn.Module):
    def __init__(self, input_size=784, layer_sizes=[256, 128, 64, 10], K=50):
        """
        SKA model: Knowledge Accumulation and Entropy Minimization-based Learning.

        Args:
            input_size (int): Number of input neurons (default 784 for MNIST).
            layer_sizes (list): List of neurons per layer.
            K (int): Number of forward passes.
        """
        super(SKAModel, self).__init__()
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.K = K  # Number of forward learning steps

        # Initialize weights and biases as trainable tensors
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        prev_size = input_size
        for size in layer_sizes:
            self.weights.append(nn.Parameter(torch.randn(prev_size, size) * 0.01))
            self.biases.append(nn.Parameter(torch.zeros(size)))
            prev_size = size

        # Knowledge, decision, entropy, and history tracking tensors
        self.Z = [None] * len(layer_sizes)  # Knowledge tensors
        self.D = [None] * len(layer_sizes)  # Decision probabilities
        self.D_prev = [None] * len(layer_sizes)  # Previous decision probabilities
        self.delta_D = [None] * len(layer_sizes)  # Decision shifts
        self.entropy = [None] * len(layer_sizes)  # Layer-wise entropy

        # History tracking for visualization
        self.entropy_history = [[] for _ in range(len(layer_sizes))]
        self.cosine_history = [[] for _ in range(len(layer_sizes))]
        self.output_history = []  # Stores decision probabilities over steps

    def forward(self, x):
        """Forward pass through SKA model."""
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  # Flatten input images

        for l in range(len(self.layer_sizes)):
            # Compute structured knowledge tensor Z = Wx + b
            z = torch.mm(x, self.weights[l]) + self.biases[l]
            # Sigmoid activation for decision probabilities
            d = torch.sigmoid(z)
            # Store knowledge and decision probabilities
            self.Z[l] = z
            self.D[l] = d
            x = d  # Feed decision probabilities to the next layer

        return x

    def calculate_entropy(self):
        """Computes entropy minimization across layers."""
        total_entropy = 0
        for l in range(len(self.layer_sizes)):
            if self.Z[l] is not None and self.D_prev[l] is not None and self.D[l] is not None:
                # Compute decision shifts
                self.delta_D[l] = self.D[l] - self.D_prev[l]
                # Compute entropy using the SKA formula
                dot_product = torch.sum(self.Z[l] * self.delta_D[l])
                layer_entropy = -1 / np.log(2) * dot_product
                self.entropy[l] = layer_entropy.item()
                self.entropy_history[l].append(layer_entropy.item())

                # Compute alignment metric (cosine similarity)
                z_norm = torch.norm(self.Z[l])
                delta_d_norm = torch.norm(self.delta_D[l])
                cos_theta = dot_product / (z_norm * delta_d_norm) if z_norm > 0 and delta_d_norm > 0 else 0.0
                self.cosine_history[l].append(cos_theta)

                total_entropy += layer_entropy

        return total_entropy

    def ska_update(self, inputs, learning_rate=0.01):
        """Weight update using entropy minimization (no backpropagation)."""
        for l in range(len(self.layer_sizes)):
            if self.delta_D[l] is not None:
                prev_output = inputs.view(inputs.shape[0], -1) if l == 0 else self.D_prev[l-1]
                d_prime = self.D[l] * (1 - self.D[l])  # Sigmoid derivative
                gradient = -1 / np.log(2) * (self.Z[l] * d_prime + self.delta_D[l])
                dW = torch.matmul(prev_output.t(), gradient) / prev_output.shape[0]
                self.weights[l] -= learning_rate * dW
                self.biases[l] -= learning_rate * gradient.mean(dim=0)

    def initialize_tensors(self):
        """Resets knowledge and decision tensors at the start of training."""
        for l in range(len(self.layer_sizes)):
            self.Z[l], self.D[l], self.D_prev[l], self.delta_D[l], self.entropy[l] = None, None, None, None, None
            self.entropy_history[l] = []
            self.cosine_history[l] = []
        self.output_history = []
