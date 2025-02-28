# Structured Knowledge Accumulation: An Autonomous Framework for Layer-Wise Entropy Reduction in Neural Learning 



## Overview

The **Structured Knowledge Accumulation (SKA) Framework** introduces a novel learning paradigm that eliminates backpropagation and instead **optimizes learning through entropy minimization**. SKA redefines entropy as a measure of structured knowledge alignment and offers a biologically inspired, forward-only approach to neural learning.

This repository contains:
- **SKA Model Definition**: The PyTorch implementation of the SKA learning framework.
- **Run SKA Script**: A standalone script that applies SKA to the **MNIST** dataset to demonstrate entropy reduction and structured knowledge evolution.

## Features

- **Forward-Only Learning**: No backpropagation, enabling biologically plausible AI training.
- **Entropy-Based Optimization**: Learning is driven by **entropy minimization** instead of gradient descent.
- **Layer-Wise Knowledge Alignment**: SKA exhibits **self-organized entropy reduction**, leading to equilibrium learning dynamics.
- **Visualization Tools**: Generates entropy heatmaps, cosine alignment plots, and decision probability evolution charts.

---

## Installation

### Prerequisites
Ensure you have Python **3.8+** installed, along with the necessary dependencies:

```bash
pip install torch torchvision numpy matplotlib seaborn
```

### Clone the Repository
```bash
git clone https://github.com/quantiota/Structured-Knowledge-Accumulation.git
cd Structured-Knowledge-Accumulation
```

### Download MNIST Subset
The SKA experiment uses a **subset of MNIST** (100 samples per class). If you don't have the dataset, generate it using:

```python
from torchvision import datasets, transforms
import torch

# Load MNIST dataset and extract 100 samples per class
transform = transforms.Compose([transforms.ToTensor()])
mnist_data = datasets.MNIST(root="./data", train=True, transform=transform, download=True)

class_counts = {i: 0 for i in range(10)}
subset = []
for img, label in mnist_data:
    if class_counts[label] < 100:
        subset.append((img, label))
        class_counts[label] += 1
    if all(count == 100 for count in class_counts.values()):
        break

# Save the dataset subset
torch.save(subset, "mnist_subset_100_per_class.pt")
```

---

## Running SKA

To execute the SKA training process:

```bash
python run_ska.py
```

### Expected Output:
- **Entropy Evolution**: Tracks layer-wise entropy reduction over **K=50** forward steps.
- **Cosine Alignment**: Measures the knowledge alignment metric across layers.
- **Output Probability Evolution**: Visualizes how decision probabilities refine over time.

After execution, the script generates the following plots:
- `entropy_history_single_pass.png`
- `cosine_history_single_pass.png`
- `output_distribution_single_pass.png`

---

## Understanding the SKA Model

### **Key Components**
1. **Knowledge Representation (`Z`)**:
   - Computed as \( Z = Wx + b \)
   - Encodes structured knowledge per neuron.
  
2. **Decision Probabilities (`D`)**:
   - Defined as \( D = \sigma(Z) \) (sigmoid transformation).
   - Represents the evolving decision space.

3. **Entropy Minimization**:
   - Entropy is redefined as:
     \[
     H^{(l)} = -\frac{1}{\ln 2} \sum_{k=1}^{K} Z_k \cdot \Delta D_k
     \]
   - Learning occurs by aligning **Z** with **decision shifts (ΔD)**.

4. **Learning Without Backpropagation**:
   - Weights are updated using:
     \[
     W \leftarrow W - \eta \frac{\partial H}{\partial W}
     \]
   - No gradients from backward error propagation.

---

## Directory Structure

```
SKA/
│── ska_model.py             # SKA model definition
│── run_ska.py               # Script to train and visualize results
│── mnist_subset_100_per_class.pt  # Pre-saved dataset
│── results/                 # Stores output images
│── README.md                # Project documentation
```

---

## Future Work

- Extending SKA to convolutional architectures.
- Applying SKA to real-time streaming datasets.
- Exploring its use in hardware-efficient AI models.



---

## License

This project is open-source under the **MIT License**.

---

**SKA is an exciting step toward self-organizing AI models! Try it today and contribute to the next revolution in AI.**
```

