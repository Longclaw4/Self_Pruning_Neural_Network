# 🧠 The Self-Pruning Neural Network

**Case Study — Tredence Analytics AI Engineering Internship**

A neural network that **learns to prune itself during training** by using learnable per-weight gates and L1 sparsity regularization. Built on CIFAR-10 with PyTorch.

---

## 📌 Overview

Traditional pruning removes unimportant weights **after** training. This project implements a network that prunes itself **during** training by:

1. **Gated Weights** — Each weight has a learnable gate (sigmoid of a score). Gates near 0 effectively remove the weight.
2. **L1 Regularization** — A sparsity penalty on gate values drives unimportant gates to zero.
3. **λ Trade-off** — A hyperparameter controls pruning aggressiveness vs. accuracy.

## 🏗️ Architecture

```
Input (3×32×32 = 3072) → Flatten
  → PrunableLinear(3072, 1024) → BatchNorm → ReLU
  → PrunableLinear(1024, 512)  → BatchNorm → ReLU
  → PrunableLinear(512, 256)   → BatchNorm → ReLU
  → PrunableLinear(256, 10)    → Output (logits)
```

### Loss Function

```
Total Loss = CrossEntropyLoss + λ × Σ sigmoid(gate_scores)
```

### Training Strategy

- **Warmup**: First 10 epochs with λ=0 (pure classification — network learns which weights matter)
- **Pruning Phase**: λ ramps linearly from 0 to target over remaining 30 epochs
- **Separate LR**: Gate parameters use 10× higher learning rate for effective pruning dynamics

## 📊 Results

| Lambda (λ) | Test Accuracy | Sparsity Level |
|:----------:|:------------:|:--------------:|
| 1e-5 (Low) | **63.91%** | 51.15% |
| 1e-4 (Medium) | **63.53%** | 88.98% |
| 5e-4 (High) | **63.09%** | **97.54%** |

> **Key Finding:** Only 0.82% accuracy drop while removing 97.5% of all weights!

See the full analysis in [REPORT.md](REPORT.md).

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. Open `Self_Pruning_Neural_Network.ipynb` in Google Colab
2. Set runtime to **T4 GPU** (Runtime → Change runtime type → T4)
3. Run all cells (~48 min total for 3 experiments)

### Option 2: Local Execution

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/tredence-self-pruning-network.git
cd tredence-self-pruning-network

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook Self_Pruning_Neural_Network.ipynb
```

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- matplotlib
- numpy
- CUDA-compatible GPU (recommended)

See [requirements.txt](requirements.txt) for exact versions.

## 📁 Repository Structure

```
├── Self_Pruning_Neural_Network.ipynb   # Main Colab notebook (single script)
├── REPORT.md                           # Analysis report with plots
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
└── .gitignore                          # Git ignore rules
```

## 🔑 Key Concepts

- **PrunableLinear Layer**: Custom `nn.Module` with `weight`, `bias`, and `gate_scores` parameters
- **Gate Mechanism**: `gates = sigmoid(gate_scores)` → `pruned_weights = weight × gates`
- **Sparsity Loss**: L1 norm (sum) of all gate values across all layers
- **λ Warmup**: Train normally first, then gradually increase pruning pressure
- **Bimodal Distribution**: Successful pruning shows gates clustered at 0 (pruned) and away from 0 (surviving)

## 📄 License

This project is submitted as part of the Tredence Analytics AI Engineering Internship case study.
