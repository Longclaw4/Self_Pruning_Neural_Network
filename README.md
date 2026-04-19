# 🧠 The Self-Pruning Neural Network — CIFAR-10

A research-oriented implementation of a neural network that identifies and removes its own redundant connections during training, achieving massive compression (**97.5%**) with negligible accuracy loss.

---

## 📘 1. Mathematical Intuition

Exact sparsity in this model is achieved through an **L1-Sigmoid gating mechanism**, which is effective for three primary reasons:

1. **Constant Gradient Pressure**: Unlike L2 regularization (weight decay), the **L1 norm** ($\|g\|_1 = \sum |g_i|$) maintains a constant gradient pressure. This ensures that parameters are pushed all the way to **precisely zero** rather than just becoming very small.
2. **Sigmoid Saturation**: The sigmoid function $\sigma(s)$ acts as a learnable "soft switch." By penalizing the *output* of the sigmoid, the optimizer is forced to drive the *input score* $s$ toward $-\infty$. This creates a clean bimodal separation where weights are either fully active or fully pruned.
3. **Dynamic Architecture Search**: The network effectively "searches" for its minimal necessary sub-architecture during the training process, preventing the need for manual post-hoc pruning or heuristic weight removal.

## 🏗️ 2. Core Architecture

The system is built around a self-contained modular pipeline:

*   **PrunableLinear Module**: A custom `nn.Module` that maintains standard `weight` parameters alongside learnable `gate_scores`. It calculates the effective weights as: $W_{eff} = W \times \sigma(gate\_scores)$.
*   **Network Flow**: A 4-layer MLP (3072 → 1024 → 512 → 256 → 10) utilizing BatchNorm and ReLU activations at each hidden stage to maintain stable gradient flow.
*   **Dual-Objective Loss**:
    $$Total\_Loss = CrossEntropy + \lambda \times \sum (\sigma(gate\_scores))$$
    where $\lambda$ (Lambda) serves as the primary control for sparsity aggressiveness.

## 📊 3. Performance Results

The model was evaluated across multiple sparsity regimes to identify the optimal Pareto frontier between model size and classification power.

| Sparsity Pressure ($\lambda$) | Test Accuracy | Sparsity Level | Interpretation |
|:----------:|:------------:|:--------------:|:---|
| Low (1e-5) | 63.91% | 51.15% | Feature Discovery Phase |
| Medium (1e-4) | 63.53% | 88.98% | **Optimal Efficiency** |
| High (5e-4) | 63.09% | **97.54%** | Extreme Compression |

### Bimodal Gate Distribution
The success of the self-pruning mechanism is confirmed by the bimodal distribution of gate values, showing a distinct separation between pruned weights and critical surviving connections.

![Gate Distribution](gate_distribution.png)

## 🛠️ 4. System Highlights
- **Layer-wise Adaptivity**: Pruning is most aggressive in earlier, wide layers (e.g., fc1), while the output layer remains dense to preserve decision boundaries.
- **$\lambda$-Warmup Strategy**: The network utilizes a warmup phase (10 epochs of λ=0) to ensure high-quality feature extraction before sparsity pressure is introduced.
- **Optimization Strategy**: Separate learning rates are used for weights ($1e^{-3}$) and gates ($1e^{-2}$) to ensure pruning dynamics converge at the correct scale.

---

## 🚀 Execution Guide

1. **Standalone Script**: Run `python main.py` for a complete training, evaluation, and logging cycle.
2. **Interactive Notebook**: Open `Self_Pruning_Neural_Network.ipynb` for detailed visualizations, training curves, and layer-wise breakdown.

```bash
# Setup
git clone https://github.com/Longclaw4/Self_Pruning_Neural_Network.git
pip install -r requirements.txt

# Run
python main.py
```

## 📁 Project Structure

```
├── main.py                             # Self-contained training script
├── Self_Pruning_Neural_Network.ipynb   # Interactive experimental suite
├── REPORT.md                           # Detailed technical analysis
├── README.md                           # Project overview
├── requirements.txt                    # Dependencies
└── results/                            # Metrics and visualizations
```
