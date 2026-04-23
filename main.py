import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time

# ══════════════════════════════════════════════════════════════
#  1. PRUNABLE LINEAR LAYER
# ══════════════════════════════════════════════════════════════
class PrunableLinear(nn.Module):
    """
    A Linear layer that learns to prune its own weights using a gating mechanism.
    Weights are multiplied by sigmoid(gate_scores). L1 penalty on gates drives sparsity.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard weights and bias
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Gating parameters: initialized to 0.0 so sigmoid(0) = 0.5 (maximum gradient flow)
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming initialization for weights
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        # Initialize gate scores to 0 (activates ~50% of weight initially)
        nn.init.constant_(self.gate_scores, 0.0)

    def forward(self, x):
        # Apply sigmoid to gate scores to get values in (0, 1)
        gates = torch.sigmoid(self.gate_scores)
        # Multiply weight by gates (element-wise) before linear operation
        pruned_weight = self.weight * gates
        # Perform the linear operation from scratch: y = xW^T + b
        return x @ pruned_weight.t() + self.bias

    def get_gate_values(self):
        return torch.sigmoid(self.gate_scores).detach()

# ══════════════════════════════════════════════════════════════
#  2. NEURAL NETWORK DEFINITION
# ══════════════════════════════════════════════════════════════
class SelfPruningNetwork(nn.Module):
    """
    MLP for CIFAR-10 using PrunableLinear layers.
    Architecture: 3072 -> 1024 -> 512 -> 256 -> 10
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            PrunableLinear(3072, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            PrunableLinear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            PrunableLinear(256, 10)
        )

    def forward(self, x):
        return self.layers(x)

    def compute_sparsity_loss(self):
        """Calculates the sum of all gate values across all prunable layers."""
        loss = 0.0
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                loss += torch.sigmoid(module.gate_scores).sum()
        return loss

    def get_overall_sparsity(self):
        """Returns the percentage of weights that are effectively pruned (gate < 0.01)."""
        total_weights = 0
        pruned_weights = 0
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates = module.get_gate_values()
                total_weights += gates.numel()
                pruned_weights += (gates < 0.01).sum().item()
        return (pruned_weights / total_weights) * 100 if total_weights > 0 else 0

# ══════════════════════════════════════════════════════════════
#  3. TRAINING AND EVALUATION LOGIC
# ══════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, criterion, lam, device):
    model.train()
    total_loss, ce_loss, sparse_loss = 0, 0, 0
    correct, total = 0, 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Combined Loss: CrossEntropy + Lambda * Sparsity Penalty
        l_ce = criterion(outputs, labels)
        l_sparse = model.compute_sparsity_loss()
        loss = l_ce + lam * l_sparse
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        ce_loss += l_ce.item()
        sparse_loss += l_sparse.item()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return {
        'avg_ce_loss': ce_loss / len(loader),
        'avg_sparsity_loss': sparse_loss / len(loader),
        'accuracy': 100 * correct / total,
        'sparsity': model.get_overall_sparsity()
    }

def evaluate(model, loader, criterion, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return {'test_accuracy': 100 * correct / total}

# ══════════════════════════════════════════════════════════════
#  4. MAIN EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data setup
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)

    # Hyperparameters
    LAMBDA_VALUES = [1e-5, 1e-4, 5e-4]
    EPOCHS = 40
    WARMUP_EPOCHS = 10

    results_summary = []

    for lam in LAMBDA_VALUES:
        print(f"\nExperiment: Lambda = {lam}")
        model = SelfPruningNetwork().to(device)
        criterion = nn.CrossEntropyLoss()
        
        # Separate learning rates: gates need faster movement than weights
        gate_params = [p for n, p in model.named_parameters() if 'gate_scores' in n]
        other_params = [p for n, p in model.named_parameters() if 'gate_scores' not in n]
        
        optimizer = optim.Adam([
            {'params': other_params, 'lr': 1e-3},
            {'params': gate_params, 'lr': 1e-2}
        ])

        for epoch in range(1, EPOCHS + 1):
            # Lambda Warmup: Start at 0, ramp up after epoch 10
            current_lam = 0 if epoch <= WARMUP_EPOCHS else lam * ((epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS))
            
            metrics = train_one_epoch(model, train_loader, optimizer, criterion, current_lam, device)
            
            if epoch % 10 == 0 or epoch == 1:
                test_metrics = evaluate(model, test_loader, criterion, device)
                print(f"Epoch {epoch:2d} | CE Loss: {metrics['avg_ce_loss']:.4f} | Test Acc: {test_metrics['test_accuracy']:.2f}% | Sparsity: {metrics['sparsity']:.2f}%")

        test_final = evaluate(model, test_loader, criterion, device)
        sparsity_final = model.get_overall_sparsity()
        results_summary.append((lam, test_final['test_accuracy'], sparsity_final))

    print("\n" + "="*50)
    print(f"{'Lambda':<12} | {'Test Accuracy':<15} | {'Sparsity (%)':<15}")
    print("-"*50)
    for lam, acc, sp in results_summary:
        print(f"{lam:<12.0e} | {acc:<15.2f} | {sp:<15.2f}")
    print("="*50)
