# MNIST-LeNet-Linear-0.95 Experiment

This branch implements **MNIST** dataset training with **LeNet-5** architecture using **Linear (1:2:3:4 ratio across clients)** in the BPFL framework.

## 🧪 Experiment Configuration

| Parameter | Value |
|-----------|-------|
| **Dataset** | MNIST (28x28 grayscale, 10 classes - handwritten digits) |
| **Architecture** | LeNet-5 CNN |
| **Data Distribution** | Linear (1:2:3:4 ratio across clients) |
| **Algorithm** | BPFL with High Personalization |
| **Gamma (γ)** | 0.95 |
| **Clients** | 4 federated participants |
| **Rounds** | 20 training rounds |
| **Local Epochs** | 5 per round |

## 📊 Key Characteristics

### **Dataset Classes:**
Digits 0-9

### **Data Distribution (Linear):**
- **Client 0**: 10% of data
- **Client 1**: 20% of data
- **Client 2**: 30% of data
- **Client 3**: 40% of data

This creates **moderate heterogeneity** for intermediate analysis.

### **Architecture Details:**
```python
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.ap = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.tanh = nn.Tanh()  # Original LeNet activation
```
**Optimized for**: 28×28 grayscale images (MNIST, Fashion-MNIST)

### **Algorithm Configuration:**
- **Type**: Blockchain-enabled Personalized Federated Learning (BPFL)
- **Personalization Strength (γ)**: 0.95
- **Contribution Assessment**: Loss improvement based
- **Reward Mechanism**: Token-based incentives
- **Aggregation**: Contribution-weighted with personalization

## 🚀 Running This Experiment

```bash
# Ensure you're on the correct branch
git checkout MNIST-LeNet-Linear-0.95

# Start the BPFL system
python run.py

# Monitor training progress
tail -f logs/aggregator.txt

# Generate results visualization
python plot.py

# Stop the system
python stop.py
```

## 📈 Expected Results

This configuration tests:
- **Digit recognition performance** on classic ML benchmark
- **Moderate heterogeneity** effects on convergence
- **BPFL personalization** advantages over traditional FL

## 📊 Results Analysis

After completion, check:
- `nodes/results/res.json` - Detailed metrics per round
- `figures/` - Generated visualization plots  
- `logs/` - Training logs and system monitoring

Key metrics to analyze:
- Global model convergence rate
- Individual client performance improvements
- Contribution score fairness
- Token reward distribution effectiveness

## 🔗 Related Experiments

**Compare with:**
- `MNIST-LeNet-Uniform` - Same setup with uniform distribution
- `MNIST-LeNet-Linear` - Same setup with linear distribution
- `MNIST-LeNet-Exponential` - Same setup with exponential distribution
- `MNIST-LeNet-Linear-FL` - Traditional FL baseline

## 📚 Full Documentation

For comprehensive documentation, system architecture, and methodology details, see the **main branch**:

```bash
git checkout main
cat README.md  # Complete BPFL framework documentation
```

## 🏷️ Branch Metadata

- **Branch Type**: Experimental Configuration
- **Created**: 2024 BPFL Research Study  
- **Status**: ✅ Results Available
- **Purpose**: MNIST + LeNet + Linear Analysis
- **Experiment ID**: mnist_lenet_linear_0.95

---

*This is one of 38 experimental configurations in the BPFL research framework. Each branch represents a unique combination of dataset, architecture, and data distribution for comprehensive federated learning analysis.*