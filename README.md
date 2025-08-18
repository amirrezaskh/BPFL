# FMNIST-LeNet-Uniform-0.95 Experiment

This branch implements **FMNIST** dataset training with **LeNet-5** architecture using **Uniform (Equal data allocation across clients)** in the BPFL framework.

## 🧪 Experiment Configuration

| Parameter | Value |
|-----------|-------|
| **Dataset** | Fashion-MNIST (28x28 grayscale, 10 classes - clothing items) |
| **Architecture** | LeNet-5 CNN |
| **Data Distribution** | Uniform (Equal data allocation across clients) |
| **Algorithm** | BPFL with High Personalization |
| **Gamma (γ)** | 0.95 |
| **Clients** | 4 federated participants |
| **Rounds** | 20 training rounds |
| **Local Epochs** | 5 per round |

## 📊 Key Characteristics

### **Dataset Classes:**
T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot

### **Data Distribution (Uniform):**
- **Client 0**: 25% of data
- **Client 1**: 25% of data
- **Client 2**: 25% of data
- **Client 3**: 25% of data

This creates **homogeneous** conditions for baseline comparison.

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
git checkout FMNIST-LeNet-Uniform-0.95

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
- **Fashion item classification** with increased visual complexity
- **Homogeneous learning** with equal data distribution
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
- `FMNIST-LeNet-Uniform` - Same setup with uniform distribution
- `FMNIST-LeNet-Linear` - Same setup with linear distribution
- `FMNIST-LeNet-Exponential` - Same setup with exponential distribution
- `FMNIST-LeNet-Uniform-FL` - Traditional FL baseline

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
- **Purpose**: FMNIST + LeNet + Uniform Analysis
- **Experiment ID**: fmnist_lenet_uniform_0.95

---

*This is one of 38 experimental configurations in the BPFL research framework. Each branch represents a unique combination of dataset, architecture, and data distribution for comprehensive federated learning analysis.*