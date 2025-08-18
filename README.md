# MNIST-LeNet-Exponential-FL Experiment

This branch implements **MNIST** dataset training with **LeNet-5** architecture using **Exponential (1:2:4:8 ratio across clients)** in the BPFL framework.

## 🧪 Experiment Configuration

| Parameter | Value |
|-----------|-------|
| **Dataset** | MNIST (28x28 grayscale, 10 classes - handwritten digits) |
| **Architecture** | LeNet-5 CNN |
| **Data Distribution** | Exponential (1:2:4:8 ratio across clients) |
| **Algorithm** | Traditional Federated Learning |
| **Gamma (γ)** | N/A (Traditional FL) |
| **Clients** | 4 federated participants |
| **Rounds** | 20 training rounds |
| **Local Epochs** | 5 per round |

## 📊 Key Characteristics

### **Dataset Classes:**
Digits 0-9

### **Data Distribution (Exponential):**
- **Client 0**: ~6.7% of data
- **Client 1**: ~13.3% of data
- **Client 2**: ~26.7% of data
- **Client 3**: ~53.3% of data

This creates **high heterogeneity** to test robustness under extreme imbalance.

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
- **Type**: Traditional Federated Learning (FedAvg)
- **Personalization**: None (pure global model)
- **Aggregation**: Simple weighted averaging
- **Incentives**: None (baseline comparison)

## 🚀 Running This Experiment

```bash
# Ensure you're on the correct branch
git checkout MNIST-LeNet-Exponential-FL

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
- **Extreme heterogeneity** and personalization benefits
- **Traditional FL baseline** performance without personalization

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
- `MNIST-LeNet-Exponential-0.95` - Higher personalization (γ=0.95)

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
- **Purpose**: MNIST + LeNet + Exponential Analysis
- **Experiment ID**: mnist_lenet_exponential_fl

---

*This is one of 38 experimental configurations in the BPFL research framework. Each branch represents a unique combination of dataset, architecture, and data distribution for comprehensive federated learning analysis.*