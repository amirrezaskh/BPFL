# CIFAR10-ResNet18-Exponential-FL Experiment

This branch implements **CIFAR10** dataset training with **ResNet18-5** architecture using **Exponential (1:2:4:8 ratio across clients)** in the BPFL framework.

## 🧪 Experiment Configuration

| Parameter | Value |
|-----------|-------|
| **Dataset** | CIFAR-10 (32x32 RGB, 10 classes - natural images) |
| **Architecture** | ResNet18-5 CNN |
| **Data Distribution** | Exponential (1:2:4:8 ratio across clients) |
| **Algorithm** | Traditional Federated Learning |
| **Gamma (γ)** | N/A (Traditional FL) |
| **Clients** | 4 federated participants |
| **Rounds** | 20 training rounds |
| **Local Epochs** | 5 per round |

## 📊 Key Characteristics

### **Dataset Classes:**
Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

### **Data Distribution (Exponential):**
- **Client 0**: ~6.7% of data
- **Client 1**: ~13.3% of data
- **Client 2**: ~26.7% of data
- **Client 3**: ~53.3% of data

This creates **high heterogeneity** to test robustness under extreme imbalance.

### **Architecture Details:**
```python
class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  # Removed for small images
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
```
**Optimized for**: 32×32 RGB images (CIFAR-10, CIFAR-100)

### **Algorithm Configuration:**
- **Type**: Traditional Federated Learning (FedAvg)
- **Personalization**: None (pure global model)
- **Aggregation**: Simple weighted averaging
- **Incentives**: None (baseline comparison)

## 🚀 Running This Experiment

```bash
# Ensure you're on the correct branch
git checkout CIFAR10-ResNet18-Exponential-FL

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
- **Natural image classification** with RGB complexity
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
- `CIFAR10-ResNet18-Uniform` - Same setup with uniform distribution
- `CIFAR10-ResNet18-Linear` - Same setup with linear distribution
- `CIFAR10-ResNet18-Exponential` - Same setup with exponential distribution
- `CIFAR10-ResNet18-Exponential-0.95` - Higher personalization (γ=0.95)

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
- **Purpose**: CIFAR10 + ResNet18 + Exponential Analysis
- **Experiment ID**: cifar10_resnet18_exponential_fl

---

*This is one of 38 experimental configurations in the BPFL research framework. Each branch represents a unique combination of dataset, architecture, and data distribution for comprehensive federated learning analysis.*