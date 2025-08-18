# CIFAR10-ResNet18-Linear Experiment

This branch implements **CIFAR10** dataset training with **ResNet18-5** architecture using **Linear (1:2:3:4 ratio across clients)** in the BPFL framework.

## 🧪 Experiment Configuration

| Parameter | Value |
|-----------|-------|
| **Dataset** | CIFAR-10 (32x32 RGB, 10 classes - natural images) |
| **Architecture** | ResNet18-5 CNN |
| **Data Distribution** | Linear (1:2:3:4 ratio across clients) |
| **Algorithm** | BPFL Standard |
| **Gamma (γ)** | 0.7 |
| **Clients** | 4 federated participants |
| **Rounds** | 20 training rounds |
| **Local Epochs** | 5 per round |

## 📊 Key Characteristics

### **Dataset Classes:**
Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

### **Data Distribution (Linear):**
- **Client 0**: 10% of data
- **Client 1**: 20% of data
- **Client 2**: 30% of data
- **Client 3**: 40% of data

This creates **moderate heterogeneity** for intermediate analysis.

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
- **Type**: Blockchain-enabled Personalized Federated Learning (BPFL)
- **Personalization Strength (γ)**: 0.7
- **Contribution Assessment**: Loss improvement based
- **Reward Mechanism**: Token-based incentives
- **Aggregation**: Contribution-weighted with personalization

## 🚀 Running This Experiment

```bash
# Ensure you're on the correct branch
git checkout CIFAR10-ResNet18-Linear

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
- `CIFAR10-ResNet18-Uniform` - Same setup with uniform distribution
- `CIFAR10-ResNet18-Linear` - Same setup with linear distribution
- `CIFAR10-ResNet18-Exponential` - Same setup with exponential distribution
- `CIFAR10-ResNet18-Linear-FL` - Traditional FL baseline
- `CIFAR10-ResNet18-Linear-0.95` - Higher personalization (γ=0.95)

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
- **Purpose**: CIFAR10 + ResNet18 + Linear Analysis
- **Experiment ID**: cifar10_resnet18_linear

---

*This is one of 38 experimental configurations in the BPFL research framework. Each branch represents a unique combination of dataset, architecture, and data distribution for comprehensive federated learning analysis.*