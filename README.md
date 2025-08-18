# CIFAR100-ResNet18-Linear-FL Experiment

This branch implements **CIFAR100** dataset training with **ResNet18-5** architecture using **Linear (1:2:3:4 ratio across clients)** in the BPFL framework.

## 🧪 Experiment Configuration

| Parameter | Value |
|-----------|-------|
| **Dataset** | CIFAR-100 (32x32 RGB, 100 classes - natural images) |
| **Architecture** | ResNet18-5 CNN |
| **Data Distribution** | Linear (1:2:3:4 ratio across clients) |
| **Algorithm** | Traditional Federated Learning |
| **Gamma (γ)** | N/A (Traditional FL) |
| **Clients** | 4 federated participants |
| **Rounds** | 20 training rounds |
| **Local Epochs** | 5 per round |

## 📊 Key Characteristics

### **Dataset Classes:**
100 fine-grained categories across 20 superclasses

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
- **Type**: Traditional Federated Learning (FedAvg)
- **Personalization**: None (pure global model)
- **Aggregation**: Simple weighted averaging
- **Incentives**: None (baseline comparison)

## 🚀 Running This Experiment

```bash
# Ensure you're on the correct branch
git checkout CIFAR100-ResNet18-Linear-FL

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
- **Fine-grained classification** with 100 diverse categories
- **Moderate heterogeneity** effects on convergence
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
- `CIFAR100-ResNet18-Uniform` - Same setup with uniform distribution
- `CIFAR100-ResNet18-Linear` - Same setup with linear distribution
- `CIFAR100-ResNet18-Exponential` - Same setup with exponential distribution
- `CIFAR100-ResNet18-Linear-0.95` - Higher personalization (γ=0.95)

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
- **Purpose**: CIFAR100 + ResNet18 + Linear Analysis
- **Experiment ID**: cifar100_resnet18_linear_fl

---

*This is one of 38 experimental configurations in the BPFL research framework. Each branch represents a unique combination of dataset, architecture, and data distribution for comprehensive federated learning analysis.*