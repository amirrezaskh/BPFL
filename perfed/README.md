# BPFL Core Framework

This directory contains the core federated learning algorithms and utilities that implement the BPFL (Blockchain-enabled Personalized Federated Learning) framework.

## Directory Structure

```
perfed/
├── __init__.py           # Package initialization
├── node.py              # Core FL client implementation
└── __pycache__/         # Python bytecode cache
```

## Overview

The `perfed` package provides the foundational components for federated learning clients, including data handling, local training, and blockchain integration. This modular design allows for easy experimentation across different datasets, architectures, and distribution schemes.

## Core Components

### 🏃 **Node Class** (`node.py`)

The main federated learning client implementation that handles all aspects of local training and participation in the distributed learning process.

#### **Key Features:**

1. **Multi-Dataset Support**
   - MNIST (28x28 grayscale, 10 classes)
   - Fashion-MNIST (28x28 grayscale, 10 classes)  
   - CIFAR-10 (32x32 RGB, 10 classes)
   - CIFAR-100 (32x32 RGB, 100 classes)

2. **Flexible Neural Architectures**
   - **LeNet-5**: Classic CNN for MNIST/Fashion-MNIST
   - **ResNet-18**: Modern residual network for CIFAR datasets

3. **Data Distribution Schemes**
   - **Uniform**: Equal data allocation across clients
   - **Linear**: Proportional allocation (1:2:3:4 ratio)
   - **Exponential**: Exponentially increasing allocation

4. **Privacy-Preserving Design**
   - Local data never leaves client devices
   - Only model parameters shared via blockchain
   - Secure aggregation through cryptographic protocols

#### **Class Architecture:**

```python
class Node:
    def __init__(self, num_nodes, port, peer_port=3000):
        # Initialize client with unique configuration
        
    def get_data(self):
        # Load and partition dataset based on allocation scheme
        
    def train(self, global_model_path):
        # Execute local training with personalized model
        
    def train_epoch(self, loss_fn, optimizer):
        # Single epoch of SGD training
        
    def validate_epoch(self, loss_fn):
        # Evaluate model on local validation set
```

#### **Neural Network Implementations:**

**ResNet-18 Classifier** (for CIFAR datasets):
```python
class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=False)
        # Adapt for 32x32 images
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  # Remove maxpool for small images
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
```

**LeNet-5** (for MNIST/Fashion-MNIST):
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

### 📊 **Data Management**

#### **Custom Dataset Wrapper:**
```python
class CustomDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    
    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
```

#### **Data Augmentation Strategies:**

**For CIFAR Datasets:**
```python
transforms = v2.Compose([
    v2.RandomHorizontalFlip(),           # 50% horizontal flip
    v2.RandomRotation(10),               # ±10° rotation
    v2.RandomAffine(0, shear=10, scale=(0.8, 1.2)),  # Geometric transforms
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Color variance
    v2.ToTensor()                        # Convert to tensor
])
```

**For MNIST/Fashion-MNIST:**
```python
transforms = v2.Compose([
    v2.RandomRotation(10),               # Slight rotation
    v2.RandomAffine(0, translate=(0.1, 0.1)),  # Small translations
    v2.ToTensor()                        # Convert to tensor
])
```

### ⚙️ **Training Configuration**

#### **Hyperparameters:**
```python
# Training Parameters
EPOCHS = 5                    # Local training epochs per round
BATCH_SIZE = 32              # Mini-batch size for SGD
LEARNING_RATE = 1e-3         # Base learning rate
WEIGHT_DECAY = 1e-4          # L2 regularization coefficient

# Client Configuration  
NUM_NODES = 4                # Number of federated participants
RANDOM_STATE = 1379          # Reproducibility seed
```

#### **Optimization Strategy:**
```python
# Adam optimizer with weight decay
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=LEARNING_RATE, 
    weight_decay=WEIGHT_DECAY
)

# Learning rate scheduling
scheduler = ReduceLROnPlateau(
    optimizer, 
    mode="min", 
    patience=EPOCHS//6
)

# Cross-entropy loss for classification
loss_fn = nn.CrossEntropyLoss()
```

## Data Distribution Schemes

### 🟢 **Uniform Distribution**
Equal data allocation across all clients for homogeneous setting.

```python
def get_data_uniform(self):
    data_portion = len(train_dataset) // self.num_nodes
    start_idx = self.index * data_portion
    end_idx = (self.index + 1) * data_portion
    indexes = list(range(start_idx, end_idx))
```

### 📈 **Linear Distribution** 
Proportional allocation creating moderate heterogeneity.

```python
def get_data_linear(self):
    total_portions = sum(range(1, self.num_nodes + 1))  # 1+2+3+4 = 10
    portion_size = len(train_dataset) // total_portions
    node_portion = (self.index + 1) * portion_size  # Client i gets i*portion_size
```

### 📊 **Exponential Distribution**
Exponentially increasing allocation for high heterogeneity.

```python
def get_data_exponential(self):
    base_portion = len(train_dataset) // (2**self.num_nodes - 1)
    node_portion = base_portion * (2**self.index)  # Client i gets base*(2^i)
```

## Training Pipeline

### 1️⃣ **Initialization Phase**
```python
node = Node(num_nodes=4, port=8000+client_id)
# - Load dataset and create data partitions
# - Initialize neural network architecture  
# - Setup optimization components
# - Create test dataset for contribution assessment
```

### 2️⃣ **Round Execution**
```python
def train(self, global_model_path):
    # Load personalized model from previous round
    self.model.load_state_dict(torch.load(global_model_path))
    
    # Local training loop
    for epoch in range(self.epochs):
        self.train_epoch(loss_fn, optimizer)
        val_loss = self.validate_epoch(loss_fn)
        scheduler.step(val_loss)
    
    # Save and submit trained model
    torch.save(self.model.state_dict(), self.model_path)
    self.submit_to_blockchain()
```

### 3️⃣ **Model Submission**
```python
# Submit trained model to blockchain network
requests.post("http://localhost:3000/api/model/", json={
    "id": f"model_{self.index}",
    "walletId": f"wallet_{self.index}", 
    "path": self.model_path,
    "testDataPath": self.tests_path
})
```

## Performance Optimizations

### 🚀 **GPU Acceleration**
```python
# Automatic device selection
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Memory management for large models
if device == "cuda":
    torch.cuda.empty_cache()  # Clear cache between rounds
```

### 📈 **Efficient Data Loading**
```python
# Optimized DataLoader configuration
train_dataloader = DataLoader(
    dataset, 
    batch_size=batch_size,
    shuffle=True,           # Randomize training order
    num_workers=2,          # Parallel data loading
    pin_memory=True         # Faster GPU transfer
)
```

### 🎯 **Adaptive Batch Sizing**
```python
# Dynamic batch size based on available memory
if torch.cuda.is_available():
    batch_size = min(64, len(dataset) // 10)  # Larger batches for GPU
else:
    batch_size = 32  # Conservative for CPU
```

## Integration with BPFL Framework

### 🔄 **Blockchain Communication**
The node communicates with the blockchain network through RESTful APIs:

```python
# Model submission endpoint
POST /api/model/
{
    "id": "model_0",
    "walletId": "wallet_0", 
    "path": "/path/to/model.pt",
    "testDataPath": "/path/to/tests.pt"
}

# Round coordination endpoint  
POST /round/
{
    "modelPath": "/path/to/global_model.pt"
}
```

### 💰 **Incentive Integration**
- **Contribution Tracking**: Models evaluated for quality improvement
- **Reward Calculation**: Tokens distributed based on performance gains
- **Reputation System**: Historical performance influences future rewards

### 🔒 **Privacy Guarantees**
- **Local Data**: Raw datasets never leave client devices
- **Model Sharing**: Only aggregated parameters transmitted
- **Differential Privacy**: Optional noise injection for enhanced privacy

## Experimental Variations

Different branches implement variations by modifying key components:

### **Dataset Switching:**
```python
# MNIST Branch
train_dataset = datasets.MNIST(root="./data", train=True, download=True)
model = LeNet5(num_classes=10)

# CIFAR-100 Branch  
train_dataset = datasets.CIFAR100(root="./data", train=True, download=True)
model = ResNet18Classifier(num_classes=100)
```

### **Architecture Changes:**
```python
# LeNet for small images (28x28)
if dataset in ['MNIST', 'FashionMNIST']:
    model = LeNet5(num_classes=num_classes)
    
# ResNet for larger images (32x32)
elif dataset in ['CIFAR10', 'CIFAR100']:
    model = ResNet18Classifier(num_classes=num_classes)
```

### **Distribution Modifications:**
```python
# Implemented in get_data() method based on branch
if branch.endswith('Uniform'):
    self.get_data_uniform()
elif branch.endswith('Linear'):
    self.get_data_linear()  
elif branch.endswith('Exponential'):
    self.get_data_exponential()
```

## Troubleshooting

### **Common Issues:**

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size or switch to CPU
   batch_size = 16  # Reduce from 32
   device = "cpu"   # Force CPU usage
   ```

2. **Dataset Download Failures**
   ```python
   # Manual dataset download
   datasets.CIFAR10(root="./data", download=True)
   ```

3. **Port Conflicts**
   ```bash
   # Check port availability
   lsof -i :8000
   netstat -an | grep 8000
   ```

4. **Model Loading Errors**
   ```python
   # Verify model compatibility
   checkpoint = torch.load(path, map_location=device)
   model.load_state_dict(checkpoint, strict=False)
   ```

---

The `perfed` package provides a robust, flexible foundation for federated learning experimentation with seamless blockchain integration and comprehensive support for diverse experimental configurations.
