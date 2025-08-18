# Federated Learning Nodes

This directory contains the core federated learning implementation including client nodes, aggregator, and neural network models for the BPFL framework.

## Directory Structure

```
nodes/
├── aggregator.py          # Central aggregation server
├── model.py              # Neural network architectures  
├── node[0-3].py         # Individual client nodes
├── models/              # Saved model states
├── results/             # Training metrics and results
├── tests/               # Test datasets for each node
└── __pycache__/         # Python bytecode cache
```

## Components Overview

### 🔄 **Aggregator** (`aggregator.py`)

The central coordination server that implements the BPFL aggregation algorithm.

**Key Features:**
- **Contribution Assessment**: Evaluates client model quality using loss-based metrics
- **Personalized Aggregation**: Combines models using adaptive gamma weighting
- **Price Dynamics**: Updates global model pricing based on performance improvements  
- **Reward Distribution**: Calculates tokens based on contribution logarithmic smoothing

**Core Methods:**
```python
def compute_contribution(submits)     # Assess client contributions
def update_global_model(submits)      # Weighted federated averaging
def update_personalized_models(submits) # Apply gamma-weighted personalization
def compute_rewards(submits, total)   # Distribute incentive tokens
```

**Configuration Parameters:**
- `gamma_max`: Personalization strength (0.7 or 0.95)
- `contribution_factor`: Balance between gap and local improvement (0.5)
- `p`: Contribution scaling exponent (0.5)

### 🧠 **Neural Network Models** (`model.py`)

Defines the neural architectures used across different experimental configurations.

**Available Architectures:**

1. **ResNet-18 Classifier** (CIFAR-10/100)
   ```python
   class ResNet18Classifier(nn.Module):
       # Modified ResNet-18 for 32x32 images
       # Removed max pooling for small image compatibility
       # Configurable output classes (10 or 100)
   ```

2. **LeNet-5** (MNIST/Fashion-MNIST)
   ```python
   class LeNet5(nn.Module):  
       # Classic CNN architecture
       # Optimized for 28x28 grayscale images
       # Tanh activations as in original paper
   ```

**Model Initialization:**
```bash
python model.py  # Creates initial global.pt model
```

### 👥 **Client Nodes** (`node[0-3].py`)

Individual federated learning participants with unique data partitions.

**Node Responsibilities:**
- **Local Training**: Execute SGD on private datasets
- **Model Submission**: Upload trained weights to blockchain
- **Privacy Preservation**: Never share raw training data
- **HTTP Interface**: RESTful endpoints for round coordination

**API Endpoints:**
```
POST /round/    # Start training round with global model
GET  /exit/     # Graceful shutdown
```

**Data Distribution:**
- **Node 0**: Port 8000, wallet_0, model_0
- **Node 1**: Port 8001, wallet_1, model_1  
- **Node 2**: Port 8002, wallet_2, model_2
- **Node 3**: Port 8003, wallet_3, model_3

## Training Pipeline

### 1. **Round Initialization**
```python
# Aggregator receives global model from blockchain
# Nodes download personalized models for local training
```

### 2. **Local Training Phase**
```python
# Each node trains for EPOCHS=5 on local data
# Applies data augmentation and regularization
# Saves model state to models/model_{i}.pt
```

### 3. **Model Submission**
```python
# Nodes upload trained models to blockchain
# Includes model path and test data for evaluation
# Triggers aggregation when all submissions received
```

### 4. **Aggregation & Reward**
```python
# Aggregator computes contributions and rewards
# Updates global and personalized models
# Distributes tokens based on performance
```

## Data Management

### **Dataset Handling**
- **Automatic Download**: Datasets downloaded to `./data/` on first run
- **Partitioning**: Different allocation schemes across experimental branches
- **Augmentation**: Real-time transforms during training
- **Test Sets**: Separate evaluation data per node

### **Data Distribution Schemes**

**Uniform Distribution:**
```python
data_portion = len(dataset) // num_nodes  # Equal splits
```

**Linear Distribution:**
```python
portions = [i * base_size for i in range(1, num_nodes+1)]  # 1:2:3:4 ratio
```

**Exponential Distribution:**
```python
portions = [base_size * (2**i) for i in range(num_nodes)]  # Exponential growth
```

## Results Management

### **Metrics Collection** (`results/res.json`)
```json
{
  "round": 1,
  "new_model_price": 55.2,
  "g_model_loss": 1.234,
  "submits": [
    {
      "walletId": "wallet_0",
      "loss": 1.456,
      "delta_local_loss": 0.123,
      "delta_gap": 0.234,  
      "contribution": 0.345,
      "reward": 75.5
    }
  ]
}
```

### **Key Metrics:**
- **Global Loss**: Performance of aggregated model
- **Local Losses**: Individual client model performance  
- **Delta Gap**: Improvement over previous global model
- **Delta Local**: Local improvement since last round
- **Contributions**: Weighted combination of improvements
- **Rewards**: Token distribution based on contributions

## Configuration

### **Training Hyperparameters**
```python
EPOCHS = 5           # Local training rounds
BATCH_SIZE = 32      # SGD batch size
LEARNING_RATE = 1e-3 # Base learning rate
WEIGHT_DECAY = 1e-4  # L2 regularization
```

### **BPFL Parameters**
```python
GAMMA_MAX = 0.7           # Personalization strength
CONTRIBUTION_FACTOR = 0.5  # Gap vs local improvement weight
P = 0.5                   # Contribution scaling exponent
```

### **Network Configuration**
```python
NUM_NODES = 4        # Number of federated clients
ROUNDS = 20          # Total training rounds
AGGREGATOR_PORT = 8080   # Central server port
```

## Usage Examples

### **Start Training Process**
```bash
cd nodes/
python aggregator.py &    # Start aggregation server
python node0.py &         # Start client nodes
python node1.py &
python node2.py &  
python node3.py &
```

### **Monitor Training**
```bash
tail -f ../logs/aggregator.txt  # View aggregation logs
tail -f ../logs/node_0.txt      # View client logs
```

### **Evaluate Results**
```bash
python ../plot.py        # Generate visualizations
cat results/res.json     # View detailed metrics
```

## Branch Variations

Different experimental branches modify key components:

### **Dataset/Architecture Changes**
- `model.py`: Switch between LeNet5 and ResNet18
- `perfed/node.py`: Change dataset loading (MNIST/FMNIST/CIFAR10/CIFAR100)

### **Distribution Scheme Changes**  
- `perfed/node.py`: Modify `get_data()` method for different allocation patterns

### **Algorithm Changes**
- `aggregator.py`: Adjust `gamma_max` for personalization strength
- Traditional FL: Remove personalization in `update_personalized_models()`

## Troubleshooting

### **Common Issues**

1. **Port Conflicts**
   ```bash
   lsof -i :8080  # Check if aggregator port in use
   ```

2. **Model Loading Errors**
   ```bash
   ls models/  # Verify model files exist
   ```

3. **Memory Issues**
   ```python
   # Reduce batch size or use CPU training
   device = "cpu"  # Force CPU usage
   ```

4. **Network Connectivity**
   ```bash
   curl http://localhost:8080/  # Test aggregator endpoint
   ```

## Performance Optimization

### **GPU Acceleration**
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

### **Batch Size Tuning**
```python
# Larger batches for better GPU utilization
batch_size = 64 if device == "cuda" else 32
```

### **Memory Management**
```python
# Clear cache between rounds
torch.cuda.empty_cache()
```

---

This directory implements the core federated learning algorithms with blockchain integration, enabling decentralized, incentivized machine learning across multiple participants while preserving data privacy.
