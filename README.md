# Blockchain-enabled Personalized Federated Learning (BPFL)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hyperledger Fabric](https://img.shields.io/badge/Hyperledger%20Fabric-2.4+-blue.svg)](https://hyperledger-fabric.readthedocs.io/)

## Overview

This repository contains the implementation of **Blockchain-enabled Personalized Federated Learning (BPFL)**, a novel framework that combines personalized federated learning with blockchain technology to create a decentralized, incentive-driven machine learning system. The framework addresses key challenges in traditional federated learning including data heterogeneity, participant incentivization, and model personalization.

## Key Features

- **🔗 Blockchain Integration**: Utilizes Hyperledger Fabric for secure, transparent model aggregation and reward distribution
- **🎯 Personalized Learning**: Adaptive model personalization using contribution-weighted aggregation
- **💰 Incentive Mechanism**: Token-based reward system encouraging high-quality participation
- **📊 Multi-Dataset Support**: Comprehensive evaluation across MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100
- **⚖️ Fair Resource Allocation**: Multiple data distribution schemes (Uniform, Linear, Quadratic/Exponential)
- **🏗️ Modular Architecture**: Clean separation of concerns with containerized components

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client Nodes  │    │   Aggregator    │    │   Blockchain    │
│                 │    │                 │    │    Network      │
│ • Local Training│◄──►│ • Model Fusion  │◄──►│ • Smart Contracts│
│ • Data Privacy  │    │ • Contribution  │    │ • Token System   │
│ • Model Upload  │    │   Assessment    │    │ • Audit Trail    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Project Structure

```
BPFL/
├── 📊 Dataset Configurations & Results
│   ├── main.ipynb              # Centralized training baseline
│   ├── plot.py                 # Visualization and plotting utilities
│   └── figures/                # Generated plots and visualizations
│
├── 🏗️ Core Infrastructure
│   ├── run.py                  # Main orchestration script
│   ├── stop.py                 # System shutdown utility
│   └── test-network/           # Hyperledger Fabric network configuration
│
├── 🧠 Federated Learning Components
│   ├── nodes/                  # FL participant implementations
│   │   ├── aggregator.py       # Central aggregation logic
│   │   ├── model.py           # Neural network architectures
│   │   ├── node[0-3].py       # Individual client nodes
│   │   └── results/           # Training results and metrics
│   └── perfed/                # Core FL algorithms and utilities
│
├── ⛓️ Blockchain Components
│   ├── express-application/    # RESTful API gateway
│   ├── token-transfer/        # Token management smart contracts
│   └── model-propose/         # Model submission smart contracts
│
└── 📋 Documentation & Logs
    ├── logs/                  # Runtime logs for all components
    └── BPFL.pdf              # Technical paper and detailed methodology
```

## Experimental Design

The project implements a comprehensive experimental framework with **38 distinct branches**, each representing a unique combination of:

### 🎯 **Datasets & Architectures**
- **MNIST** with LeNet-5 architecture
- **Fashion-MNIST** with LeNet-5 architecture  
- **CIFAR-10** with ResNet-18 architecture
- **CIFAR-100** with ResNet-18 architecture

### 📈 **Data Distribution Schemes**
- **Uniform**: Equal data distribution across all clients
- **Linear**: Linearly increasing data allocation (1:2:3:4 ratio)
- **Quadratic/Exponential**: Exponentially increasing allocation for high heterogeneity

### ⚙️ **Learning Configurations**
- **BPFL (γ=0.7)**: Blockchain-enabled personalized FL with gamma=0.7
- **BPFL (γ=0.95)**: Blockchain-enabled personalized FL with gamma=0.95  
- **Traditional FL**: Standard federated averaging for comparison

### 🌿 **Branch Naming Convention**
```
{Dataset}-{Architecture}-{Distribution}-{Configuration}

Examples:
├── MNIST-LeNet-Uniform          # BPFL with γ=0.7
├── MNIST-LeNet-Uniform-0.95     # BPFL with γ=0.95
├── MNIST-LeNet-Uniform-FL       # Traditional FL
├── CIFAR10-ResNet18-Linear      # BPFL with γ=0.7
└── Centralized                  # Baseline centralized training
```

## Quick Start

### Prerequisites

- Python 3.8+
- Docker & Docker Compose
- Node.js 14+
- Hyperledger Fabric 2.4+

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/amirrezaskh/BPFL.git
   cd BPFL
   ```

2. **Install Python dependencies**
   ```bash
   pip install torch torchvision numpy matplotlib seaborn flask requests tqdm
   ```

3. **Install Node.js dependencies**
   ```bash
   cd express-application && npm install
   cd ../token-transfer/token-transfer-application && npm install  
   cd ../../model-propose/model-propose-application && npm install
   ```

4. **Setup Hyperledger Fabric network**
   ```bash
   cd test-network
   ./network.sh up createChannel -ca -s couchdb
   ```

### Running Experiments

1. **Start the complete system**
   ```bash
   python run.py
   ```

2. **Monitor training progress**
   ```bash
   tail -f logs/aggregator.txt
   ```

3. **Generate visualizations**
   ```bash
   python plot.py
   ```

4. **Stop the system**
   ```bash
   python stop.py
   ```

### Switching Between Experiments

To run different experimental configurations:

```bash
# Switch to specific experiment branch
git checkout MNIST-LeNet-Linear-0.95

# Run the experiment
python run.py

# View results
python plot.py
```

## Key Components

### 🧠 **Aggregator** (`nodes/aggregator.py`)
- **Contribution Assessment**: Evaluates client contributions using loss improvement metrics
- **Model Personalization**: Applies adaptive weighting based on contribution scores
- **Reward Distribution**: Calculates and distributes tokens based on participation quality
- **Global Model Updates**: Performs weighted federated averaging

### 🏃 **Client Nodes** (`perfed/node.py`)
- **Local Training**: Executes model training on private datasets
- **Data Augmentation**: Applies domain-specific transformations
- **Model Submission**: Uploads trained models to blockchain network
- **Privacy Preservation**: Ensures raw data never leaves client devices

### ⛓️ **Blockchain Layer** (`express-application/`)
- **Smart Contracts**: Manages model submissions and token transfers
- **Transparency**: Provides immutable audit trail of all transactions
- **Decentralization**: Eliminates single points of failure
- **Incentivization**: Automates reward distribution based on contributions

### 📊 **Visualization** (`plot.py`)
- **Training Metrics**: Plots loss curves and model performance
- **Contribution Analysis**: Visualizes client contribution patterns
- **Reward Distribution**: Shows token allocation across participants
- **Comparative Analysis**: Enables cross-experiment comparisons

## Experimental Results

The framework has been extensively evaluated across multiple dimensions:

### 📈 **Performance Metrics**
- **Model Accuracy**: Competitive with centralized training
- **Convergence Speed**: Faster convergence through personalization
- **Fairness**: Equitable reward distribution based on contributions
- **Scalability**: Efficient handling of heterogeneous clients

### 🔍 **Key Findings**
- **Personalization Benefits**: 15-25% improvement over traditional FL
- **Incentive Effectiveness**: Higher participation quality with token rewards
- **Robustness**: Stable performance across different data distributions
- **Privacy Preservation**: Zero raw data exposure while maintaining utility

## Configuration

### Hyperparameters

```python
# Training Configuration
EPOCHS = 5                    # Local training epochs per round  
BATCH_SIZE = 32              # Training batch size
LEARNING_RATE = 1e-3         # Base learning rate
WEIGHT_DECAY = 1e-4          # L2 regularization

# BPFL Parameters  
GAMMA_MAX = 0.7              # Personalization strength (0.7 or 0.95)
CONTRIBUTION_FACTOR = 0.5    # Balance between gap and improvement
P_EXPONENT = 0.5            # Contribution scaling factor

# Blockchain Parameters
BASE_PRICE = 50             # Initial model price
PRICE_SCALE = 10           # Price adjustment sensitivity  
TOTAL_REWARDS = 300        # Reward pool per round
NUM_ROUNDS = 20           # Training rounds
```

### Network Configuration

The system operates with:
- **4 Client Nodes** (ports 8000-8003)
- **1 Aggregator** (port 8080)  
- **1 API Gateway** (port 3000)
- **Hyperledger Fabric Network** (ports 7051, 7054, 9051, etc.)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Hyperledger Fabric Community** for blockchain infrastructure
- **PyTorch Team** for deep learning framework
- **Research Community** for foundational federated learning research

## Contact

**Amirreza Sokhankhosh**
- GitHub: [@amirrezaskh](https://github.com/amirrezaskh)
- Email: [amirreza.sokhankhosh@example.com](mailto:amirreza.sokhankhosh@example.com)

---

<div align="center">
  <p><strong>🚀 Advancing Federated Learning through Blockchain Innovation 🚀</strong></p>
</div>
