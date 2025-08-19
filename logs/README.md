# System Logs Directory

This directory contains runtime logs from all BPFL system components, providing comprehensive monitoring and debugging capabilities for the federated learning and blockchain infrastructure.

## Directory Structure

```
logs/
├── aggregator.txt          # Central aggregation server logs
├── app.txt                 # Express API gateway logs
├── node_0.txt             # Client node 0 training logs
├── node_1.txt             # Client node 1 training logs  
├── node_2.txt             # Client node 2 training logs
└── node_3.txt             # Client node 3 training logs
```

## Log File Overview

### 🔄 **Aggregator Logs** (`aggregator.txt`)

**Content:**
- Model aggregation process details
- Contribution calculation results
- Reward distribution outcomes
- Global model update confirmations
- Blockchain interaction status

**Sample Log Entries:**
```
2024-01-15 10:30:15 - Starting aggregation for round 5
2024-01-15 10:30:16 - Received 4 model submissions
2024-01-15 10:30:17 - Computing contributions: [0.245, 0.187, 0.332, 0.201]
2024-01-15 10:30:18 - Updated global model with weighted average
2024-01-15 10:30:19 - Applied personalization with gamma=0.7
2024-01-15 10:30:20 - Distributed rewards: [73.5, 56.1, 99.6, 60.3] tokens
2024-01-15 10:30:21 - Stored results on blockchain
2024-01-15 10:30:22 - Round 5 aggregation complete
```

**Key Monitoring Points:**
- **Submission Validation**: Ensures all expected models received
- **Contribution Fairness**: Monitors contribution score distribution
- **Performance Tracking**: Global model loss improvements
- **Economic Balance**: Token distribution and pricing updates

### 🌐 **API Gateway Logs** (`app.txt`)

**Content:**
- HTTP request/response handling
- Blockchain transaction confirmations
- Round orchestration events
- Smart contract invocation results
- System health monitoring

**Sample Log Entries:**
```
2024-01-15 10:25:00 - Server starting on port 3000
2024-01-15 10:25:01 - Connected to Hyperledger Fabric network
2024-01-15 10:25:02 - Token contract initialized: tokenCC
2024-01-15 10:25:03 - Model contract initialized: modelCC
2024-01-15 10:28:45 - POST /api/model/ - Model submission from wallet_0
2024-01-15 10:28:46 - Blockchain transaction confirmed: model_0 created
2024-01-15 10:29:15 - All 4 models submitted - triggering aggregation
2024-01-15 10:30:25 - POST /api/aggregator/ - Processing reward distribution
2024-01-15 10:30:26 - Round 5 complete - starting round 6
```

**Monitoring Capabilities:**
- **Request Tracking**: All API endpoint usage
- **Blockchain Status**: Smart contract transaction confirmations
- **Round Coordination**: Training round lifecycle management
- **Error Detection**: Failed transactions and system issues

### 🤖 **Client Node Logs** (`node_[0-3].txt`)

**Content:**
- Local training progress and metrics
- Data loading and preprocessing status
- Model submission confirmations
- HTTP communication with API gateway
- Training hyperparameter details

**Sample Log Entries:**
```
2024-01-15 10:28:30 - Node 0 starting training round 5
2024-01-15 10:28:31 - Loaded personalized model from /nodes/models/g_model_0.pt
2024-01-15 10:28:32 - Training dataset: 12500 samples (uniform distribution)
2024-01-15 10:28:33 - Epoch 1/5: Loss = 1.2456
2024-01-15 10:28:45 - Epoch 2/5: Loss = 1.1892
2024-01-15 10:29:05 - Epoch 3/5: Loss = 1.1345
2024-01-15 10:29:25 - Epoch 4/5: Loss = 1.0987
2024-01-15 10:29:45 - Epoch 5/5: Loss = 1.0654
2024-01-15 10:29:46 - Training complete - submitting model to blockchain
2024-01-15 10:29:47 - Model submission confirmed: model_0
2024-01-15 10:29:48 - Waiting for next round...
```

**Training Insights:**
- **Convergence Monitoring**: Local model loss trajectories
- **Data Distribution**: Training set characteristics
- **Performance Analysis**: Epoch-by-epoch improvement
- **Network Communication**: Blockchain submission status