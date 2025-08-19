# Model Propose Smart Contract

This directory contains the blockchain smart contract (chaincode) that manages federated learning model submissions, metadata tracking, and round coordination for the BPFL framework.

## Directory Structure

```
model-propose/
├── model-propose-application/      # Application layer interface
│   ├── modelApp.js                # Model management API wrapper
│   └── package.json               # Application dependencies
└── model-propose-chaincode/        # Smart contract implementation
    ├── index.js                   # Chaincode entry point
    ├── package.json               # Chaincode dependencies
    └── lib/
        └── modelPropose.js        # Core model management logic
```

## Overview

The model propose system provides the coordination layer for federated learning by:

- **📤 Model Submission**: Secure tracking of trained model submissions from clients
- **🔄 Round Management**: Coordinating training rounds and submission collection
- **📊 Metadata Storage**: Immutable record of model paths, ownership, and timestamps
- **🔍 Query Interface**: Efficient retrieval of submitted models for aggregation
- **🧹 State Management**: Cleanup and preparation for subsequent training rounds

## Smart Contract Architecture

### 🏗️ **Chaincode Structure** (`model-propose-chaincode/`)

#### **Entry Point** (`index.js`)
```javascript
const { Contract } = require('fabric-contract-api');
const ModelPropose = require('./lib/modelPropose');

class ModelProposeContract extends Contract {
    // Delegate all operations to ModelPropose class
    async InitLedger(ctx) {
        return await ModelPropose.InitLedger(ctx);
    }
    
    async CreateModel(ctx, id, walletId, path, testDataPath) {
        return await ModelPropose.CreateModel(ctx, id, walletId, path, testDataPath);
    }
    
    // ... other delegated methods
}

module.exports = ModelProposeContract;
```

#### **Core Implementation** (`lib/modelPropose.js`)

The main smart contract logic for model coordination:

```javascript
class ModelPropose {
    // Initialize the model management system
    async InitLedger(ctx) { /* ... */ }
    
    // Model submission operations
    async CreateModel(ctx, id, walletId, path, testDataPath) { /* ... */ }
    async ReadModel(ctx, id) { /* ... */ }
    async GetAllModels(ctx) { /* ... */ }
    
    // Round management
    async InitRoundInfo(ctx, numNodes) { /* ... */ }
    async DeleteAllModels(ctx) { /* ... */ }
    
    // Utility functions
    async ModelExists(ctx, id) { /* ... */ }
}
```

### 🧠 **Model Data Structure**

Each submitted model contains comprehensive metadata:

```json
{
  "ID": "model_0",
  "WalletId": "wallet_0",
  "Path": "/nodes/models/model_0.pt",
  "TestDataPath": "/nodes/tests/tests_0.pt",
  "SubmissionTime": "2024-01-15T10:30:00Z",
  "Round": 5,
  "FileSize": 2048576,
  "ModelHash": "sha256:a1b2c3d4...",
  "NodeId": "node_0",
  "Status": "submitted"
}
```

### 📊 **Round Information Structure**

Global round state and participant tracking:

```json
{
  "currentRound": 5,
  "expectedNodes": 4,
  "submittedNodes": 3,
  "roundStartTime": "2024-01-15T10:00:00Z",
  "submissionDeadline": "2024-01-15T10:05:00Z",
  "status": "collecting",
  "participants": ["wallet_0", "wallet_1", "wallet_2"]
}
```

## Application Layer

### 🔌 **Model Application Interface** (`model-propose-application/`)

#### **ModelApp Class** (`modelApp.js`)

High-level JavaScript interface for model contract interactions:

```javascript
class ModelApp {
    // Initialize model management system
    async initRoundInfo(contract, numNodes) {
        const result = await contract.submitTransaction('InitRoundInfo', numNodes);
        return result.toString();
    }
    
    // Model submission
    async createModel(contract, id, walletId, path, testDataPath) {
        const result = await contract.submitTransaction('CreateModel', id, walletId, path, testDataPath);
        return result.toString();
    }
    
    // Model retrieval
    async readModel(contract, id) {
        const result = await contract.evaluateTransaction('ReadModel', id);
        return JSON.parse(result.toString());
    }
    
    async getAllModels(contract) {
        const result = await contract.evaluateTransaction('GetAllModels');
        return result; // Returns array directly
    }
    
    // Round management
    async deleteAllModels(contract) {
        const result = await contract.submitTransaction('DeleteAllModels');
        return result.toString();
    }
    
    async getRoundInfo(contract) {
        const result = await contract.evaluateTransaction('GetRoundInfo');
        return JSON.parse(result.toString());
    }
}

module.exports = { ModelApp };
```

## Integration with BPFL Workflow

### 🔄 **Training Round Coordination**

1. **Round Initialization**
   ```javascript
   // Express app initializes new round
   await modelApp.initRoundInfo(contract, numNodes);
   ```

2. **Model Submissions**
   ```javascript
   // Each client submits trained model
   await modelApp.createModel(contract, 'model_0', 'wallet_0', '/path/to/model.pt', '/path/to/tests.pt');
   ```

3. **Aggregation Trigger**
   ```javascript
   // When all models submitted, trigger aggregation
   const allModels = await modelApp.getAllModels(contract);
   if (allModels.length === expectedNodes) {
       callAggregator(allModels);
   }
   ```

4. **Round Cleanup**
   ```javascript
   // After aggregation and reward distribution
   await modelApp.deleteAllModels(contract);
   ```

### 🚀 **Client Integration**

Client nodes interact with the model contract through HTTP API:

```javascript
// Node submits model after local training
const response = await fetch('http://localhost:3000/api/model/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        id: 'model_0',
        walletId: 'wallet_0',
        path: '/nodes/models/model_0.pt',
        testDataPath: '/nodes/tests/tests_0.pt'
    })
});
```

### 📊 **Aggregator Integration**

Aggregator retrieves models for processing:

```javascript
// Express app calls aggregator when all models submitted
const submits = await modelApp.getAllModels(contractModel);
const response = await axios.post('http://localhost:8080/aggregate/', {
    submits: submits,
    // ... other parameters
});
```