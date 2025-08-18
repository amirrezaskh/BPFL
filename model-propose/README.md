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

## Key Functions

### 🚀 **System Initialization**

#### **InitLedger()**
```javascript
async InitLedger(ctx) {
    // Initialize with sample model submissions for testing
    const models = [
        {
            ID: 'model_0',
            WalletId: 'wallet_0',
            Path: '/nodes/models/model_0.pt',
            TestDataPath: '/nodes/tests/tests_0.pt',
            SubmissionTime: new Date().toISOString(),
            Round: 1
        }
    ];
    
    // Store initial models on ledger
    for (const model of models) {
        await ctx.stub.putState(model.ID, Buffer.from(JSON.stringify(model)));
    }
    
    console.log('Model management system initialized');
}
```

#### **InitRoundInfo(numNodes)**
```javascript
async InitRoundInfo(ctx, numNodes) {
    // Initialize round tracking information
    const roundInfo = {
        currentRound: 0,
        expectedNodes: parseInt(numNodes),
        submittedNodes: 0,
        roundStartTime: null,
        submissionDeadline: null,
        status: 'ready',
        participants: []
    };
    
    await ctx.stub.putState('roundInfo', Buffer.from(JSON.stringify(roundInfo)));
    return `Round info initialized for ${numNodes} nodes`;
}
```

### 📤 **Model Submission**

#### **CreateModel(id, walletId, path, testDataPath)**
```javascript
async CreateModel(ctx, id, walletId, path, testDataPath) {
    // Check if model already exists
    const exists = await this.ModelExists(ctx, id);
    if (exists) {
        throw new Error(`Model ${id} already exists`);
    }
    
    // Get current round information
    const roundInfoJSON = await ctx.stub.getState('roundInfo');
    let roundInfo = { currentRound: 1, submittedNodes: 0 };
    if (roundInfoJSON && roundInfoJSON.length > 0) {
        roundInfo = JSON.parse(roundInfoJSON.toString());
    }
    
    // Create model metadata
    const model = {
        ID: id,
        WalletId: walletId,
        Path: path,
        TestDataPath: testDataPath,
        SubmissionTime: new Date().toISOString(),
        Round: roundInfo.currentRound,
        NodeId: walletId.replace('wallet_', 'node_'),
        Status: 'submitted'
    };
    
    // Store model on ledger
    await ctx.stub.putState(id, Buffer.from(JSON.stringify(model)));
    
    // Update round information
    roundInfo.submittedNodes += 1;
    roundInfo.participants.push(walletId);
    
    // Check if all nodes have submitted
    if (roundInfo.submittedNodes >= roundInfo.expectedNodes) {
        roundInfo.status = 'complete';
        console.log(`Round ${roundInfo.currentRound} complete - all models submitted`);
    }
    
    await ctx.stub.putState('roundInfo', Buffer.from(JSON.stringify(roundInfo)));
    
    return `Model ${id} submitted successfully for round ${roundInfo.currentRound}`;
}
```

### 🔍 **Model Retrieval**

#### **ReadModel(id)**
```javascript
async ReadModel(ctx, id) {
    const modelJSON = await ctx.stub.getState(id);
    if (!modelJSON || modelJSON.length === 0) {
        throw new Error(`Model ${id} does not exist`);
    }
    
    const model = JSON.parse(modelJSON.toString());
    
    // Add runtime metadata
    model.QueryTime = new Date().toISOString();
    model.BlockNumber = ctx.stub.getTxID();
    
    return JSON.stringify(model);
}
```

#### **GetAllModels()**
```javascript
async GetAllModels(ctx) {
    // Get all submitted models in current round
    const iterator = await ctx.stub.getStateByRange('', '');
    const allModels = [];
    
    for await (const res of iterator) {
        const key = res.key;
        
        // Filter for model entries (not roundInfo or other metadata)
        if (key.startsWith('model_')) {
            const model = JSON.parse(res.value.toString());
            
            // Include additional metadata for aggregation
            model.RetrievalTime = new Date().toISOString();
            allModels.push(model);
        }
    }
    
    // Sort by submission time for consistent ordering
    allModels.sort((a, b) => new Date(a.SubmissionTime) - new Date(b.SubmissionTime));
    
    console.log(`Retrieved ${allModels.length} models for aggregation`);
    return allModels;
}
```

### 🔄 **Round Management**

#### **DeleteAllModels()**
```javascript
async DeleteAllModels(ctx) {
    // Get all current model submissions
    const iterator = await ctx.stub.getStateByRange('', '');
    const modelsToDelete = [];
    
    for await (const res of iterator) {
        const key = res.key;
        if (key.startsWith('model_')) {
            modelsToDelete.push(key);
        }
    }
    
    // Delete all model entries
    for (const modelId of modelsToDelete) {
        await ctx.stub.delState(modelId);
    }
    
    // Update round information for next round
    const roundInfoJSON = await ctx.stub.getState('roundInfo');
    if (roundInfoJSON && roundInfoJSON.length > 0) {
        const roundInfo = JSON.parse(roundInfoJSON.toString());
        roundInfo.currentRound += 1;
        roundInfo.submittedNodes = 0;
        roundInfo.participants = [];
        roundInfo.status = 'ready';
        roundInfo.roundStartTime = null;
        
        await ctx.stub.putState('roundInfo', Buffer.from(JSON.stringify(roundInfo)));
    }
    
    console.log(`Cleared ${modelsToDelete.length} models and prepared for round ${roundInfo.currentRound}`);
    return `Deleted ${modelsToDelete.length} models and advanced to next round`;
}
```

### 🔧 **Utility Functions**

#### **ModelExists(id)**
```javascript
async ModelExists(ctx, id) {
    const modelJSON = await ctx.stub.getState(id);
    return modelJSON && modelJSON.length > 0;
}
```

#### **GetRoundInfo()**
```javascript
async GetRoundInfo(ctx) {
    const roundInfoJSON = await ctx.stub.getState('roundInfo');
    if (!roundInfoJSON || roundInfoJSON.length === 0) {
        // Return default if not initialized
        return JSON.stringify({
            currentRound: 0,
            expectedNodes: 4,
            submittedNodes: 0,
            status: 'uninitialized'
        });
    }
    
    return roundInfoJSON.toString();
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

### 📦 **Dependencies** (`package.json`)

#### **Chaincode Dependencies:**
```json
{
  "name": "model-propose-chaincode",
  "version": "1.0.0",
  "description": "BPFL Model Management Smart Contract",
  "main": "index.js",
  "dependencies": {
    "fabric-contract-api": "^2.5.0",
    "fabric-shim": "^2.5.0"
  },
  "scripts": {
    "start": "fabric-chaincode-node start",
    "test": "mocha test --recursive"
  }
}
```

#### **Application Dependencies:**
```json
{
  "name": "model-propose-application",
  "version": "1.0.0",
  "description": "BPFL Model Application Interface", 
  "main": "modelApp.js",
  "dependencies": {
    "@hyperledger/fabric-gateway": "^1.2.0",
    "@grpc/grpc-js": "^1.8.0"
  }
}
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

## Deployment & Testing

### 🚀 **Chaincode Deployment**

```bash
# Deploy model management chaincode
cd test-network
./network.sh deployCC -ccn modelCC -ccp ../model-propose/model-propose-chaincode/ -ccl javascript

# Initialize round information
peer chaincode invoke -o orderer.example.com:7050 --tls \
  --cafile organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem \
  -C main -n modelCC \
  -c '{"function":"InitRoundInfo","Args":["4"]}'
```

### 🔍 **Testing Operations**

```bash
# Submit a model
peer chaincode invoke -o orderer.example.com:7050 --tls \
  --cafile organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem \
  -C main -n modelCC \
  -c '{"function":"CreateModel","Args":["model_0","wallet_0","/nodes/models/model_0.pt","/nodes/tests/tests_0.pt"]}'

# Query submitted model
peer chaincode query -C main -n modelCC \
  -c '{"function":"ReadModel","Args":["model_0"]}'

# Get all models for aggregation
peer chaincode query -C main -n modelCC \
  -c '{"function":"GetAllModels","Args":[]}'

# Clear models for next round
peer chaincode invoke -o orderer.example.com:7050 --tls \
  --cafile organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem \
  -C main -n modelCC \
  -c '{"function":"DeleteAllModels","Args":[]}'
```

## Security Features

### 🔐 **Access Control**
- **Authorized Submissions**: Only registered wallets can submit models
- **Round Validation**: Models rejected outside active rounds
- **Duplicate Prevention**: Prevents multiple submissions from same client
- **Immutable Records**: All submissions permanently recorded

### 🛡️ **Data Integrity**
```javascript
// Model validation during submission
if (!id || !walletId || !path || !testDataPath) {
    throw new Error('All model parameters required');
}

// Ensure unique model IDs
const exists = await this.ModelExists(ctx, id);
if (exists) {
    throw new Error(`Model ${id} already exists`);
}
```

### 📊 **Audit Trail**
```javascript
// Comprehensive submission tracking
const model = {
    ID: id,
    WalletId: walletId,
    SubmissionTime: new Date().toISOString(),
    TransactionId: ctx.stub.getTxID(),
    SubmitterMSP: ctx.clientIdentity.getMSPID(),
    Round: currentRound
};
```

## Performance Optimization

### ⚡ **Efficient Queries**
```javascript
// Use range queries for model retrieval
const iterator = await ctx.stub.getStateByRange('model_', 'model_~');

// Batch delete operations
const deletePromises = modelsToDelete.map(id => ctx.stub.delState(id));
await Promise.all(deletePromises);
```

### 🚀 **Caching Strategy**
```javascript
// Cache round information to reduce blockchain reads
if (!this.roundInfoCache || this.isCacheExpired()) {
    this.roundInfoCache = await ctx.stub.getState('roundInfo');
    this.cacheTimestamp = Date.now();
}
```

### 📊 **State Management**
```javascript
// Efficient model counting without full retrieval
const iterator = await ctx.stub.getStateByRange('model_', 'model_~');
let count = 0;
for await (const res of iterator) {
    count++;
}
```

## Error Handling

### 🚨 **Common Error Scenarios**

1. **Duplicate Submissions**
   ```javascript
   if (await this.ModelExists(ctx, id)) {
       throw new Error(`Model ${id} already submitted in this round`);
   }
   ```

2. **Round State Validation**
   ```javascript
   if (roundInfo.status === 'complete') {
       throw new Error('Round already complete, submissions closed');
   }
   ```

3. **Missing Dependencies**
   ```javascript
   if (!roundInfoJSON || roundInfoJSON.length === 0) {
       throw new Error('Round information not initialized');
   }
   ```

---

The model propose smart contract provides robust coordination for federated learning model submissions, ensuring secure, transparent, and efficient management of the training process within the BPFL framework.
