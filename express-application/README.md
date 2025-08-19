# Express Application - API Gateway

This directory contains the Express.js-based API gateway that serves as the central coordination layer for the BPFL system, managing communication between federated learning nodes, the aggregator, and the blockchain network.

## Directory Structure

```
express-application/
├── app.js           # Main Express server application
├── package.json     # Node.js dependencies and scripts
└── node_modules/    # Installed dependencies (after npm install)
```

## Overview

The Express application acts as the **orchestration hub** for the entire BPFL system, providing:

- **RESTful API endpoints** for system coordination
- **Blockchain integration** with Hyperledger Fabric
- **Round management** for federated learning cycles
- **Model and wallet management** through smart contracts
- **Automated reward distribution** based on contributions

## Core Components

### 🚀 **Main Application** (`app.js`)

The central Express server that coordinates all system components.

#### **Key Features:**

1. **Blockchain Connectivity**
   - Hyperledger Fabric SDK integration
   - Smart contract invocation for tokens and models
   - Cryptographic identity management
   - TLS-secured peer communication

2. **Round Orchestration**
   - Automated training round management
   - Client coordination and synchronization
   - Aggregation triggering when submissions complete
   - Progress monitoring and logging

3. **API Gateway Functions**
   - RESTful endpoints for all system operations
   - JSON payload handling with 50MB limit
   - Error handling and response management
   - Cross-component communication

#### **Architecture Overview:**

```javascript
const express = require("express");
const { TokenApp } = require("../token-transfer/token-transfer-application/tokenApp");
const { ModelApp } = require("../model-propose/model-propose-application/modelApp");

// Initialize blockchain connections
const contractToken = InitConnection("main", "tokenCC");
const contractModel = InitConnection("main", "modelCC");

// Initialize application modules
const tokenApp = new TokenApp();
const modelApp = new ModelApp();
```

### 🔗 **Blockchain Integration**

#### **Hyperledger Fabric Configuration:**
```javascript
// Network parameters
const mspId = "Org1MSP";
const peerEndPoint = "localhost:7051";
const peerHostAlias = "peer0.org1.example.com";

// Cryptographic paths
const cryptoPath = "../test-network/organizations/peerOrganizations/org1.example.com";
const keyDirPath = cryptoPath + "/users/User1@org1.example.com/msp/keystore";
const certPath = cryptoPath + "/users/User1@org1.example.com/msp/signcerts/User1@org1.example.com-cert.pem";
const tlsCertPath = cryptoPath + "/peers/peer0.org1.example.com/tls/ca.crt";
```

#### **Smart Contract Interfaces:**
```javascript
// Token management contract
const contractToken = InitConnection("main", "tokenCC");

// Model submission contract  
const contractModel = InitConnection("main", "modelCC");
```

### ⚙️ **System Configuration**

#### **Training Parameters:**
```javascript
const numNodes = 4;          // Number of federated clients
const rounds = 20;           // Total training rounds
const basePrice = 50;        // Initial model price
const scale = 10;            // Price adjustment factor
const totalRewards = 300;    // Token pool per round
const aggregatorPort = 8080; // Aggregator service port
```

#### **Network Topology:**
```javascript
// Client node ports: 8000-8003
// Aggregator port: 8080  
// Express gateway: 3000
// Blockchain network: 7051, 7054, 9051, etc.
```

## API Endpoints

### 🏦 **Token Management APIs**

#### **Initialize Wallet System**
```http
POST /api/wallets/
```
Creates wallets for all participants with initial token balances.

#### **Create Individual Wallet**
```http
POST /api/wallet/
Content-Type: application/json

{
  "id": "wallet_0",
  "balance": 1000
}
```

#### **Get Wallet Information**
```http
GET /api/wallet/
Content-Type: application/json

{
  "id": "wallet_0"
}
```

#### **Process Transaction**
```http
POST /api/transaction/
Content-Type: application/json

{
  "id": "wallet_0", 
  "amount": 50
}
```

### 🧠 **Model Management APIs**

#### **Initialize Round Information**
```http
POST /api/init/
Content-Type: application/json

{
  "numNodes": 4
}
```

#### **Submit Trained Model**
```http
POST /api/model/
Content-Type: application/json

{
  "id": "model_0",
  "walletId": "wallet_0",
  "path": "/path/to/model.pt",
  "testDataPath": "/path/to/tests.pt"
}
```

#### **Retrieve Model Information**
```http
GET /api/model/
Content-Type: application/json

{
  "id": "model_0"
}
```

#### **Get All Submitted Models**
```http
GET /api/models/
```

#### **Clear All Models**
```http
DELETE /api/models/
```

### 🎮 **System Control APIs**

#### **Start Training Process**
```http
POST /api/start/
Content-Type: application/json

{
  "globalModelPath": "/path/to/global.pt"
}
```

#### **Process Aggregation Results**
```http
POST /api/aggregator/
Content-Type: application/json

{
  "newPrice": 55.5,
  "submits": [
    {
      "walletId": "wallet_0",
      "reward": 75.0,
      "modelPath": "/path/to/personalized_model.pt"
    }
  ]
}
```

#### **System Health Check**
```http
GET /
```

#### **Graceful Shutdown**
```http
GET /exit/
```

## Training Workflow

### 1️⃣ **System Initialization**
```javascript
// 1. Start Hyperledger Fabric network
// 2. Deploy smart contracts
// 3. Initialize wallets with starting balances
// 4. Create initial global model
```

### 2️⃣ **Round Execution**
```javascript
async function startRound(submits) {
    currentRound += 1;
    if (currentRound <= rounds) {
        // Send personalized models to each client
        for (const submit of submits) {
            const port = 8000 + parseInt(submit.walletId[submit.walletId.length - 1]);
            await axios.post(`http://localhost:${port}/round/`, {
                modelPath: submit.modelPath
            });
        }
        console.log(`*** ROUND ${currentRound} STARTED ***`);
    }
}
```

### 3️⃣ **Model Aggregation Trigger**
```javascript
// Triggered when all clients submit models
async function callAggregator() {
    const submits = await modelApp.getAllModels(contractModel);
    const priceInfo = await tokenApp.readKey(contractToken, "priceInfo");
    
    await axios.post(`http://localhost:${aggregatorPort}/aggregate/`, {
        ...priceInfo,
        submits: submits
    });
}
```

### 4️⃣ **Reward Processing**
```javascript
async function processRewards(newPrice, submits) {
    // Update model pricing on blockchain
    await tokenApp.updatePrice(contractToken, newPrice.toString());
    
    // Distribute rewards to participants
    await tokenApp.processRewards(contractToken, JSON.stringify(submits));
    
    // Clear submitted models for next round
    await modelApp.deleteAllModels(contractModel);
    
    // Start next training round
    await startRound(submits);
}
```

## Blockchain Smart Contract Integration

### 🪙 **Token Contract Operations**

```javascript
// Initialize token system
await tokenApp.initWallets(contractToken, numNodes.toString(), basePrice.toString(), scale.toString(), totalRewards.toString());

// Create wallet for participant
await tokenApp.createWallet(contractToken, walletId, balance.toString());

// Read wallet state
const wallet = await tokenApp.readKey(contractToken, walletId);

// Process token transaction
await tokenApp.processTransaction(contractToken, walletId, amount.toString());

// Update model pricing
await tokenApp.updatePrice(contractToken, newPrice.toString());

// Distribute rewards
await tokenApp.processRewards(contractToken, JSON.stringify(submits));
```

### 🧠 **Model Contract Operations**

```javascript
// Initialize model management
await modelApp.initRoundInfo(contractModel, numNodes.toString());

// Submit trained model
await modelApp.createModel(contractModel, modelId, walletId, modelPath, testDataPath);

// Retrieve model information
const model = await modelApp.readModel(contractModel, modelId);

// Get all submitted models
const allModels = await modelApp.getAllModels(contractModel);

// Clear models for next round
await modelApp.deleteAllModels(contractModel);
```

## Configuration Management

### 🔧 **Environment Variables**
```javascript
// Server configuration
const port = process.env.PORT || 3000;
const aggregatorPort = process.env.AGGREGATOR_PORT || 8080;

// Blockchain configuration
const mspId = process.env.MSP_ID || "Org1MSP";
const peerEndPoint = process.env.PEER_ENDPOINT || "localhost:7051";

// Training parameters
const numNodes = parseInt(process.env.NUM_NODES) || 4;
const rounds = parseInt(process.env.ROUNDS) || 20;
const totalRewards = parseInt(process.env.TOTAL_REWARDS) || 300;
```

### 📝 **Logging Configuration**
```javascript
// Request logging middleware
app.use((req, res, next) => {
    console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
    next();
});

// Error handling middleware
app.use((error, req, res, next) => {
    console.error(`Error: ${error.message}`);
    res.status(500).json({ error: error.message });
});
```

