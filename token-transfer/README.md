# Token Transfer Smart Contract

This directory contains the blockchain smart contract (chaincode) that implements the incentive token system for the BPFL framework, managing participant wallets, transactions, model pricing, and reward distribution.

## Directory Structure

```
token-transfer/
├── token-transfer-application/     # Application layer interface
│   ├── tokenApp.js                # Token management API wrapper
│   └── package.json               # Application dependencies
└── token-transfer-chaincode/       # Smart contract implementation
    ├── index.js                   # Chaincode entry point
    ├── package.json               # Chaincode dependencies
    └── lib/
        └── tokenTransfer.js       # Core token logic implementation
```

## Overview

The token transfer system provides the economic foundation for the BPFL framework by:

- **💰 Wallet Management**: Creating and managing participant token wallets
- **🔄 Transaction Processing**: Handling secure token transfers between participants
- **💎 Model Pricing**: Dynamic pricing based on global model performance
- **🎁 Reward Distribution**: Incentivizing quality contributions through token rewards
- **📊 Economic Analytics**: Tracking system-wide economic metrics

## Smart Contract Architecture

### 🏗️ **Chaincode Structure** (`token-transfer-chaincode/`)

#### **Entry Point** (`index.js`)
```javascript
const { Contract } = require('fabric-contract-api');
const TokenTransfer = require('./lib/tokenTransfer');

class TokenTransferContract extends Contract {
    // Delegate all operations to TokenTransfer class
    async InitLedger(ctx) {
        return await TokenTransfer.InitLedger(ctx);
    }
    
    async CreateWallet(ctx, id, balance) {
        return await TokenTransfer.CreateWallet(ctx, id, balance);
    }
    
    // ... other delegated methods
}

module.exports = TokenTransferContract;
```

#### **Core Implementation** (`lib/tokenTransfer.js`)

The main smart contract logic implementing the token economy:

```javascript
class TokenTransfer {
    // Initialize the token system with default state
    async InitLedger(ctx) { /* ... */ }
    
    // Wallet operations
    async CreateWallet(ctx, id, balance) { /* ... */ }
    async ReadWallet(ctx, id) { /* ... */ }
    async GetAllWallets(ctx) { /* ... */ }
    
    // Transaction processing
    async ProcessTransaction(ctx, id, amount) { /* ... */ }
    
    // Model pricing and rewards
    async UpdatePrice(ctx, newPrice) { /* ... */ }
    async ProcessRewards(ctx, submitsJSON) { /* ... */ }
    
    // System state management
    async InitWallets(ctx, numNodes, basePrice, scale, totalRewards) { /* ... */ }
}
```

### 🏦 **Wallet Data Structure**

Each participant wallet contains:

```json
{
  "ID": "wallet_0",
  "Balance": 1000,
  "TotalEarned": 0,
  "TotalSpent": 0,
  "TransactionHistory": [],
  "CreatedAt": "2024-01-15T10:30:00Z",
  "LastUpdated": "2024-01-15T10:30:00Z"
}
```

### 💰 **Pricing Information Structure**

Global model pricing and reward pool:

```json
{
  "price": 50,
  "scale": 10,
  "totalRewards": 300,
  "lastUpdated": "2024-01-15T10:30:00Z",
  "priceHistory": [45, 48, 50, 52, 49]
}
```

## Token Economics

### 💎 **Initial Configuration**
```javascript
// Default wallet setup
INITIAL_BALANCE = 1000;     // Starting tokens per participant
BASE_PRICE = 50;            // Initial global model price
PRICE_SCALE = 10;           // Price adjustment sensitivity
TOTAL_REWARDS = 300;        // Token pool distributed per round
```

### 📈 **Dynamic Pricing Model**

The global model price adjusts based on performance improvements:

```javascript
// Price update formula
newPrice = prevPrice + scale * deltaLoss

// Where:
// deltaLoss = prevGlobalLoss - newGlobalLoss
// scale = price adjustment sensitivity (default: 10)
```

**Price Behavior:**
- **Improvement** (deltaLoss > 0): Price increases, rewarding better models
- **Degradation** (deltaLoss < 0): Price decreases, adjusting for poor performance
- **Stability** (deltaLoss ≈ 0): Price remains relatively stable

### 🎁 **Reward Distribution Algorithm**

Rewards are distributed based on contribution quality using logarithmic smoothing:

```javascript
// Contribution-based reward calculation
contributions = [c1, c2, c3, c4];                    // Raw contribution scores
smoothed = contributions.map(c => Math.log1p(c));    // Logarithmic smoothing
weights = smoothed.map(s => s / sum(smoothed));       // Normalize to weights
rewards = weights.map(w => w * totalRewards);         // Distribute reward pool
```

**Benefits of Logarithmic Smoothing:**
- **Fairness**: Prevents single dominant participant from capturing all rewards
- **Participation**: Ensures all contributors receive meaningful compensation
- **Stability**: Reduces volatility from outlier contributions

## Key Functions

### 🚀 **System Initialization**

#### **InitLedger()** 
```javascript
async InitLedger(ctx) {
    // Initialize with sample data for testing
    const wallets = [
        { ID: 'wallet_0', Balance: 1000, TotalEarned: 0, TotalSpent: 0 },
        { ID: 'wallet_1', Balance: 1000, TotalEarned: 0, TotalSpent: 0 }
    ];
    
    // Store initial wallets on ledger
    for (const wallet of wallets) {
        await ctx.stub.putState(wallet.ID, Buffer.from(JSON.stringify(wallet)));
    }
}
```

#### **InitWallets(numNodes, basePrice, scale, totalRewards)**
```javascript
async InitWallets(ctx, numNodes, basePrice, scale, totalRewards) {
    // Create wallets for all federated learning participants
    for (let i = 0; i < parseInt(numNodes); i++) {
        await this.CreateWallet(ctx, `wallet_${i}`, '1000');
    }
    
    // Initialize pricing information
    const priceInfo = {
        price: parseInt(basePrice),
        scale: parseInt(scale), 
        totalRewards: parseInt(totalRewards),
        lastUpdated: new Date().toISOString()
    };
    
    await ctx.stub.putState('priceInfo', Buffer.from(JSON.stringify(priceInfo)));
}
```

### 🏦 **Wallet Management**

#### **CreateWallet(id, balance)**
```javascript
async CreateWallet(ctx, id, balance) {
    const exists = await this.WalletExists(ctx, id);
    if (exists) {
        throw new Error(`Wallet ${id} already exists`);
    }
    
    const wallet = {
        ID: id,
        Balance: parseInt(balance),
        TotalEarned: 0,
        TotalSpent: 0,
        TransactionHistory: [],
        CreatedAt: new Date().toISOString(),
        LastUpdated: new Date().toISOString()
    };
    
    await ctx.stub.putState(id, Buffer.from(JSON.stringify(wallet)));
    return `Wallet ${id} created with balance ${balance}`;
}
```

#### **ReadWallet(id)**
```javascript
async ReadWallet(ctx, id) {
    const walletJSON = await ctx.stub.getState(id);
    if (!walletJSON || walletJSON.length === 0) {
        throw new Error(`Wallet ${id} does not exist`);
    }
    return walletJSON.toString();
}
```

#### **GetAllWallets()**
```javascript
async GetAllWallets(ctx) {
    const iterator = await ctx.stub.getStateByRange('', '');
    const allResults = [];
    
    for await (const res of iterator) {
        const key = res.key;
        if (key.startsWith('wallet_')) {
            const value = JSON.parse(res.value.toString());
            allResults.push(value);
        }
    }
    
    return JSON.stringify(allResults);
}
```

### 💸 **Transaction Processing**

#### **ProcessTransaction(id, amount)**
```javascript
async ProcessTransaction(ctx, id, amount) {
    const wallet = JSON.parse(await this.ReadWallet(ctx, id));
    const transactionAmount = parseInt(amount);
    
    // Validate sufficient balance
    if (wallet.Balance < transactionAmount) {
        throw new Error(`Insufficient balance. Current: ${wallet.Balance}, Required: ${transactionAmount}`);
    }
    
    // Process transaction
    wallet.Balance -= transactionAmount;
    wallet.TotalSpent += transactionAmount;
    wallet.TransactionHistory.push({
        amount: transactionAmount,
        type: 'debit',
        timestamp: new Date().toISOString(),
        description: 'Model purchase'
    });
    wallet.LastUpdated = new Date().toISOString();
    
    await ctx.stub.putState(id, Buffer.from(JSON.stringify(wallet)));
    return `Transaction processed: ${transactionAmount} tokens debited from ${id}`;
}
```

### 💎 **Pricing & Rewards**

#### **UpdatePrice(newPrice)**
```javascript
async UpdatePrice(ctx, newPrice) {
    const priceInfoJSON = await ctx.stub.getState('priceInfo');
    if (!priceInfoJSON || priceInfoJSON.length === 0) {
        throw new Error('Price information not initialized');
    }
    
    const priceInfo = JSON.parse(priceInfoJSON.toString());
    const oldPrice = priceInfo.price;
    priceInfo.price = parseFloat(newPrice);
    priceInfo.lastUpdated = new Date().toISOString();
    
    // Maintain price history for analytics
    if (!priceInfo.priceHistory) {
        priceInfo.priceHistory = [];
    }
    priceInfo.priceHistory.push(oldPrice);
    
    // Keep only last 20 prices for efficiency
    if (priceInfo.priceHistory.length > 20) {
        priceInfo.priceHistory = priceInfo.priceHistory.slice(-20);
    }
    
    await ctx.stub.putState('priceInfo', Buffer.from(JSON.stringify(priceInfo)));
    return `Price updated from ${oldPrice} to ${newPrice}`;
}
```

#### **ProcessRewards(submitsJSON)**
```javascript
async ProcessRewards(ctx, submitsJSON) {
    const submits = JSON.parse(submitsJSON);
    const priceInfo = JSON.parse(await ctx.stub.getState('priceInfo'));
    
    for (const submit of submits) {
        const wallet = JSON.parse(await this.ReadWallet(ctx, submit.walletId));
        const reward = parseFloat(submit.reward);
        
        // Credit reward to wallet
        wallet.Balance += reward;
        wallet.TotalEarned += reward;
        wallet.TransactionHistory.push({
            amount: reward,
            type: 'credit',
            timestamp: new Date().toISOString(),
            description: 'Training contribution reward'
        });
        wallet.LastUpdated = new Date().toISOString();
        
        await ctx.stub.putState(submit.walletId, Buffer.from(JSON.stringify(wallet)));
    }
    
    return `Rewards processed for ${submits.length} participants`;
}
```

## Application Layer

### 🔌 **Token Application Interface** (`token-transfer-application/`)

#### **TokenApp Class** (`tokenApp.js`)

Provides a high-level JavaScript interface for interacting with the token smart contract:

```javascript
class TokenApp {
    // Initialize token system
    async initWallets(contract, numNodes, basePrice, scale, totalRewards) {
        const result = await contract.submitTransaction('InitWallets', numNodes, basePrice, scale, totalRewards);
        return result.toString();
    }
    
    // Wallet operations
    async createWallet(contract, id, balance) {
        const result = await contract.submitTransaction('CreateWallet', id, balance);
        return result.toString();
    }
    
    async readKey(contract, key) {
        const result = await contract.evaluateTransaction('ReadWallet', key);
        return JSON.parse(result.toString());
    }
    
    async getAllWallets(contract) {
        const result = await contract.evaluateTransaction('GetAllWallets');
        return JSON.parse(result.toString());
    }
    
    // Transaction processing
    async processTransaction(contract, id, amount) {
        const result = await contract.submitTransaction('ProcessTransaction', id, amount);
        return result.toString();
    }
    
    // Pricing and rewards
    async updatePrice(contract, newPrice) {
        const result = await contract.submitTransaction('UpdatePrice', newPrice);
        return result.toString();
    }
    
    async processRewards(contract, submitsJSON) {
        const result = await contract.submitTransaction('ProcessRewards', submitsJSON);
        return result.toString();
    }
}

module.exports = { TokenApp };
```

### 📦 **Dependencies** (`package.json`)

#### **Chaincode Dependencies:**
```json
{
  "name": "token-transfer-chaincode",
  "version": "1.0.0",
  "description": "BPFL Token Management Smart Contract",
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
  "name": "token-transfer-application",
  "version": "1.0.0", 
  "description": "BPFL Token Application Interface",
  "main": "tokenApp.js",
  "dependencies": {
    "@hyperledger/fabric-gateway": "^1.2.0",
    "@grpc/grpc-js": "^1.8.0"
  }
}
```

## Deployment

### 🚀 **Chaincode Deployment**

```bash
# Package the chaincode
cd test-network
./network.sh deployCC -ccn tokenCC -ccp ../token-transfer/token-transfer-chaincode/ -ccl javascript

# Initialize the token system
peer chaincode invoke -o orderer.example.com:7050 --tls \
  --cafile organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem \
  -C main -n tokenCC \
  -c '{"function":"InitWallets","Args":["4","50","10","300"]}'
```

### 🔍 **Testing & Validation**

```bash
# Query wallet balance
peer chaincode query -C main -n tokenCC \
  -c '{"function":"ReadWallet","Args":["wallet_0"]}'

# Process a transaction
peer chaincode invoke -o orderer.example.com:7050 --tls \
  --cafile organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem \
  -C main -n tokenCC \
  -c '{"function":"ProcessTransaction","Args":["wallet_0","50"]}'

# Check updated balance
peer chaincode query -C main -n tokenCC \
  -c '{"function":"ReadWallet","Args":["wallet_0"]}'
```

## Security Features

### 🔐 **Access Control**
- **Chaincode ACLs**: Only authorized peers can invoke transactions
- **Input Validation**: All parameters validated before processing
- **Balance Verification**: Insufficient balance transactions rejected
- **Immutable Records**: All transactions permanently recorded on blockchain

### 🛡️ **Error Handling**
```javascript
// Comprehensive error handling
try {
    const wallet = JSON.parse(await this.ReadWallet(ctx, id));
    // Process transaction...
} catch (error) {
    throw new Error(`Transaction failed: ${error.message}`);
}
```

### 📊 **Audit Trail**
```javascript
// Transaction history tracking
wallet.TransactionHistory.push({
    amount: transactionAmount,
    type: 'debit' | 'credit',
    timestamp: new Date().toISOString(),
    description: 'Descriptive message',
    blockNumber: ctx.stub.getTxID()
});
```

## Performance Optimization

### ⚡ **Efficient State Management**
```javascript
// Batch operations for multiple wallets
const promises = wallets.map(wallet => 
    ctx.stub.putState(wallet.ID, Buffer.from(JSON.stringify(wallet)))
);
await Promise.all(promises);
```

### 📊 **Query Optimization**
```javascript
// Use composite keys for complex queries
const compositeKey = ctx.stub.createCompositeKey('wallet', [orgId, userId]);
await ctx.stub.putState(compositeKey, walletData);
```

### 🚀 **Caching Strategy**
```javascript
// Cache frequently accessed data
if (!this.priceInfoCache || this.cacheExpired()) {
    this.priceInfoCache = await ctx.stub.getState('priceInfo');
    this.cacheTimestamp = Date.now();
}
```

---

The token transfer smart contract provides a robust, secure, and efficient foundation for the incentive economy within the BPFL system, ensuring fair reward distribution and transparent financial operations on the blockchain.
