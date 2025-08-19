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

## Token Economics

### 💎 **Initial Configuration**
```javascript
// Default wallet setup
INITIAL_BALANCE = 0;     // Starting tokens per participant
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