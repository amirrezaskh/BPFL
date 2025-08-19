# Visualization and Plotting

This directory contains visualization utilities and generated plots that provide comprehensive insights into the BPFL system performance, contribution patterns, and reward distribution across different experimental configurations.

## Files Overview

```
figures/
├── delta_gap.png           # Performance gap improvements over rounds
├── delta_local_loss.png    # Local model improvements over rounds  
├── global_metrics.png      # Global model performance and pricing
├── wallet_contributions.png # Client contribution patterns
├── wallet_local_loss.png   # Individual client loss trajectories
└── wallet_rewards.png      # Token reward distribution over time
```

## Visualization Script

### 📊 **Main Plotting Script** (`plot.py`)

The comprehensive visualization utility that generates all performance plots from training results:

```python
import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn as sns

# Configure plotting style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16, 
    'axes.titlesize': 18,
    'legend.fontsize': 12,
    'figure.figsize': (10, 6),
    'lines.linewidth': 2,
    'lines.markersize': 8
})

# Load experimental results
with open('./nodes/results/res.json', 'r') as f:
    data = json.load(f)

# Generate comprehensive visualization suite
generate_global_metrics_plot(data)
generate_contribution_analysis(data)
generate_reward_distribution(data)
generate_loss_trajectories(data)
generate_improvement_metrics(data)
```

## Generated Visualizations

### 🌍 **Global System Metrics** (`global_metrics.png`)

**Content:**
- **Global Model Loss**: Training loss of the aggregated global model over rounds
- **Model Price Evolution**: Dynamic pricing based on performance improvements

**Insights:**
- Convergence behavior of the federated learning process
- Economic dynamics of the token-based incentive system
- Correlation between model quality and pricing

**Code Implementation:**
```python
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Global model loss trajectory
axs[0].plot(rounds, global_losses, marker='o', color='navy', linewidth=2)
axs[0].set_xlabel('Training Round')
axs[0].set_ylabel('Global Model Loss')
axs[0].set_title('Global Model Loss Over Training Rounds')
axs[0].grid(True, alpha=0.3)

# Model price evolution
axs[1].plot(rounds, new_model_prices, marker='s', color='darkorange', linewidth=2)
axs[1].set_xlabel('Training Round')
axs[1].set_ylabel('Model Price (Tokens)')
axs[1].set_title('Model Price Evolution')
axs[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./figures/global_metrics.png', dpi=300, bbox_inches='tight')
```

### 🤝 **Client Contribution Patterns** (`wallet_contributions.png`)

**Content:**
- Individual client contribution scores over training rounds
- Comparative analysis of participant engagement
- Identification of high/low contributors

**Insights:**
- **Fairness Analysis**: Distribution of contributions across participants
- **Participation Patterns**: Consistent vs. sporadic contributors
- **System Balance**: Dominance detection and mitigation effectiveness

**Code Implementation:**
```python
plt.figure(figsize=(12, 8))

# Plot contribution trajectory for each client
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']

for i, (wid, metrics) in enumerate(wallets.items()):
    plt.plot(rounds, metrics['contribution'], 
             marker=markers[i], 
             color=colors[i],
             label=f'Client {wid}',
             linewidth=2,
             markersize=6)

plt.xlabel('Training Round')
plt.ylabel('Contribution Score')
plt.title('Client Contribution Patterns Over Training Rounds')
plt.legend(title='Participant ID', loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./figures/wallet_contributions.png', dpi=300, bbox_inches='tight')
```

### 💰 **Reward Distribution Analysis** (`wallet_rewards.png`)

**Content:**
- Token rewards earned by each client per round
- Cumulative reward accumulation
- Fairness of incentive distribution

**Insights:**
- **Economic Fairness**: Correlation between contribution and compensation
- **Participation Incentives**: Effectiveness of reward mechanism
- **Long-term Sustainability**: Token flow and accumulation patterns

**Code Implementation:**
```python
plt.figure(figsize=(12, 8))

# Plot reward distribution for each client
for i, (wid, metrics) in enumerate(wallets.items()):
    plt.plot(rounds, metrics['reward'],
             marker=markers[i],
             color=colors[i], 
             label=f'Client {wid}',
             linewidth=2,
             markersize=6)

plt.xlabel('Training Round')
plt.ylabel('Token Rewards')
plt.title('Token Reward Distribution Over Training Rounds')
plt.legend(title='Participant ID', loc='best')
plt.grid(True, alpha=0.3)

# Add cumulative rewards as secondary analysis
fig2, ax2 = plt.subplots(figsize=(10, 6))
for i, (wid, metrics) in enumerate(wallets.items()):
    cumulative_rewards = np.cumsum(metrics['reward'])
    ax2.plot(rounds, cumulative_rewards,
             marker=markers[i],
             color=colors[i],
             label=f'Client {wid}',
             linewidth=2)

ax2.set_xlabel('Training Round')
ax2.set_ylabel('Cumulative Token Rewards')
ax2.set_title('Cumulative Reward Accumulation')
ax2.legend(title='Participant ID')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./figures/wallet_rewards.png', dpi=300, bbox_inches='tight')
```

### 📉 **Local Loss Trajectories** (`wallet_local_loss.png`)

**Content:**
- Individual client model performance over rounds
- Local model improvement patterns
- Heterogeneity in learning progress

**Insights:**
- **Learning Heterogeneity**: Different convergence rates across clients
- **Data Quality Impact**: Correlation between data distribution and performance
- **Personalization Benefits**: Individual model adaptation effectiveness

**Code Implementation:**
```python
plt.figure(figsize=(12, 8))

# Plot local loss for each client
for i, (wid, metrics) in enumerate(wallets.items()):
    plt.plot(rounds, metrics['loss'],
             marker=markers[i],
             color=colors[i],
             label=f'Client {wid}',
             linewidth=2,
             markersize=6)

plt.xlabel('Training Round')
plt.ylabel('Local Model Loss')
plt.title('Local Model Performance Over Training Rounds')
plt.legend(title='Participant ID', loc='best')
plt.grid(True, alpha=0.3)

# Add global model loss for comparison
plt.plot(rounds, global_losses, 
         color='black',
         linestyle='--',
         linewidth=3,
         label='Global Model',
         alpha=0.7)

plt.legend(title='Model Type', loc='best')
plt.tight_layout()
plt.savefig('./figures/wallet_local_loss.png', dpi=300, bbox_inches='tight')
```

### 📈 **Performance Improvement Metrics**

#### **Delta Local Loss** (`delta_local_loss.png`)
**Content:**
- Round-to-round improvement in local model performance
- Momentum and consistency of learning progress

**Insights:**
- **Learning Momentum**: Sustained improvement vs. plateauing
- **Training Stability**: Consistent progress patterns
- **Optimization Effectiveness**: Local SGD performance

```python
plt.figure(figsize=(12, 8))

# Plot delta local loss (skip first round as no previous comparison)
for i, (wid, metrics) in enumerate(wallets.items()):
    delta_values = metrics['delta_local_loss'][1:]  # Skip first round
    rounds_delta = rounds[1:]
    
    plt.plot(rounds_delta, delta_values,
             marker=markers[i],
             color=colors[i],
             label=f'Client {wid}',
             linewidth=2,
             markersize=6)

plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Training Round')
plt.ylabel('Delta Local Loss')
plt.title('Round-to-Round Local Model Improvement')
plt.legend(title='Participant ID', loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./figures/delta_local_loss.png', dpi=300, bbox_inches='tight')
```

#### **Delta Gap Analysis** (`delta_gap.png`)
**Content:**
- Improvement over previous global model
- Client ability to enhance federated learning

**Insights:**
- **Contribution Quality**: Magnitude of improvements provided
- **Global Impact**: Individual client effect on system performance
- **Incentive Justification**: Basis for reward calculation

```python
plt.figure(figsize=(12, 8))

# Plot delta gap for all clients
for i, (wid, metrics) in enumerate(wallets.items()):
    plt.plot(rounds, metrics['delta_gap'],
             marker=markers[i],
             color=colors[i],
             label=f'Client {wid}',
             linewidth=2,
             markersize=6)

plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Training Round')
plt.ylabel('Delta Gap (Global Model Improvement)')
plt.title('Client Contribution to Global Model Performance')
plt.legend(title='Participant ID', loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./figures/delta_gap.png', dpi=300, bbox_inches='tight')
```

## Usage Instructions

### 🚀 **Generating Visualizations**

```bash
# After completing a training experiment
cd BPFL/
python plot.py

# Visualizations will be saved to ./figures/ directory
ls figures/
# Output: delta_gap.png, delta_local_loss.png, global_metrics.png, 
#         wallet_contributions.png, wallet_local_loss.png, wallet_rewards.png
```