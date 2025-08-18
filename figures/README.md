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

### 📊 **Customizing Plots**

#### **Modify Plot Styling:**
```python
# Change color scheme
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

# Adjust figure dimensions
plt.rcParams['figure.figsize'] = (12, 8)

# Change line styles
linestyles = ['-', '--', '-.', ':']
```

#### **Add Statistical Analysis:**
```python
# Add trend lines
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(rounds, global_losses)
trend_line = slope * np.array(rounds) + intercept
plt.plot(rounds, trend_line, 'r--', alpha=0.8, label=f'Trend (R²={r_value**2:.3f})')

# Add confidence intervals
mean_contribution = np.mean([metrics['contribution'] for metrics in wallets.values()], axis=0)
std_contribution = np.std([metrics['contribution'] for metrics in wallets.values()], axis=0)
plt.fill_between(rounds, mean_contribution - std_contribution, 
                 mean_contribution + std_contribution, alpha=0.2)
```

#### **Export Data for External Analysis:**
```python
# Export to CSV for external analysis
import pandas as pd

# Create DataFrame from results
df_data = []
for round_data in data:
    for submit in round_data['submits']:
        df_data.append({
            'round': round_data['round'],
            'global_loss': round_data['g_model_loss'],
            'model_price': round_data['new_model_price'],
            'wallet_id': submit['walletId'],
            'local_loss': submit['loss'],
            'contribution': submit['contribution'],
            'reward': submit['reward']
        })

df = pd.DataFrame(df_data)
df.to_csv('./results/analysis_data.csv', index=False)
```

## Comparative Analysis

### 🔍 **Cross-Experiment Comparison**

```python
def compare_experiments(experiment_paths):
    """Compare multiple BPFL experiments"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for exp_path in experiment_paths:
        with open(f'{exp_path}/nodes/results/res.json') as f:
            data = json.load(f)
        
        # Extract experiment name from path
        exp_name = exp_path.split('/')[-1]
        
        # Plot global loss comparison
        rounds = [r['round'] for r in data]
        global_losses = [r['g_model_loss'] for r in data]
        axes[0,0].plot(rounds, global_losses, label=exp_name, linewidth=2)
    
    axes[0,0].set_title('Global Loss Comparison Across Experiments')
    axes[0,0].set_xlabel('Training Round')
    axes[0,0].set_ylabel('Global Model Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./figures/experiment_comparison.png', dpi=300, bbox_inches='tight')
```

### 📈 **Performance Metrics Summary**

```python
def generate_summary_statistics(data):
    """Generate comprehensive performance summary"""
    
    # Extract key metrics
    final_global_loss = data[-1]['g_model_loss']
    initial_global_loss = data[0]['g_model_loss']
    improvement = (initial_global_loss - final_global_loss) / initial_global_loss * 100
    
    # Calculate fairness metrics
    all_rewards = []
    for round_data in data:
        round_rewards = [submit['reward'] for submit in round_data['submits']]
        all_rewards.extend(round_rewards)
    
    reward_gini = calculate_gini_coefficient(all_rewards)
    
    # Generate summary report
    summary = {
        'Final Global Loss': final_global_loss,
        'Total Improvement (%)': improvement,
        'Convergence Rounds': len(data),
        'Reward Fairness (Gini)': reward_gini,
        'Average Round Reward': np.mean(all_rewards),
        'Price Stability': np.std([r['new_model_price'] for r in data])
    }
    
    return summary
```

## Advanced Analytics

### 🧮 **Statistical Analysis**

```python
# Correlation analysis between contributions and rewards
from scipy.stats import pearsonr

all_contributions = []
all_rewards = []

for round_data in data:
    for submit in round_data['submits']:
        all_contributions.append(submit['contribution'])
        all_rewards.append(submit['reward'])

correlation, p_value = pearsonr(all_contributions, all_rewards)
print(f"Contribution-Reward Correlation: {correlation:.3f} (p={p_value:.3f})")

# Fairness analysis using Gini coefficient
def gini_coefficient(rewards):
    """Calculate Gini coefficient for reward distribution"""
    sorted_rewards = np.sort(rewards)
    n = len(sorted_rewards)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_rewards)) / (n * np.sum(sorted_rewards)) - (n + 1) / n

gini = gini_coefficient(all_rewards)
print(f"Reward Distribution Gini Coefficient: {gini:.3f}")
```

### 📊 **Performance Benchmarking**

```python
def benchmark_against_centralized(centralized_results, federated_results):
    """Compare BPFL performance against centralized baseline"""
    
    plt.figure(figsize=(10, 6))
    
    # Plot centralized performance
    plt.plot(centralized_results['rounds'], centralized_results['losses'], 
             label='Centralized Training', linewidth=3, color='red', alpha=0.8)
    
    # Plot federated performance
    plt.plot(federated_results['rounds'], federated_results['global_losses'],
             label='BPFL (Global)', linewidth=3, color='blue', alpha=0.8)
    
    plt.xlabel('Training Epoch/Round')
    plt.ylabel('Model Loss')
    plt.title('BPFL vs. Centralized Training Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Calculate performance gap
    final_gap = federated_results['global_losses'][-1] - centralized_results['losses'][-1]
    plt.text(0.7, 0.9, f'Final Performance Gap: {final_gap:.4f}', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('./figures/centralized_comparison.png', dpi=300, bbox_inches='tight')
```

---

The visualization system provides comprehensive insights into BPFL performance, enabling researchers to analyze training dynamics, contribution patterns, reward fairness, and system-wide behavior across different experimental configurations.
