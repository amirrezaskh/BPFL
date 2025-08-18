#!/usr/bin/env python3
"""
BPFL Documentation Generator with Immediate Commit
Creates README files for each branch and commits them immediately
"""

import subprocess
import re
import os
from typing import Dict, List

def run_git_command(command: List[str]) -> tuple[bool, str]:
    """Run a git command and return success status and output"""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return result.returncode == 0, result.stdout.strip()
    except Exception as e:
        return False, str(e)

def get_experimental_branches() -> List[str]:
    """Get list of all experimental branches"""
    success, output = run_git_command(['git', 'branch', '-r'])
    if not success:
        return []
    
    branches = []
    for line in output.split('\n'):
        branch = line.strip().replace('origin/', '')
        if branch and 'HEAD' not in branch and branch not in ['main', 'Centralized']:
            branches.append(branch)
    return branches

def parse_branch_name(branch: str) -> Dict[str, str]:
    """Parse branch name to extract experiment components"""
    pattern = r'^([^-]+)-([^-]+)-([^-]+)(?:-(.+))?$'
    match = re.match(pattern, branch)
    
    if not match:
        return {}
    
    dataset, architecture, distribution, variant = match.groups()
    
    if variant == "FL":
        experiment_type = "Traditional Federated Learning"
        gamma_value = "N/A (Traditional FL)"
    elif variant == "0.95":
        experiment_type = "BPFL with High Personalization"
        gamma_value = "0.95"
    else:
        experiment_type = "BPFL Standard"
        gamma_value = "0.7"
    
    return {
        'dataset': dataset,
        'architecture': architecture,
        'distribution': distribution,
        'variant': variant or '',
        'experiment_type': experiment_type,
        'gamma_value': gamma_value
    }

def get_dataset_info(dataset: str) -> Dict[str, str]:
    """Get dataset-specific information"""
    datasets = {
        'MNIST': {
            'full_name': 'MNIST (28x28 grayscale, 10 classes - handwritten digits)',
            'classes': 'Digits 0-9',
            'complexity': 'Low (classic ML benchmark)'
        },
        'FMNIST': {
            'full_name': 'Fashion-MNIST (28x28 grayscale, 10 classes - clothing items)',
            'classes': 'T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot',
            'complexity': 'Medium (fashion domain complexity)'
        },
        'CIFAR10': {
            'full_name': 'CIFAR-10 (32x32 RGB, 10 classes - natural images)',
            'classes': 'Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck',
            'complexity': 'High (natural image complexity)'
        },
        'CIFAR100': {
            'full_name': 'CIFAR-100 (32x32 RGB, 100 classes - natural images)',
            'classes': '100 fine-grained categories across 20 superclasses',
            'complexity': 'Very High (fine-grained classification)'
        }
    }
    return datasets.get(dataset, {'full_name': dataset, 'classes': 'Unknown', 'complexity': 'Unknown'})

def get_distribution_info(distribution: str) -> Dict[str, str]:
    """Get distribution scheme information"""
    distributions = {
        'Uniform': {
            'full_name': 'Uniform (Equal data allocation across clients)',
            'details': """- **Client 0**: 25% of data
- **Client 1**: 25% of data
- **Client 2**: 25% of data
- **Client 3**: 25% of data

This creates **homogeneous** conditions for baseline comparison.""",
            'heterogeneity': 'Low'
        },
        'Linear': {
            'full_name': 'Linear (1:2:3:4 ratio across clients)',
            'details': """- **Client 0**: 10% of data
- **Client 1**: 20% of data
- **Client 2**: 30% of data
- **Client 3**: 40% of data

This creates **moderate heterogeneity** for intermediate analysis.""",
            'heterogeneity': 'Medium'
        },
        'Exponential': {
            'full_name': 'Exponential (1:2:4:8 ratio across clients)',
            'details': """- **Client 0**: ~6.7% of data
- **Client 1**: ~13.3% of data
- **Client 2**: ~26.7% of data
- **Client 3**: ~53.3% of data

This creates **high heterogeneity** to test robustness under extreme imbalance.""",
            'heterogeneity': 'High'
        }
    }
    return distributions.get(distribution, {
        'full_name': distribution,
        'details': 'Custom distribution scheme.',
        'heterogeneity': 'Unknown'
    })

def generate_readme_content(branch: str, config: Dict[str, str]) -> str:
    """Generate README content for a specific branch"""
    
    dataset_info = get_dataset_info(config['dataset'])
    dist_info = get_distribution_info(config['distribution'])
    
    # Architecture code
    if config['architecture'] == 'LeNet':
        arch_code = """```python
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.ap = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.tanh = nn.Tanh()  # Original LeNet activation
```
**Optimized for**: 28×28 grayscale images (MNIST, Fashion-MNIST)"""
    else:  # ResNet18
        arch_code = """```python
class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  # Removed for small images
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
```
**Optimized for**: 32×32 RGB images (CIFAR-10, CIFAR-100)"""
    
    # Algorithm configuration
    if config['variant'] == 'FL':
        algo_config = """- **Type**: Traditional Federated Learning (FedAvg)
- **Personalization**: None (pure global model)
- **Aggregation**: Simple weighted averaging
- **Incentives**: None (baseline comparison)"""
    else:
        algo_config = f"""- **Type**: Blockchain-enabled Personalized Federated Learning (BPFL)
- **Personalization Strength (γ)**: {config['gamma_value']}
- **Contribution Assessment**: Loss improvement based
- **Reward Mechanism**: Token-based incentives
- **Aggregation**: Contribution-weighted with personalization"""
    
    # Expected results
    dataset_expectations = {
        'MNIST': "- **Digit recognition performance** on classic ML benchmark",
        'FMNIST': "- **Fashion item classification** with increased visual complexity",
        'CIFAR10': "- **Natural image classification** with RGB complexity",
        'CIFAR100': "- **Fine-grained classification** with 100 diverse categories"
    }
    
    dist_expectations = {
        'Uniform': "- **Homogeneous learning** with equal data distribution",
        'Linear': "- **Moderate heterogeneity** effects on convergence",
        'Exponential': "- **Extreme heterogeneity** and personalization benefits"
    }
    
    algo_expectations = ("- **Traditional FL baseline** performance without personalization" 
                        if config['variant'] == 'FL' 
                        else "- **BPFL personalization** advantages over traditional FL")
    
    # Related experiments
    base_name = f"{config['dataset']}-{config['architecture']}"
    related_experiments = [
        f"- `{base_name}-Uniform` - Same setup with uniform distribution",
        f"- `{base_name}-Linear` - Same setup with linear distribution", 
        f"- `{base_name}-Exponential` - Same setup with exponential distribution"
    ]
    
    if config['variant'] != 'FL':
        related_experiments.append(f"- `{base_name}-{config['distribution']}-FL` - Traditional FL baseline")
    
    if config['variant'] != '0.95':
        related_experiments.append(f"- `{base_name}-{config['distribution']}-0.95` - Higher personalization (γ=0.95)")
    
    readme_content = f"""# {branch} Experiment

This branch implements **{config['dataset']}** dataset training with **{config['architecture']}-5** architecture using **{dist_info['full_name']}** in the BPFL framework.

## 🧪 Experiment Configuration

| Parameter | Value |
|-----------|-------|
| **Dataset** | {dataset_info['full_name']} |
| **Architecture** | {config['architecture']}-5 CNN |
| **Data Distribution** | {dist_info['full_name']} |
| **Algorithm** | {config['experiment_type']} |
| **Gamma (γ)** | {config['gamma_value']} |
| **Clients** | 4 federated participants |
| **Rounds** | 20 training rounds |
| **Local Epochs** | 5 per round |

## 📊 Key Characteristics

### **Dataset Classes:**
{dataset_info['classes']}

### **Data Distribution ({config['distribution']}):**
{dist_info['details']}

### **Architecture Details:**
{arch_code}

### **Algorithm Configuration:**
{algo_config}

## 🚀 Running This Experiment

```bash
# Ensure you're on the correct branch
git checkout {branch}

# Start the BPFL system
python run.py

# Monitor training progress
tail -f logs/aggregator.txt

# Generate results visualization
python plot.py

# Stop the system
python stop.py
```

## 📈 Expected Results

This configuration tests:
{dataset_expectations.get(config['dataset'], f"- **{config['dataset']} performance** analysis")}
{dist_expectations.get(config['distribution'], f"- **{config['distribution']} distribution** effects")}
{algo_expectations}

## 📊 Results Analysis

After completion, check:
- `nodes/results/res.json` - Detailed metrics per round
- `figures/` - Generated visualization plots  
- `logs/` - Training logs and system monitoring

Key metrics to analyze:
- Global model convergence rate
- Individual client performance improvements
- Contribution score fairness
- Token reward distribution effectiveness

## 🔗 Related Experiments

**Compare with:**
{chr(10).join(related_experiments)}

## 📚 Full Documentation

For comprehensive documentation, system architecture, and methodology details, see the **main branch**:

```bash
git checkout main
cat README.md  # Complete BPFL framework documentation
```

## 🏷️ Branch Metadata

- **Branch Type**: Experimental Configuration
- **Created**: 2024 BPFL Research Study  
- **Status**: ✅ Results Available
- **Purpose**: {config['dataset']} + {config['architecture']} + {config['distribution']} Analysis
- **Experiment ID**: {branch.lower().replace('-', '_')}

---

*This is one of 38 experimental configurations in the BPFL research framework. Each branch represents a unique combination of dataset, architecture, and data distribution for comprehensive federated learning analysis.*"""
    
    return readme_content

def process_single_branch(branch: str) -> bool:
    """Process a single branch: create README and commit"""
    print(f"📝 Processing: {branch}")
    
    # Parse branch configuration
    config = parse_branch_name(branch)
    if not config:
        print(f"❌ Could not parse branch name: {branch}")
        return False
    
    # Checkout the branch
    success, _ = run_git_command(['git', 'checkout', branch])
    if not success:
        print(f"❌ Failed to checkout {branch}")
        return False
    
    print(f"✅ Switched to {branch}")
    
    # Generate README content
    readme_content = generate_readme_content(branch, config)
    
    # Write README file
    try:
        with open('README.md', 'w') as f:
            f.write(readme_content)
        print(f"✅ README.md created for {branch}")
    except Exception as e:
        print(f"❌ Failed to write README for {branch}: {e}")
        return False
    
    # Add and commit
    success, _ = run_git_command(['git', 'add', 'README.md'])
    if not success:
        print(f"❌ Failed to add README for {branch}")
        return False
    
    commit_msg = f"""Add experiment-specific documentation for {branch}

- Comprehensive experiment configuration details
- Dataset and architecture specifications  
- Data distribution analysis
- Running instructions and expected results
- Cross-references to related experiments

Generated by automated documentation system."""
    
    success, _ = run_git_command(['git', 'commit', '-m', commit_msg])
    if not success:
        print(f"ℹ️  No changes to commit for {branch}")
        return True  # This is OK, might already exist
    
    print(f"✅ Committed changes for {branch}")
    
    # Push to origin
    success, output = run_git_command(['git', 'push', 'origin', branch])
    if not success:
        print(f"❌ Failed to push {branch}: {output}")
        return False
    
    print(f"🚀 Pushed {branch}")
    return True

def main():
    """Main function"""
    print("🚀 BPFL One-by-One Documentation Generator")
    print("==========================================")
    
    # Get current branch
    success, original_branch = run_git_command(['git', 'branch', '--show-current'])
    if not success:
        print("❌ Could not determine current branch")
        return
    
    print(f"📍 Starting from: {original_branch}")
    
    # Get experimental branches
    branches = get_experimental_branches()
    if not branches:
        print("❌ No experimental branches found")
        return
    
    print(f"📝 Found {len(branches)} experimental branches")
    
    # Confirm
    response = input(f"\nGenerate and commit README.md for all {len(branches)} branches? (y/n): ")
    if response.lower() != 'y':
        print("❌ Operation cancelled")
        return
    
    print(f"\n🔄 Processing {len(branches)} branches...")
    
    # Process each branch
    success_count = 0
    failed_branches = []
    
    for i, branch in enumerate(branches, 1):
        print(f"\n[{i}/{len(branches)}] {branch}")
        
        if process_single_branch(branch):
            success_count += 1
        else:
            failed_branches.append(branch)
    
    # Return to original branch
    print(f"\n🔄 Returning to: {original_branch}")
    run_git_command(['git', 'checkout', original_branch])
    
    # Summary
    print(f"\n🎉 Documentation process complete!")
    print(f"=" * 40)
    print(f"✅ Successfully processed: {success_count}/{len(branches)} branches")
    
    if failed_branches:
        print(f"❌ Failed branches: {', '.join(failed_branches)}")
    else:
        print(f"🎊 All branches processed successfully!")
    
    print(f"\n📊 Your BPFL research framework is now fully documented!")

if __name__ == "__main__":
    main()
