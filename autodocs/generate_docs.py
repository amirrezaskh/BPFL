#!/usr/bin/env python3
"""
BPFL Branch Documentation Generator
Automatically creates experiment-specific README files for all branches
"""

import subprocess
import re
from typing import Dict, List, Tuple

def get_experimental_branches() -> List[str]:
    """Get list of all experimental branches"""
    result = subprocess.run(['git', 'branch', '-r'], capture_output=True, text=True)
    branches = []
    for line in result.stdout.strip().split('\n'):
        branch = line.strip().replace('origin/', '')
        if branch and 'HEAD' not in branch and branch != 'main':
            branches.append(branch)
    return branches

def parse_branch_name(branch: str) -> Dict[str, str]:
    """Parse branch name to extract experiment components"""
    # Pattern: DATASET-ARCHITECTURE-DISTRIBUTION(-VARIANT)?
    pattern = r'^([^-]+)-([^-]+)-([^-]+)(?:-(.+))?$'
    match = re.match(pattern, branch)
    
    if not match:
        return {}
    
    dataset, architecture, distribution, variant = match.groups()
    
    # Determine experiment type and gamma
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

def get_architecture_code(architecture: str) -> str:
    """Get architecture-specific code and description"""
    if architecture == 'LeNet':
        return """```python
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
        return """```python
class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  # Removed for small images
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
```
**Optimized for**: 32×32 RGB images (CIFAR-10, CIFAR-100)"""

def generate_readme_content(branch: str, config: Dict[str, str]) -> str:
    """Generate README content for a specific branch"""
    
    dataset_info = get_dataset_info(config['dataset'])
    dist_info = get_distribution_info(config['distribution'])
    arch_code = get_architecture_code(config['architecture'])
    
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

def main():
    """Main function to generate documentation for all branches"""
    print("🚀 BPFL Branch Documentation Generator")
    print("=====================================")
    
    # Get current branch to return to later
    current_branch = subprocess.run(['git', 'branch', '--show-current'], 
                                   capture_output=True, text=True).stdout.strip()
    
    # Get all experimental branches
    branches = get_experimental_branches()
    print(f"📝 Found {len(branches)} experimental branches")
    
    # Confirm before proceeding
    response = input(f"\nDo you want to add README.md to all {len(branches)} experimental branches? (y/n): ")
    if response.lower() != 'y':
        print("❌ Operation cancelled")
        return
    
    success_count = 0
    failed_branches = []
    
    for i, branch in enumerate(branches, 1):
        print(f"\n📝 Processing branch {i}/{len(branches)}: {branch}")
        
        try:
            # Parse branch configuration
            config = parse_branch_name(branch)
            if not config:
                print(f"❌ Could not parse branch name: {branch}")
                failed_branches.append(branch)
                continue
            
            # Checkout the branch
            result = subprocess.run(['git', 'checkout', branch], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Failed to checkout {branch}")
                failed_branches.append(branch)
                continue
            
            print(f"✅ Switched to {branch}")
            
            # Generate README content
            readme_content = generate_readme_content(branch, config)
            
            # Write README file
            with open('README.md', 'w') as f:
                f.write(readme_content)
            
            print(f"✅ README.md created for {branch}")
            success_count += 1
            
        except Exception as e:
            print(f"❌ Error processing {branch}: {e}")
            failed_branches.append(branch)
    
    # Return to original branch
    print(f"\n🔄 Returning to original branch: {current_branch}")
    subprocess.run(['git', 'checkout', current_branch])
    
    # Summary
    print(f"\n🎉 Documentation generation complete!")
    print(f"✅ Successfully processed: {success_count}/{len(branches)} branches")
    
    if failed_branches:
        print(f"❌ Failed branches: {', '.join(failed_branches)}")
    
    print(f"\n📋 Next steps:")
    print(f"  1. Review generated README files in each branch")
    print(f"  2. Commit changes to each branch:")
    print(f"     for branch in {' '.join(branches[:3])}...; do")
    print(f"       git checkout $branch")
    print(f"       git add README.md")
    print(f"       git commit -m 'Add experiment-specific documentation'")
    print(f"       git push origin $branch")
    print(f"     done")

if __name__ == "__main__":
    main()
