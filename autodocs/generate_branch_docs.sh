#!/bin/bash

# BPFL Branch Documentation Generator
# Automatically creates experiment-specific README files for all branches

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 BPFL Branch Documentation Generator${NC}"
echo -e "${BLUE}=====================================${NC}"

# Get list of all experimental branches (exclude main and HEAD)
branches=$(git branch -r | grep -v 'HEAD' | grep -v 'main' | sed 's/origin\///' | tr -d ' ')

echo -e "${YELLOW}Found experimental branches:${NC}"
echo "$branches" | nl

# Confirm before proceeding
read -p $'\nDo you want to add README.md to all experimental branches? (y/n): ' -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}❌ Operation cancelled${NC}"
    exit 1
fi

# Track current branch to return to it later
current_branch=$(git branch --show-current)
echo -e "${BLUE}📍 Current branch: $current_branch${NC}"

# Counter for progress
total_branches=$(echo "$branches" | wc -l)
counter=0

# Process each branch
echo "$branches" | while read branch; do
    counter=$((counter + 1))
    echo -e "\n${YELLOW}📝 Processing branch $counter/$total_branches: $branch${NC}"
    
    # Skip if branch is empty or malformed
    if [ -z "$branch" ]; then
        continue
    fi
    
    # Checkout the branch
    if git checkout "$branch" 2>/dev/null; then
        echo -e "${GREEN}✅ Switched to $branch${NC}"
        
        # Parse branch name to extract components
        if [[ $branch =~ ^([^-]+)-([^-]+)-([^-]+)(-(.+))?$ ]]; then
            dataset="${BASH_REMATCH[1]}"
            architecture="${BASH_REMATCH[2]}"
            distribution="${BASH_REMATCH[3]}"
            variant="${BASH_REMATCH[5]}"
            
            # Determine experiment type
            if [[ $variant == "FL" ]]; then
                experiment_type="Traditional Federated Learning"
                gamma_value="N/A (Traditional FL)"
            elif [[ $variant == "0.95" ]]; then
                experiment_type="BPFL with High Personalization"
                gamma_value="0.95"
            else
                experiment_type="BPFL Standard"
                gamma_value="0.7"
            fi
            
            # Generate branch-specific README
            generate_branch_readme "$dataset" "$architecture" "$distribution" "$variant" "$experiment_type" "$gamma_value" "$branch"
            
            echo -e "${GREEN}✅ README.md created for $branch${NC}"
        else
            echo -e "${RED}❌ Could not parse branch name: $branch${NC}"
        fi
    else
        echo -e "${RED}❌ Failed to checkout $branch${NC}"
    fi
done

# Return to original branch
echo -e "\n${BLUE}🔄 Returning to original branch: $current_branch${NC}"
git checkout "$current_branch"

echo -e "\n${GREEN}🎉 Branch documentation generation complete!${NC}"
echo -e "${BLUE}📋 Next steps:${NC}"
echo -e "  1. Review generated README files in each branch"
echo -e "  2. Commit changes: ${YELLOW}git add README.md && git commit -m 'Add experiment-specific documentation'${NC}"
echo -e "  3. Push to remote: ${YELLOW}git push origin <branch-name>${NC}"

# Function to generate branch-specific README
generate_branch_readme() {
    local dataset=$1
    local architecture=$2
    local distribution=$3
    local variant=$4
    local experiment_type=$5
    local gamma_value=$6
    local branch_name=$7
    
    # Map dataset names to full descriptions
    case $dataset in
        "MNIST") dataset_full="MNIST (28x28 grayscale, 10 classes - handwritten digits)" ;;
        "FMNIST") dataset_full="Fashion-MNIST (28x28 grayscale, 10 classes - clothing items)" ;;
        "CIFAR10") dataset_full="CIFAR-10 (32x32 RGB, 10 classes - natural images)" ;;
        "CIFAR100") dataset_full="CIFAR-100 (32x32 RGB, 100 classes - natural images)" ;;
        *) dataset_full="$dataset" ;;
    esac
    
    # Map architecture names
    case $architecture in
        "LeNet") architecture_full="LeNet-5 CNN" ;;
        "ResNet18") architecture_full="ResNet-18 CNN" ;;
        *) architecture_full="$architecture" ;;
    esac
    
    # Map distribution schemes
    case $distribution in
        "Uniform") 
            distribution_full="Uniform (Equal data allocation across clients)"
            distribution_details="- **Client 0**: 25% of data\n- **Client 1**: 25% of data\n- **Client 2**: 25% of data\n- **Client 3**: 25% of data\n\nThis creates **homogeneous** conditions for baseline comparison."
            ;;
        "Linear") 
            distribution_full="Linear (1:2:3:4 ratio across clients)"
            distribution_details="- **Client 0**: 10% of data\n- **Client 1**: 20% of data\n- **Client 2**: 30% of data\n- **Client 3**: 40% of data\n\nThis creates **moderate heterogeneity** for intermediate analysis."
            ;;
        "Exponential") 
            distribution_full="Exponential (1:2:4:8 ratio across clients)"
            distribution_details="- **Client 0**: ~6.7% of data\n- **Client 1**: ~13.3% of data\n- **Client 2**: ~26.7% of data\n- **Client 3**: ~53.3% of data\n\nThis creates **high heterogeneity** to test robustness under extreme imbalance."
            ;;
        *) 
            distribution_full="$distribution"
            distribution_details="Custom distribution scheme."
            ;;
    esac
    
    # Generate README content
    cat > README.md << EOF
# $branch_name Experiment

This branch implements **$dataset** dataset training with **$architecture_full** architecture using **$distribution_full** in the BPFL framework.

## 🧪 Experiment Configuration

| Parameter | Value |
|-----------|-------|
| **Dataset** | $dataset_full |
| **Architecture** | $architecture_full |
| **Data Distribution** | $distribution_full |
| **Algorithm** | $experiment_type |
| **Gamma (γ)** | $gamma_value |
| **Clients** | 4 federated participants |
| **Rounds** | 20 training rounds |
| **Local Epochs** | 5 per round |

## 📊 Key Characteristics

### **Data Distribution ($distribution):**
$distribution_details

### **Architecture Details:**
EOF

    # Add architecture-specific details
    if [[ $architecture == "LeNet" ]]; then
        cat >> README.md << EOF
```python
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
**Optimized for**: 28×28 grayscale images (MNIST, Fashion-MNIST)
EOF
    else
        cat >> README.md << EOF
```python
class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  # Removed for small images
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
```
**Optimized for**: 32×32 RGB images (CIFAR-10, CIFAR-100)
EOF
    fi

    # Add algorithm-specific details
    cat >> README.md << EOF

### **Algorithm Configuration:**
EOF

    if [[ $variant == "FL" ]]; then
        cat >> README.md << EOF
- **Type**: Traditional Federated Learning (FedAvg)
- **Personalization**: None (pure global model)
- **Aggregation**: Simple weighted averaging
- **Incentives**: None (baseline comparison)
EOF
    else
        cat >> README.md << EOF
- **Type**: Blockchain-enabled Personalized Federated Learning (BPFL)
- **Personalization Strength (γ)**: $gamma_value
- **Contribution Assessment**: Loss improvement based
- **Reward Mechanism**: Token-based incentives
- **Aggregation**: Contribution-weighted with personalization
EOF
    fi

    # Add the rest of the README
    cat >> README.md << EOF

## 🚀 Running This Experiment

\`\`\`bash
# Ensure you're on the correct branch
git checkout $branch_name

# Start the BPFL system
python run.py

# Monitor training progress
tail -f logs/aggregator.txt

# Generate results visualization
python plot.py

# Stop the system
python stop.py
\`\`\`

## 📈 Expected Results

This configuration tests:
EOF

    # Add experiment-specific expectations
    case $dataset in
        "MNIST")
            echo "- **Digit recognition performance** on classic ML benchmark" >> README.md
            ;;
        "FMNIST")
            echo "- **Fashion item classification** with increased visual complexity" >> README.md
            ;;
        "CIFAR10")
            echo "- **Natural image classification** with RGB complexity" >> README.md
            ;;
        "CIFAR100")
            echo "- **Fine-grained classification** with 100 diverse categories" >> README.md
            ;;
    esac

    case $distribution in
        "Uniform")
            echo "- **Homogeneous learning** with equal data distribution" >> README.md
            ;;
        "Linear")
            echo "- **Moderate heterogeneity** effects on convergence" >> README.md
            ;;
        "Exponential")
            echo "- **Extreme heterogeneity** and personalization benefits" >> README.md
            ;;
    esac

    if [[ $variant == "FL" ]]; then
        echo "- **Traditional FL baseline** performance without personalization" >> README.md
    else
        echo "- **BPFL personalization** advantages over traditional FL" >> README.md
    fi

    cat >> README.md << EOF

## 📊 Results Analysis

After completion, check:
- \`nodes/results/res.json\` - Detailed metrics per round
- \`figures/\` - Generated visualization plots
- \`logs/\` - Training logs and system monitoring

Key metrics to analyze:
- Global model convergence rate
- Individual client performance improvements
- Contribution score fairness
- Token reward distribution effectiveness

## 🔗 Related Experiments

**Compare with:**
EOF

    # Add related experiment suggestions
    echo "- \`$dataset-$architecture-Uniform\` - Same setup with uniform distribution" >> README.md
    echo "- \`$dataset-$architecture-Linear\` - Same setup with linear distribution" >> README.md
    echo "- \`$dataset-$architecture-Exponential\` - Same setup with exponential distribution" >> README.md
    
    if [[ $variant != "FL" ]]; then
        echo "- \`$dataset-$architecture-$distribution-FL\` - Traditional FL baseline" >> README.md
    fi
    
    if [[ $variant != "0.95" ]]; then
        echo "- \`$dataset-$architecture-$distribution-0.95\` - Higher personalization (γ=0.95)" >> README.md
    fi

    cat >> README.md << EOF

## 📚 Full Documentation

For comprehensive documentation, system architecture, and methodology details, see the **main branch**:

\`\`\`bash
git checkout main
cat README.md  # Complete BPFL framework documentation
\`\`\`

## 🏷️ Branch Metadata

- **Branch Type**: Experimental Configuration
- **Created**: 2024 BPFL Research Study
- **Status**: ✅ Results Available
- **Purpose**: $dataset + $architecture + $distribution Analysis
- **Experiment ID**: $(echo $branch_name | tr '[:upper:]' '[:lower:]' | tr '-' '_')

---

*This is one of 38 experimental configurations in the BPFL research framework. Each branch represents a unique combination of dataset, architecture, and data distribution for comprehensive federated learning analysis.*
EOF

    echo -e "${GREEN}📄 Generated README.md for $branch_name${NC}"
}

# Export the function so it's available in the subshell
export -f generate_branch_readme
