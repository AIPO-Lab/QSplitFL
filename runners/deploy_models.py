"""
Unified deployment script to integrate all 4 model architectures 
(CNN, ResNet50, MobileNetV4, ConvNeXt) into all dataset folders
"""

import os
import shutil

# Read the standalone model files
with open('resnet50_model.py', 'r', encoding='utf-8') as f:
    resnet50_code = f.read()

with open('mobilenetv4_model.py', 'r', encoding='utf-8') as f:
    mobilenetv4_code = f.read()

with open('convnext_model.py', 'r', encoding='utf-8') as f:
    convnext_code = f.read()

# Extract just the class definitions (remove test code)
def extract_classes(code):
    """Extract only class definitions from code"""
    lines = code.split('\n')
    result = []
    in_main = False
    
    for line in lines:
        if 'if __name__ == "__main__"' in line:
            in_main = True
        if not in_main:
            result.append(line)
    
    return '\n'.join(result)

resnet50_classes = extract_classes(resnet50_code)
mobilenetv4_classes = extract_classes(mobilenetv4_code)
convnext_classes = extract_classes(convnext_code)

# Dataset configurations
datasets = {
    'MNIST': {'input_channels': 1, 'num_classes': 10, 'cnn_class': 'SimpleMNISTCNN'},
    'FMNIST': {'input_channels': 1, 'num_classes': 10, 'cnn_class': 'SimpleMNISTCNN'},
    'CIFAR10': {'input_channels': 3, 'num_classes': 10, 'cnn_class': 'ResNetFed'},
    'CIFAR-100': {'input_channels': 3, 'num_classes': 100, 'cnn_class': 'ResNetFed'}
}

print("=" * 80)
print("DEPLOYING MULTI-MODEL ARCHITECTURE TO ALL DATASETS")
print("=" * 80)

for dataset, config in datasets.items():
    print(f"\n[{dataset}] Updating models.py...")
    
    dataset_path = f"./{dataset}"
    models_file = f"{dataset_path}/models.py"
    
    # Backup existing models.py
    if os.path.exists(models_file):
        shutil.copy(models_file, f"{models_file}.backup")
        print(f"  - Backed up existing models.py")
    
    # Create unified models.py with all 4 architectures
    unified_code = f'''"""
Neural Network Models for Federated Split Learning - {dataset}

This module contains all available CNN architectures:
1. CNN - {config['cnn_class']} (original/baseline)
2. ResNet50 - Deep residual network with bottleneck blocks
3. MobileNetV4 - Efficient mobile architecture
4. ConvNeXt - Modern CNN inspired by vision transformers

All models support split learning with configurable split points.
Input: {config['input_channels']} channels, Output: {config['num_classes']} classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    
    # Add existing CNN classes (read from current models.py)
    if os.path.exists(models_file):
        with open(models_file, 'r', encoding='utf-8') as f:
            existing_code = f.read()
            # Extract existing classes (ResNetFed, SimpleMNISTCNN, etc.)
            unified_code += "\n# ============================================================================\n"
            unified_code += f"# Original CNN Architecture for {dataset}\n"
            unified_code += "# ============================================================================\n\n"
            
            # Find and extract existing classes
            import re
            class_pattern = r'(class \w+\(nn\.Module\):.*?)(?=\nclass |\nif __name__|$)'
            matches = re.findall(class_pattern, existing_code, re.DOTALL)
            for match in matches:
                unified_code += match + "\n\n"
    
    # Add ResNet50
    unified_code += "\n# ============================================================================\n"
    unified_code += "# ResNet50 Architecture\n"
    unified_code += "# ============================================================================\n\n"
    unified_code += resnet50_classes + "\n\n"
    
    # Add MobileNetV4
    unified_code += "# ============================================================================\n"
    unified_code += "# MobileNetV4 Architecture\n"
    unified_code += "# ============================================================================\n\n"
    unified_code += mobilenetv4_classes + "\n\n"
    
    # Add ConvNeXt
    unified_code += "# ============================================================================\n"
    unified_code += "# ConvNeXt Architecture\n"
    unified_code += "# ============================================================================\n\n"
    unified_code += convnext_classes + "\n\n"
    
    # Add model selection helper
    unified_code += '''
# ============================================================================
# Model Selection Helper
# ============================================================================

def get_model_by_choice(choice, num_classes, input_channels):
    """
    Get model instance based on user choice.
    
    Args:
        choice: Model number (1-4)
        num_classes: Number of output classes
        input_channels: Number of input channels (1 for grayscale, 3 for RGB)
    
    Returns:
        model: Selected model instance
        model_name: Name of the model
    """
    if choice == 1:
        # CNN (baseline)
'''
    
    if config['cnn_class'] == 'SimpleMNISTCNN':
        unified_code += f'''        model = SimpleMNISTCNN(num_classes=num_classes)
        return model, "CNN"
'''
    else:
        unified_code += f'''        model = ResNetFed(num_classes=num_classes)
        return model, "CNN"
'''
    
    unified_code += '''    elif choice == 2:
        # ResNet50
        model = ResNet50(num_classes=num_classes, input_channels=input_channels)
        return model, "ResNet50"
    elif choice == 3:
        # MobileNetV4
        model = MobileNetV4(num_classes=num_classes, input_channels=input_channels)
        return model, "MobileNetV4"
    elif choice == 4:
        # ConvNeXt
        model = ConvNeXt(num_classes=num_classes, input_channels=input_channels)
        return model, "ConvNeXt"
    else:
        raise ValueError(f"Invalid model choice: {choice}. Must be 1-4.")
'''
    
    # Write the unified models.py
    with open(models_file, 'w', encoding='utf-8') as f:
        f.write(unified_code)
    
    print(f"  ✓ Updated {dataset}/models.py with all 4 architectures")

print("\n" + "=" * 80)
print("✓ All dataset models.py files updated!")
print("=" * 80)
print("\nNext: Update main_complete_rl.py files to add model selection menu")
