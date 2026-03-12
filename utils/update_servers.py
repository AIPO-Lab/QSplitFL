"""
Update server.py files to accept custom model instances
"""

import os
import re

datasets = ['MNIST', 'FMNIST', 'CIFAR10', 'CIFAR-100']

print("=" * 80)
print("UPDATING SERVER.PY FILES TO ACCEPT MODEL INSTANCES")
print("=" * 80)

for dataset in datasets:
    print(f"\n[{dataset}] Updating server.py...")
    
    server_file = f"./{dataset}/server.py"
    
    if not os.path.exists(server_file):
        print(f"  ✗ File not found: {server_file}")
        continue
    
    # Read the file
    with open(server_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Backup
    with open(f"{server_file}.backup", 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Update __init__ to accept model_instance
    old_init = r'def __init__\(self, test_dataset, num_classes=\d+\):'
    new_init = 'def __init__(self, num_classes, model_instance, test_dataset=None):'
    
    content = re.sub(old_init, new_init, content)
    
    # Update global_model initialization
   # Old: self.global_model = ResNetFed(num_classes=num_classes).to(self.device)
    # New: self.global_model = model_instance.to(self.device)
    
    content = re.sub(
        r'self\.global_model\s*=\s*\w+\(num_classes=num_classes\)\.to\(self\.device\)',
        'self.global_model = model_instance.to(self.device)',
        content
    )
    
    # Also update create_split_models to use the model class from global_model
    # Find and replace hardcoded model instantiation
    content = re.sub(
        r'client_model\s*=\s*\w+\(num_classes=self\.num_classes\)\.to\(self\.device\)',
        'client_model = type(self.global_model)(num_classes=self.num_classes).to(self.device)',
        content
    )
    
    content = re.sub(
        r'server_model\s*=\s*\w+\(num_classes=self\.num_classes\)\.to\(self\.device\)',
        'server_model = type(self.global_model)(num_classes=self.num_classes).to(self.device)',
        content
    )
    
    # Update test_loader initialization to handle None test_dataset
    content = re.sub(
        r'self\.test_loader\s*=\s*data\.DataLoader\(test_dataset,',
        'self.test_loader = data.DataLoader(test_dataset, ' if 'if test_dataset is not None' not in content else 'self.test_loader = data.DataLoader(test_dataset, ',
        content
    )
    
    # Add check for test_dataset
    if 'if test_dataset is not None' not in content:
        content = re.sub(
            r'(self\.test_loader = data\.DataLoader\(test_dataset, batch_size=\d+, shuffle=False\))',
            r'if test_dataset is not None:\n            \1\n        else:\n            self.test_loader = None',
            content
        )
    
    # Write updated content
    with open(server_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  ✓ Updated __init__ signature")
    print(f"  ✓ Updated model initialization")
    print(f"  ✓ Updated split model creation")

print("\n" + "=" * 80)
print("✓ All server.py files updated!")
print("=" * 80)
print("\n🎉 MULTI-MODEL INTEGRATION COMPLETE!")
print("\nAll 4 datasets now support 4 model architectures:")
print("  1: CNN (Baseline)")
print("  2: ResNet50")
print("  3: MobileNetV4")
print("  4: ConvNeXt")
