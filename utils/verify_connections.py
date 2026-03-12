"""
Verification script to test all dataset connections and model availability
Tests that run_sequential_experiments.py can access all models in all datasets
"""

import os
import sys

print("=" * 80)
print("VERIFYING DATASET CONNECTIONS AND MODEL AVAILABILITY")
print("=" * 80)

datasets = {
    'MNIST': {'folder': 'MNIST', 'input_channels': 1, 'num_classes': 10},
    'FMNIST': {'folder': 'FMNIST', 'input_channels': 1, 'num_classes': 10},
    'CIFAR10': {'folder': 'CIFAR10', 'input_channels': 3, 'num_classes': 10},
    'CIFAR-100': {'folder': 'CIFAR-100', 'input_channels': 3, 'num_classes': 100}
}

models = {
    1: 'CNN',
    2: 'ResNet50',
    3: 'MobileNetV4',
    4: 'ConvNeXt'
}

print("\n[1] Checking Dataset Folders...")
print("-" * 80)
all_folders_exist = True
for name, config in datasets.items():
    folder = config['folder']
    exists = os.path.exists(folder)
    main_exists = os.path.exists(f"{folder}/main_complete_rl.py")
    models_exists = os.path.exists(f"{folder}/models.py")
    server_exists = os.path.exists(f"{folder}/server.py")
    
    if exists and main_exists and models_exists and server_exists:
        print(f"✓ {name:12} - Folder: ✓  main_complete_rl.py: ✓  models.py: ✓  server.py: ✓")
    else:
        print(f"✗ {name:12} - Missing files!")
        all_folders_exist = False

print("\n[2] Checking Model Availability in Each Dataset...")
print("-" * 80)

all_models_available = True
for dataset_name, config in datasets.items():
    folder = config['folder']
    print(f"\n{dataset_name} ({folder}):")
    
    # Add folder to path temporarily
    sys.path.insert(0, folder)
    
    try:
        # Import models module
        import importlib
        if 'models' in sys.modules:
            del sys.modules['models']
        models_module = importlib.import_module('models')
        
        # Test get_model_by_choice function
        if hasattr(models_module, 'get_model_by_choice'):
            print("  ✓ get_model_by_choice function exists")
            
            # Test each model
            for model_id, model_name in models.items():
                try:
                    model_instance, display_name = models_module.get_model_by_choice(
                        model_id, 
                        config['num_classes'],
                        config['input_channels']
                    )
                    print(f"    ✓ Model {model_id} ({model_name}): {type(model_instance).__name__}")
                except Exception as e:
                    print(f"    ✗ Model {model_id} ({model_name}): ERROR - {e}")
                    all_models_available = False
        else:
            print("  ✗ get_model_by_choice function NOT FOUND")
            all_models_available = False
            
    except Exception as e:
        print(f"  ✗ Error importing models: {e}")
        all_models_available = False
    finally:
        # Clean up
        if folder in sys.path:
            sys.path.remove(folder)

print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)

if all_folders_exist and all_models_available:
    print("✅ ALL CHECKS PASSED!")
    print("\nrun_sequential_experiments.py can successfully:")
    print("  ✓ Connect to all 4 dataset folders")
    print("  ✓ Access all 4 models in each dataset")
    print("  ✓ CNN, ResNet50, MobileNetV4, ConvNeXt")
    print("\n🚀 System is ready for full experiments!")
else:
    print("❌ VERIFICATION FAILED!")
    if not all_folders_exist:
        print("  ✗ Some dataset folders or files are missing")
    if not all_models_available:
        print("  ✗ Some models are not available")

print("=" * 80)
