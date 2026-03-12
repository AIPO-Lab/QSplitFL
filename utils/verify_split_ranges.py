
import sys
import os
import torch
import torch.nn as nn

# Add paths to sys.path to import from subdirectories
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'CIFAR10'))
sys.path.append(os.path.join(os.getcwd(), 'FMNIST'))
sys.path.append(os.path.join(os.getcwd(), 'MNIST'))
sys.path.append(os.path.join(os.getcwd(), 'CIFAR-100'))

# ... (previous code)

def log(msg):
    print(msg)
    with open("verification.log", "a") as f:
        f.write(msg + "\n")

# Clear log
with open("verification.log", "w") as f:
    f.write("Starting Verification\n")

def test_resnet50_splits(model_cls, model_name):
    log(f"\nTesting {model_name} (ResNet50, Split Range: 25-49)...")
    try:
        model = model_cls(num_classes=10)
    except:
        model = model_cls(num_classes=10, input_channels=3)
        
    splits = [25, 30, 40, 49]
    dummy_input = torch.randn(1, 3, 32, 32)
    
    for split in splits:
        msg = f"  Testing Split {split}..."
        try:
            model.configure_split('client', split)
            client_out = model(dummy_input)
            model.configure_split('server', split)
            server_out = model(client_out)
            
            if server_out.shape != (1, 10):
                log(msg + f" FAIL: Shape {server_out.shape}")
            else:
                log(msg + " OK")
        except Exception as e:
            log(msg + f" CRASH: {e}")

def test_mobilenetv4_splits(model_cls, model_name):
    log(f"\nTesting {model_name} (MobileNetV4, Split Range: 27-52)...")
    try:
        model = model_cls(num_classes=10)
    except:
        model = model_cls() 
        
    splits = [27, 30, 40, 52]
    dummy_input = torch.randn(1, 3, 32, 32)
    
    for split in splits:
        msg = f"  Testing Split {split}..."
        try:
            model.configure_split('client', split)
            client_out = model(dummy_input)
            model.configure_split('server', split)
            server_out = model(client_out)
            
            if server_out.shape != (1, 10):
                log(msg + f" FAIL: Shape {server_out.shape}")
            else:
                log(msg + " OK")
        except Exception as e:
            log(msg + f" CRASH: {e}")

def test_convnext_splits(model_cls, model_name):
    log(f"\nTesting {model_name} (ConvNeXt, Split Range: 30-58)...")
    try:
        model = model_cls(num_classes=10)
    except:
        model = model_cls()
        
    splits = [30, 37, 45, 50, 58]
    dummy_input = torch.randn(1, 3, 32, 32)
    
    for split in splits:
        msg = f"  Testing Split {split}..."
        try:
            model.configure_split('client', split)
            client_out = model(dummy_input)
            model.configure_split('server', split)
            server_out = model(client_out)
            
            if server_out.shape != (1, 10):
                log(msg + f" FAIL: Shape {server_out.shape}")
            else:
                log(msg + " OK")
        except Exception as e:
            log(msg + f" CRASH: {e}")

try:
    # 1. Root Models
    import resnet50_model
    import mobilenetv4_model
    import convnext_model
    
    test_resnet50_splits(resnet50_model.ResNet50, "Root ResNet50")
    test_mobilenetv4_splits(mobilenetv4_model.MobileNetV4, "Root MobileNetV4")
    test_convnext_splits(convnext_model.ConvNeXt, "Root ConvNeXt")
    
    log("\n--- Testing CIFAR10 Models ---")
    import CIFAR10.models as c10_models
    test_resnet50_splits(c10_models.ResNet50, "CIFAR10 ResNet50")
    test_mobilenetv4_splits(c10_models.MobileNetV4, "CIFAR10 MobileNetV4")
    test_convnext_splits(c10_models.ConvNeXt, "CIFAR10 ConvNeXt")
    
    log("\n--- Testing FMNIST Models ---")
    import FMNIST.models as fm_models
    def test_wrapper_fmnist(cls, name):
        log(f"\nTesting {name}...")
        try:
             model = cls(num_classes=10, input_channels=1)
             dummy = torch.randn(1, 1, 28, 28)
             model.configure_split('client', 30)
             out = model(dummy)
             model.configure_split('server', 30)
             final = model(out)
             log(" FMNIST Custom OK")
        except Exception as e:
            log(f" FMNIST Init/Run Error: {e}")

    test_wrapper_fmnist(fm_models.ResNet50, "FMNIST ResNet50")
    
    log("\n--- Testing CIFAR-100 Models ---")
    import importlib
    c100 = importlib.import_module("CIFAR-100.models")
    test_resnet50_splits(c100.ResNet50, "CIFAR-100 ResNet50")
    test_mobilenetv4_splits(c100.MobileNetV4, "CIFAR-100 MobileNetV4")
    test_convnext_splits(c100.ConvNeXt, "CIFAR-100 ConvNeXt")

except Exception as e:
    log(f"\nGlobal Crash: {e}")
    import traceback
    traceback.print_exc()
