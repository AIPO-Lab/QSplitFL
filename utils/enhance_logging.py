"""
Enhance server.py to explicitly print per-client loss and accuracy
for each round during training.
"""
import os
import re

datasets = ['MNIST', 'FMNIST', 'CIFAR10', 'CIFAR-100']

print("=" * 80)
print("ENHANCING SERVER LOGGING")
print("Target: Print Client Loss and Accuracy per Round")
print("=" * 80)

for dataset in datasets:
    server_file = f"./{dataset}/server.py"
    
    if not os.path.exists(server_file):
        continue
    
    print(f"[{dataset}] Updating {server_file}...")
    
    with open(server_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # We want to ensure this print statement exists and is formatted nicely:
    # print(f"  Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")
    
    # Check if it already exists (it seems to exist in the viewed file)
    # The user says "it should show...". Maybe it's not showing because stdout is buffered or something?
    # Or maybe they want it more prominent?
    
    # Let's make it clearer:
    # Client 1: Loss 0.1234, Acc 0.98.
    
    # Search for the print statement
    if 'print(f"  Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")' in content:
        # Replace it with something better
        new_print = '                print(f"    [Client {client.client_id}] Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc*100:.2f}%")'
        
        content = content.replace(
            'print(f"  Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")',
            new_print
        )
        print("  ✓ Updated logging format")
        
        with open(server_file, 'w', encoding='utf-8') as f:
            f.write(content)
    else:
        print("  ⚠ Could not find standard print statement to replace")

print("\n" + "=" * 80)
print("✓ LOGGING ENHANCEMENT COMPLETE")
print("=" * 80)
