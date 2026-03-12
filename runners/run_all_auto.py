"""
Automated script to run the full sequential experiment suite
Configuration: ResNet50, 5 Clients, 10 Rounds per dataset
Estimated Time: ~8.2 hours
"""
import subprocess
import sys
import time

# Configuration Input Sequence:
# 2   (Model: ResNet50)
# 5   (MNIST Clients)
# 10  (MNIST Rounds)
# 5   (FMNIST Clients)
# 10  (FMNIST Rounds)
# 5   (CIFAR-10 Clients)
# 10  (CIFAR-10 Rounds)
# 5   (CIFAR-100 Clients)
# 10  (CIFAR-100 Rounds)
inputs = "2\n5\n10\n5\n10\n5\n10\n5\n10\n"

print("=" * 80)
print("AUTOMATED EXPERIMENT LAUNCHER")
print("Target: Run ResNet50 across all datasets")
print("=" * 80)
print("Configuration:")
print("  Model:      ResNet50")
print("  MNIST:      5 Clients, 10 Rounds")
print("  FMNIST:     5 Clients, 10 Rounds")
print("  CIFAR-10:   5 Clients, 10 Rounds")
print("  CIFAR-100:  5 Clients, 10 Rounds")
print("-" * 80)
print("Injecting configuration inputs...")

# We use subprocess to run the main script and pipe the inputs
try:
    # Use unbuffered output (-u) to ensure we see logs in real-time
    process = subprocess.Popen(
        ['python', '-u', 'run_sequential_experiments.py'],
        stdin=subprocess.PIPE,
        stdout=sys.stdout, 
        stderr=sys.stderr,
        text=True,
        bufsize=1
    )
    
    # Send all inputs at once (the script buffers them until requested)
    process.communicate(input=inputs)
    
except KeyboardInterrupt:
    print("\n\nExperiment interrupted by user.")
except Exception as e:
    print(f"\n\nError occurred: {e}")
