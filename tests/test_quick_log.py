"""
Fast test to check server logging output.
Runs MNIST with CNN (Model 1), 1 Client, 1 Round.
Prints output to stdout.
"""

import subprocess
import sys
import os

print("=" * 80)
print("FAST LOGGING CHECK")
print("Target: Verify '[Client X] Epoch Y/Z: Loss=..., Accuracy=...%' output")
print("=" * 80)

# Model=1 (CNN), Clients=1, Rounds=1
input_data = "1\n1\n1\n"

process = subprocess.Popen(
    ['python', 'main_complete_rl.py'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd='MNIST',
    text=True,
    bufsize=1,
    encoding='utf-8',
    errors='replace'
)

print("[INFO] Process started...")

try:
    stdout, stderr = process.communicate(input=input_data, timeout=60)
    
    print("\n--- OUTPUT START ---")
    print(stdout)
    print("--- OUTPUT END ---\n")
    
    if stderr:
        print("\n--- ERRORS ---")
        print(stderr)
        
    # Check for target string pattern
    if "Accuracy=" in stdout and "%" in stdout and "[Client" in stdout:
        print("\n✅ LOGGING VERIFIED: Found enhanced log pattern.")
    else:
        print("\n❌ LOGGING FAILED: Could not find enhanced log pattern.")

except subprocess.TimeoutExpired:
    process.kill()
    print("\n❌ TIMEOUT: Process took too long.")
    
print("-" * 80)
