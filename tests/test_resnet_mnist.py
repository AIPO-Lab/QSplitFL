"""
Specific test for ResNet50 on MNIST to verify the fix for state_dict loading error.
Writes output to test_resnet_result.txt (No Unicode)
"""

import subprocess
import sys
from datetime import datetime
import os

def log(msg):
    print(msg)
    # Use utf-8 encoding explicitly
    with open("test_resnet_result.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# Clear log
with open("test_resnet_result.txt", "w", encoding="utf-8") as f:
    f.write(f"Test Started: {datetime.now()}\n")

log("=" * 80)
log("VERIFICATION TEST: MNIST with ResNet50")
log("Target: Fix 'Unexpected key(s) in state_dict' error")
log("=" * 80)

try:
    # Model=2 (ResNet50), Clients=2, Rounds=2
    input_data = "2\n2\n2\n"
    
    log("\nStarting test...")
    
    start_time = datetime.now()
    
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
    
    # Send inputs and wait (timeout 10 mins)
    stdout, stderr = process.communicate(input=input_data, timeout=600)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    log(f"  End time: {end_time.strftime('%H:%M:%S')}")
    log(f"  Duration: {duration:.1f} seconds")
    log(f"  Return code: {process.returncode}")
    
    if process.returncode == 0:
        log("\n" + "=" * 80)
        log("[PASS] TEST PASSED!")
        log("=" * 80)
        # Check for result file
        result_file = 'MNIST/mnist_resnet50_results_clients2_rounds2.csv'
        if os.path.exists(result_file):
            log(f"Result file created: {result_file}")
            
    else:
        log("\n" + "=" * 80)
        log("[FAIL] TEST FAILED!")
        log("=" * 80)
        log("\nError output:")
        log("-" * 80)
        log(stderr)
        log("-" * 80)
        
except Exception as e:
    log(f"\n[ERROR] Error during test: {e}")
