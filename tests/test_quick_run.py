"""
Test script with hardcoded values to verify the fix
Model: CNN (1), Clients: 2, Rounds: 2 for quick testing
"""

import subprocess
from datetime import datetime

print("=" * 80)
print("TESTING FIXED IMPLEMENTATION - HARDCODED VALUES")
print("=" * 80)
print("\nTest Configuration:")
print("  Model: 1 (CNN - fastest for testing)")
print("  Clients: 2")
print("  Rounds: 2")
print("  Dataset: MNIST (fastest dataset)")
print("\nThis is a quick test to verify the syntax fixes work.")
print("=" * 80)

confirm = input("\nProceed with test? (yes/no): ").strip().lower()
if confirm not in ['yes', 'y']:
    print("Test cancelled.")
    exit(0)

print("\n" + "=" * 80)
print("RUNNING TEST EXPERIMENT")
print("=" * 80)

try:
    # Hardcoded inputs: Model=1, Clients=2, Rounds=2
    input_data = "1\n2\n2\n"
    
    print("\n[MNIST] Running with CNN, 2 clients, 2 rounds...")
    print("Started:", datetime.now().strftime('%H:%M:%S'))
    
    start_time = datetime.now()
    
    process = subprocess.Popen(
        ['python', 'main_complete_rl.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd='MNIST',
        text=True,
        bufsize=1
    )
    
    # Send inputs and wait
    stdout, stderr = process.communicate(input=input_data, timeout=600)  # 10 min timeout
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("Completed:", end_time.strftime('%H:%M:%S'))
    print(f"Duration: {duration:.1f} seconds")
    
    if process.returncode == 0:
        print("\n" + "=" * 80)
        print("✓ TEST SUCCESSFUL!")
        print("=" * 80)
        print("\nThe fix is working correctly. You can now run full experiments.")
        print("\nLast 20 lines of output:")
        print("-" * 80)
        lines = stdout.split('\n')
        for line in lines[-20:]:
            print(line)
        print("-" * 80)
        
        # Check for result file
        import os
        result_file = 'MNIST/mnist_cnn_results_clients2_rounds2.csv'
        if os.path.exists(result_file):
            print(f"\n✓ Result file created: {result_file}")
        
    else:
        print("\n" + "=" * 80)
        print("✗ TEST FAILED!")
        print("=" * 80)
        print(f"Return code: {process.returncode}")
        print("\nStderr:")
        print(stderr)
        print("\nStdout (last 30 lines):")
        lines = stdout.split('\n')
        for line in lines[-30:]:
            print(line)
        
except subprocess.TimeoutExpired:
    print("\n✗ Test timed out (>10 minutes)")
    process.kill()
except Exception as e:
    print(f"\n✗ Error during test: {e}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
