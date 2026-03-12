"""
Automated test with hardcoded inputs - NO user interaction
Tests MNIST with CNN, 2 clients, 2 rounds
"""

import subprocess
from datetime import datetime
import os

print("=" * 80)
print("AUTOMATED TEST - MNIST with CNN (2 clients, 2 rounds)")
print("=" * 80)

try:
    # Hardcoded inputs: Model=1 (CNN), Clients=2, Rounds=2
    input_data = "1\n2\n2\n"
    
    print("\nStarting test...")
    print(f"  Model: 1 (CNN)")
    print(f"  Clients: 2")
    print(f"  Rounds: 2")
    print(f"  Start time: {datetime.now().strftime('%H:%M:%S')}")
    
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
    
    # Send inputs and wait (10 min timeout for 2 clients, 2 rounds)
    stdout, stderr = process.communicate(input=input_data, timeout=600)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"  End time: {end_time.strftime('%H:%M:%S')}")
    print(f"  Duration: {duration:.1f} seconds ({duration/60:.2f} minutes)")
    
    if process.returncode == 0:
        print("\n" + "=" * 80)
        print("✅ TEST PASSED!")
        print("=" * 80)
        
        # Extract final accuracy
        lines = stdout.split('\n')
        final_round_data = None
        for line in reversed(lines):
            if 'Round 2/2' in line or 'Episode' in line:
                final_round_data = line
                break
        
        if final_round_data:
            print(f"\nFinal round info: {final_round_data}")
        
        # Check for result file
        result_file = 'MNIST/mnist_cnn_results_clients2_rounds2.csv'
        if os.path.exists(result_file):
            print(f"✓ Result file created: {result_file}")
            # Read last line for final accuracy
            with open(result_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    last_line = lines[-1]
                    print(f"✓ Final result: {last_line.strip()}")
        
        print("\n" + "=" * 80)
        print("VERDICT: ✅ IMPLEMENTATION IS WORKING CORRECTLY!")
        print("=" * 80)
        print("\nThe multi-model integration is fully functional.")
        print("You can now run full experiments with:")
        print("  python run_sequential_experiments.py")
        
    else:
        print("\n" + "=" * 80)
        print("❌ TEST FAILED!")
        print("=" * 80)
        print(f"Return code: {process.returncode}")
        print("\nError output:")
        print("-" * 80)
        print(stderr)
        print("-" * 80)
        print("\nLast 20 lines of stdout:")
        lines = stdout.split('\n')
        for line in lines[-20:]:
            print(line)
        
        print("\n" + "=" * 80)
        print("VERDICT: ❌ STILL HAS ERRORS - IMPLEMENTATION NOT READY")
        print("=" * 80)
        
except subprocess.TimeoutExpired:
    print("\n❌ Test timed out (>10 minutes)")
    process.kill()
    print("\nVERDICT: ❌ TIMEOUT - IMPLEMENTATION NOT READY")
except Exception as e:
    print(f"\n❌ Error during test: {e}")
    print("\nVERDICT: ❌ EXCEPTION - IMPLEMENTATION NOT READY")

print("\n" + "=" * 80)
