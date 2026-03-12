"""
Sequential Multi-Dataset Experiment Runner with Custom Configurations
Run selected model across all datasets with user-defined clients and rounds
"""

import subprocess
import sys
import os
from datetime import datetime

def estimate_time(dataset, model_choice, clients, rounds):
    """
    Estimate experiment time based on dataset, model, clients, and rounds.
    Returns time in minutes.
    """
    # Base time per round per client (in minutes)
    base_times = {
        'MNIST': {'CNN': 0.3, 'ResNet50': 0.6, 'MobileNetV4': 0.5, 'ConvNeXt': 0.7},
        'FMNIST': {'CNN': 0.4, 'ResNet50': 0.7, 'MobileNetV4': 0.6, 'ConvNeXt': 0.8},
        'CIFAR-10': {'CNN': 1.5, 'ResNet50': 3.5, 'MobileNetV4': 2.5, 'ConvNeXt': 4.5},
        'CIFAR-100': {'CNN': 2.0, 'ResNet50': 5.0, 'MobileNetV4': 3.5, 'ConvNeXt': 6.0}
    }
    
    model_names = {1: 'CNN', 2: 'ResNet50', 3: 'MobileNetV4', 4: 'ConvNeXt'}
    model_name = model_names[model_choice]
    
    base_time = base_times[dataset][model_name]
    estimated_minutes = base_time * clients * rounds
    
    return estimated_minutes

def format_time(minutes):
    """Convert minutes to human-readable format"""
    if minutes < 60:
        return f"~{int(minutes)} min"
    else:
        hours = minutes / 60
        return f"~{hours:.1f} hours"

print("=" * 80)
print("SEQUENTIAL MULTI-MODEL EXPERIMENT RUNNER")
print("Custom Configuration for Each Dataset")
print("=" * 80)

# Model selection
print("\nSelect Model Architecture to test across all datasets:")
print("=" * 60)
print("1: CNN (Baseline)")
print("2: ResNet50 (Deep Residual Network)")
print("3: MobileNetV4 (Efficient Mobile Architecture)")
print("4: ConvNeXt (Modern CNN with Vision Transformer Design)")
print("=" * 60)

while True:
    try:
        model_choice = int(input("\nEnter model number (1-4): "))
        if 1 <= model_choice <= 4:
            break
        else:
            print("Please enter a number between 1 and 4.")
    except ValueError:
        print("Invalid input. Please enter a valid integer.")

model_names = {1: "CNN", 2: "ResNet50", 3: "MobileNetV4", 4: "ConvNeXt"}
selected_model = model_names[model_choice]

print(f"\n✓ Selected model: {selected_model}")

# Configure each dataset
print("\n" + "=" * 80)
print("CONFIGURE EXPERIMENTS FOR EACH DATASET")
print("=" * 80)

dataset_configs = []
datasets_info = [
    {'name': 'MNIST', 'folder': 'MNIST', 'desc': 'Grayscale digits (28x28x1, 10 classes)'},
    {'name': 'FMNIST', 'folder': 'FMNIST', 'desc': 'Fashion items (28x28x1, 10 classes)'},
    {'name': 'CIFAR-10', 'folder': 'CIFAR10', 'desc': 'Color images (32x32x3, 10 classes)'},
    {'name': 'CIFAR-100', 'folder': 'CIFAR-100', 'desc': 'Color images (32x32x3, 100 classes)'}
]

total_estimated_time = 0

for idx, dataset_info in enumerate(datasets_info, 1):
    print(f"\n[{idx}/4] {dataset_info['name']} - {dataset_info['desc']}")
    print("-" * 60)
    
    # Get number of clients
    while True:
        try:
            clients = int(input(f"  Enter number of clients for {dataset_info['name']}: "))
            if clients > 0:
                break
            else:
                print("  Please enter a positive number.")
        except ValueError:
            print("  Invalid input. Please enter a valid integer.")
    
    # Get number of rounds
    while True:
        try:
            rounds = int(input(f"  Enter number of rounds for {dataset_info['name']}: "))
            if rounds > 0:
                break
            else:
                print("  Please enter a positive number.")
        except ValueError:
            print("  Invalid input. Please enter a valid integer.")
    
    # Estimate time
    estimated_time = estimate_time(dataset_info['name'], model_choice, clients, rounds)
    total_estimated_time += estimated_time
    
    dataset_configs.append({
        'dataset': dataset_info['name'],
        'folder': dataset_info['folder'],
        'clients': clients,
        'rounds': rounds,
        'description': dataset_info['desc'],
        'estimated_time': estimated_time
    })
    
    print(f"  ✓ Configured: {clients} clients, {rounds} rounds")
    print(f"  Estimated time: {format_time(estimated_time)}")

# Show summary and confirm
print("\n" + "=" * 80)
print("EXPERIMENT SUMMARY")
print("=" * 80)
print(f"Model: {selected_model}")
print("\nConfigurations:")
print("-" * 80)
print(f"{'Dataset':<15} {'Clients':<10} {'Rounds':<10} {'Est. Time':<15}")
print("-" * 80)
for config in dataset_configs:
    print(f"{config['dataset']:<15} {config['clients']:<10} {config['rounds']:<10} {format_time(config['estimated_time']):<15}")
print("-" * 80)
print(f"{'TOTAL':<35} {format_time(total_estimated_time):<15}")
print("-" * 80)

# confirm = input("\nProceed with experiments? (yes/no): ").strip().lower()
# if confirm not in ['yes', 'y']:
#    print("Experiments cancelled.")
#    sys.exit(0)
print("\nProceeding with experiments automatically...")

# Log file
log_file = f"experiment_log_{selected_model.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(log_file, 'w') as f:
    f.write(f"Experiment Log - {selected_model}\n")
    f.write(f"Started: {datetime.now()}\n")
    f.write("=" * 80 + "\n\n")
    f.write("Configurations:\n")
    for config in dataset_configs:
        f.write(f"  {config['dataset']}: {config['clients']} clients, {config['rounds']} rounds\n")
    f.write("\n" + "=" * 80 + "\n\n")

print("\n" + "=" * 80)
print(f"STARTING EXPERIMENTS WITH {selected_model.upper()}")
print("=" * 80)

results_summary = []

for idx, exp in enumerate(dataset_configs, 1):
    print(f"\n{'=' * 80}")
    print(f"EXPERIMENT {idx}/4: {exp['dataset']} with {selected_model}")
    print(f"{'=' * 80}")
    print(f"Description: {exp['description']}")
    print(f"Configuration: {exp['clients']} clients, {exp['rounds']} rounds")
    print(f"Estimated time: {format_time(exp['estimated_time'])}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    # Change to dataset directory
    dataset_path = exp['folder']
    
    try:
        # Create input string
        input_data = f"{model_choice}\n{exp['clients']}\n{exp['rounds']}\n"
        
        # Run experiment
        print(f"[INFO] Running {exp['dataset']} experiment...")
        print(f"[INFO] Command: python main_complete_rl.py")
        print(f"[INFO] Inputs: Model={model_choice}, Clients={exp['clients']}, Rounds={exp['rounds']}")
        print()
        
        start_time = datetime.now()
        
        process = subprocess.Popen(
            ['python', 'main_complete_rl.py'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=dataset_path,
            text=True,
            bufsize=1
        )
        
        # Send inputs
        # Removed timeout as requested by user
        stdout, stderr = process.communicate(input=input_data)
        
        end_time = datetime.now()
        actual_time = (end_time - start_time).total_seconds() / 60  # minutes
        
        # Log output
        with open(log_file, 'a') as f:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Experiment {idx}: {exp['dataset']}\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Started: {start_time}\n")
            f.write(f"Completed: {end_time}\n")
            f.write(f"Duration: {format_time(actual_time)}\n")
            f.write(f"Estimated: {format_time(exp['estimated_time'])}\n\n")
            f.write(stdout)
            if stderr:
                f.write("\nErrors:\n")
                f.write(stderr)
            f.write("\n\n")
        
        if process.returncode == 0:
            print(f"\n✓ {exp['dataset']} experiment completed successfully!")
            print(f"   Actual time: {format_time(actual_time)} (estimated: {format_time(exp['estimated_time'])})")
            
            # Parse final accuracy from output
            lines = stdout.split('\n')
            final_acc = "N/A"
            for line in reversed(lines):
                # Robust parsing for "Final Accuracy: 0.1234" or "Accuracy: 12.34%"
                if 'Final Accuracy:' in line:
                    try:
                        # Extract value after colon
                        val_str = line.split(':')[-1].strip()
                        # specific handling if % is present or not
                        if '%' in val_str:
                            final_acc = val_str
                        else:
                            # Convert 0.8543 to 85.43%
                            float_val = float(val_str)
                            final_acc = f"{float_val*100:.2f}%"
                        break
                    except:
                        pass
                elif 'Accuracy' in line and '%' in line and final_acc == "N/A":
                     try:
                        final_acc = line.split(':')[-1].strip()
                        break
                     except:
                        pass
            
            results_summary.append({
                'dataset': exp['dataset'],
                'model': selected_model,
                'clients': exp['clients'],
                'rounds': exp['rounds'],
                'status': 'SUCCESS',
                'accuracy': final_acc,
                'time': format_time(actual_time)
            })
        else:
            print(f"\n✗ {exp['dataset']} experiment failed with return code {process.returncode}")
            results_summary.append({
                'dataset': exp['dataset'],
                'model': selected_model,
                'clients': exp['clients'],
                'rounds': exp['rounds'],
                'status': 'FAILED',
                'accuracy': 'N/A',
                'time': format_time(actual_time)
            })
            
            # Ask if should continue
            if idx < 4:
                cont = input(f"\nExperiment failed. Continue with next dataset? (yes/no): ").strip().lower()
                if cont not in ['yes', 'y']:
                    print("Stopping experiments.")
                    break
        
    except subprocess.TimeoutExpired:
        print(f"\n✗ {exp['dataset']} experiment timed out (>4 hours)")
        process.kill()
        results_summary.append({
            'dataset': exp['dataset'],
            'model': selected_model,
            'clients': exp['clients'],
            'rounds': exp['rounds'],
            'status': 'TIMEOUT',
            'accuracy': 'N/A',
            'time': '>4 hours'
        })
    except Exception as e:
        print(f"\n✗ Error running {exp['dataset']}: {e}")
        results_summary.append({
            'dataset': exp['dataset'],
            'model': selected_model,
            'clients': exp['clients'],
            'rounds': exp['rounds'],
            'status': 'ERROR',
            'accuracy': str(e),
            'time': 'N/A'
        })

# Final summary
print("\n" + "=" * 80)
print("EXPERIMENT SUMMARY")
print("=" * 80)
print(f"Model: {selected_model}")
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nResults:")
print("-" * 80)
print(f"{'Dataset':<12} {'Clients':<8} {'Rounds':<8} {'Status':<10} {'Accuracy':<15} {'Time':<15}")
print("-" * 80)
for result in results_summary:
    print(f"{result['dataset']:<12} {result['clients']:<8} {result['rounds']:<8} {result['status']:<10} {result['accuracy']:<15} {result['time']:<15}")
print("-" * 80)

# Save summary
with open(log_file, 'a') as f:
    f.write("\n" + "=" * 80 + "\n")
    f.write("FINAL SUMMARY\n")
    f.write("=" * 80 + "\n")
    f.write(f"Model: {selected_model}\n")
    f.write(f"Completed: {datetime.now()}\n\n")
    f.write("Results:\n")
    f.write("-" * 80 + "\n")
    for result in results_summary:
        f.write(f"{result['dataset']}: {result['status']} - Clients: {result['clients']}, Rounds: {result['rounds']}, Accuracy: {result['accuracy']}, Time: {result['time']}\n")

print(f"\n✓ Full log saved to: {log_file}")
print("\nNext steps:")
print("1. Check individual result CSV files in each dataset's Results folder")
print("2. Run comparison plot scripts to visualize results")
print("3. Compare with other models if needed")
print("\n" + "=" * 80)
