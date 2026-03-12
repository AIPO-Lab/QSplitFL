import os
import subprocess

# Model name mappings - All use CNN
MODEL_NAMES = {
    'mnist': 'CNN',
    'fmnist': 'CNN',
    'cifar10': 'CNN',
    'cifar100': 'CNN'
}

DATASET_DISPLAY = {
    'mnist': 'MNIST',
    'fmnist': 'Fashion-MNIST',
    'cifar10': 'CIFAR-10',
    'cifar100': 'CIFAR-100'
}

def update_regenerate_comparison_plots(dataset, result_dir):
    """Update regenerate_comparison_plots.py to include model name"""
    
    model_name = MODEL_NAMES[dataset]
    dataset_display = DATASET_DISPLAY[dataset]
    
    script_content = f'''"""
Regenerate comparison plots for {dataset_display} with model name
Model: {model_name}
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_results(data_dir, num_clients, num_rounds):
    """Load results CSV file"""
    filename = f"{dataset}_results_clients{{num_clients}}_rounds{{num_rounds}}.csv"
    filepath = os.path.join(data_dir, filename)
    return pd.read_csv(filepath)

def regenerate_comparison_by_clients(data_dir, output_dir):
    """Create comparison plots showing effect of different client counts"""
    
    client_counts = [5, 10, 100, 200]
    round_setting = 100  # Use 100 rounds for comparison
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{dataset_display} ({model_name}): Impact of Client Count (100 Rounds)', 
                 fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, num_clients in enumerate(client_counts):
        df = load_results(data_dir, num_clients, round_setting)
        
        # Accuracy
        axes[0, 0].plot(df['Round'], df['Accuracy'], label=f'{{num_clients}} clients',
                       color=colors[idx], linewidth=2, alpha=0.7)
        
        # Loss
        axes[0, 1].plot(df['Round'], df['Loss'], label=f'{{num_clients}} clients',
                       color=colors[idx], linewidth=2, alpha=0.7)
        
        # Reward
        axes[1, 0].plot(df['Round'], df['Reward'], label=f'{{num_clients}} clients',
                       color=colors[idx], linewidth=2, alpha=0.7)
        
        # Split layers - DOTS with average line
        axes[1, 1].scatter(df['Round'], df['SplitLayer'], label=f'{{num_clients}} clients',
                          color=colors[idx], s=30, alpha=0.6)
        
        # Calculate and display average split point
        avg_split = df['SplitLayer'].mean()
        axes[1, 1].axhline(y=avg_split, color=colors[idx], linestyle='--', 
                          linewidth=1.5, alpha=0.8)
    
    # Format subplots
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title(f'{dataset_display} - Model Performance Across Client Scales')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title(f'{dataset_display} - Training Loss Minimization Trends')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].set_title(f'{dataset_display} - Reward Over Time')
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Split Layer')
    axes[1, 1].set_title(f'{dataset_display} - Split Point Selection ({model_name})')
    axes[1, 1].set_yticks([5, 6, 7, 8, 9])
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    filename = "{dataset}_comparison_by_clients.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Regenerated: {{filename}}")

def regenerate_comparison_by_rounds(data_dir, output_dir):
    """Create comparison plots showing effect of different round counts"""
    
    round_counts = [10, 20, 50, 100]
    client_setting = 10  # Use 10 clients for comparison
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{dataset_display} ({model_name}): Impact of Training Rounds (10 Clients)', 
                 fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, num_rounds in enumerate(round_counts):
        df = load_results(data_dir, client_setting, num_rounds)
        
        # Accuracy
        axes[0, 0].plot(df['Round'], df['Accuracy'], label=f'{{num_rounds}} rounds',
                       color=colors[idx], linewidth=2, alpha=0.7, marker='o', 
                       markersize=4 if num_rounds <= 20 else 0)
        
        # Loss
        axes[0, 1].plot(df['Round'], df['Loss'], label=f'{{num_rounds}} rounds',
                       color=colors[idx], linewidth=2, alpha=0.7, marker='o',
                       markersize=4 if num_rounds <= 20 else 0)
        
        # Reward
        axes[1, 0].plot(df['Round'], df['Reward'], label=f'{{num_rounds}} rounds',
                       color=colors[idx], linewidth=2, alpha=0.7, marker='o',
                       markersize=4 if num_rounds <= 20 else 0)
        
        # Final accuracy bar chart
        axes[1, 1].bar(idx, df['Accuracy'].iloc[-1], color=colors[idx], 
                      alpha=0.7, label=f'{{num_rounds}} rounds')
    
    # Format subplots
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title(f'{dataset_display} - Impact of Training Duration on Accuracy')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title(f'{dataset_display} - Loss Optimization Over Training Rounds')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].set_title(f'{dataset_display} - Reward Over Time')
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    axes[1, 1].set_xlabel('Training Rounds')
    axes[1, 1].set_ylabel('Final Accuracy (%)')
    axes[1, 1].set_title(f'{dataset_display} - Final Accuracy by Round Count ({model_name})')
    axes[1, 1].set_xticks(range(len(round_counts)))
    axes[1, 1].set_xticklabels([f'{{r}}' for r in round_counts])
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    filename = "{dataset}_comparison_by_rounds.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Regenerated: {{filename}}")

if __name__ == "__main__":
    print("=" * 80)
    print("REGENERATING {dataset_display.upper()} COMPARISON PLOTS WITH MODEL NAME")
    print(f"Model: {model_name}")
    print("=" * 80)
    
    result_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\\n[INFO] Regenerating {dataset_display} comparison plots...")
    print(f"Directory: {{result_dir}}\\n")
    regenerate_comparison_by_clients(result_dir, result_dir)
    regenerate_comparison_by_rounds(result_dir, result_dir)
    
    print("\\n" + "=" * 80)
    print(f"✅ {dataset_display.upper()} COMPARISON PLOTS REGENERATED")
    print("=" * 80)
'''
    
    # Write the script with UTF-8 encoding
    script_path = os.path.join(result_dir, 'regenerate_comparison_plots.py')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"[OK] Updated: {script_path}")

def main():
    print("="*80)
    print("UPDATING ALL PLOTTING SCRIPTS WITH MODEL NAMES")
    print("="*80)
    
    base_dir = r"c:\Users\nshadin\OneDrive - Kennesaw State University\QSplitFL\complete_rl_implementation"
    
    datasets = {
        'mnist': os.path.join(base_dir, 'MNIST_Results'),
        'fmnist': os.path.join(base_dir, 'FMNIST_Results'),
        'cifar10': os.path.join(base_dir, 'CIFAR10_Results'),
        'cifar100': os.path.join(base_dir, 'CIFAR-100_Results')
    }
    
    # Update regenerate_comparison_plots.py for each dataset
    for dataset, result_dir in datasets.items():
        print(f"\n[{DATASET_DISPLAY[dataset]}] Updating scripts...")
        update_regenerate_comparison_plots(dataset, result_dir)
    
    print("\n" + "="*80)
    print("✅ ALL SCRIPTS UPDATED")
    print("="*80)
    
    # Now regenerate all plots
    print("\n" + "="*80)
    print("REGENERATING ALL COMPARISON PLOTS")
    print("="*80)
    
    for dataset, result_dir in datasets.items():
        print(f"\n[{DATASET_DISPLAY[dataset]}] Regenerating plots...")
        script_path = os.path.join(result_dir, 'regenerate_comparison_plots.py')
        
        try:
            result = subprocess.run(['python', script_path], 
                                  capture_output=True, text=True, cwd=result_dir)
            print(result.stdout)
            if result.returncode != 0:
                print(f"Error: {result.stderr}")
        except Exception as e:
            print(f"Error running script: {e}")
    
    print("\n" + "="*80)
    print("✅ ALL PLOTS REGENERATED WITH MODEL NAMES")
    print("="*80)

if __name__ == "__main__":
    main()
