"""
Enhanced Plot Generator with:
- Comparison plots for ALL round combinations
- Dataset-specific titles
- Support for ALL Models (CNN, ResNet50, MobileNetV4, ConvNeXt)
- Dynamic Split Layer Depths
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ==============================================================================
# CONFIG
# ==============================================================================

MODELS = ['CNN', 'ResNet50', 'MobileNetV4', 'ConvNeXt']
DATASETS = ['MNIST', 'CIFAR10', 'FMNIST', 'CIFAR-100']

FOLDER_MAP = {
    'MNIST': 'MNIST', 'FMNIST': 'FMNIST',
    'CIFAR10': 'CIFAR10', 'CIFAR-100': 'CIFAR-100'
}

def get_model_depth(model_name):
    """Return max split layer based on model architecture."""
    if 'ResNet50' in model_name:
        return 50
    elif 'MobileNetV4' in model_name:
        return 53
    elif 'ConvNeXt' in model_name:
        return 59
    else:
        return 10 # CNN

def load_results(data_dir, num_clients, num_rounds, dataset, model):
    """Load results CSV file with model name"""
    # Pattern: [dataset]_[model]_results_clients[N]_rounds[N].csv
    # e.g. mnist_cnn_results_clients5_rounds10.csv
    filename = f"{FOLDER_MAP[dataset].lower()}_{model.lower()}_results_clients{num_clients}_rounds{num_rounds}.csv"
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return None

def plot_individual_config(df, num_clients, num_rounds, dataset, model, output_dir):
    """Create comprehensive plot for one configuration"""
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    dataset_name = dataset.upper()
    fig.suptitle(f'{dataset_name} ({model}): {num_clients} Clients, {num_rounds} Rounds', 
                 fontsize=16, fontweight='bold')
    
    rounds = df['Round'].values
    
    # Plot 1: Accuracy
    axes[0, 0].plot(rounds, df['Accuracy'], 'b-', linewidth=2, alpha=0.7, label='Accuracy')
    axes[0, 0].set_xlabel('Round', fontsize=11)
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=11)
    axes[0, 0].set_title(f'Test Accuracy Over Time', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Loss
    axes[0, 1].plot(rounds, df['Loss'], 'r-', linewidth=2, alpha=0.7, label='Loss')
    axes[0, 1].set_xlabel('Round', fontsize=11)
    axes[0, 1].set_ylabel('Loss', fontsize=11)
    axes[0, 1].set_title(f'Test Loss Over Time', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Rewards
    axes[1, 0].plot(rounds, df['Reward'], 'g-', linewidth=2, alpha=0.7, label='Reward')
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].set_xlabel('Round', fontsize=11)
    axes[1, 0].set_ylabel('Reward', fontsize=11)
    axes[1, 0].set_title('RL Reward Signal', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 4: Split Points
    axes[1, 1].plot(rounds, df['SplitLayer'], 'c-', alpha=0.5, marker='o', markersize=4, linewidth=1.5, label='Split Layer')
    axes[1, 1].set_xlabel('Round', fontsize=11)
    axes[1, 1].set_ylabel('Split Layer', fontsize=11)
    axes[1, 1].set_title('Selected Split Points', fontsize=12, fontweight='bold')
    
    # Dynamic Limits
    max_depth = get_model_depth(model)
    axes[1, 1].set_ylim(0, max_depth + 2)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(loc='upper right') # Added Legend
    
    # Plot 5: Epsilon Decay
    axes[2, 0].plot(rounds, df['Epsilon'], 'm-', linewidth=2, alpha=0.7, label='Epsilon')
    axes[2, 0].set_xlabel('Round', fontsize=11)
    axes[2, 0].set_ylabel('Epsilon', fontsize=11)
    axes[2, 0].set_title('Exploration Rate Decay', fontsize=12, fontweight='bold')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()
    
    # Plot 6: Summary Statistics
    axes[2, 1].axis('off')
    summary_text = f"""
    Summary Statistics
    {'=' * 40}
    
    Model: {model}
    
    Final Accuracy: {df['Accuracy'].iloc[-1]:.2f}%
    Avg Accuracy: {df['Accuracy'].mean():.2f}%
    
    Final Loss: {df['Loss'].iloc[-1]:.4f}
    Avg Loss: {df['Loss'].mean():.4f}
    
    Avg Reward: {df['Reward'].mean():.4f}
    
    Most Common Split: {df['SplitLayer'].mode()[0]}
    Max Depth Available: {max_depth}
    """
    axes[2, 1].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                    verticalalignment='center')
    
    plt.tight_layout()
    
    filename = f"{FOLDER_MAP[dataset].lower()}_{model.lower()}_plot_clients{num_clients}_rounds{num_rounds}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    # print(f"✓ {filename}")

def generate_all_plots_for_dataset(dataset):
    """Generate all plots for a dataset"""
    
    data_dir = f"c:/Users/nshadin/OneDrive - Kennesaw State University/QSplitFL/complete_rl_implementation/{FOLDER_MAP[dataset]}_Results"
    
    if not os.path.exists(data_dir):
        print(f"Skipping {dataset} (No results folder)")
        return

    client_counts = [5, 10, 100, 200]
    round_counts = [10, 20, 50, 100]
    
    print(f"\n📊 Generating {dataset.upper()} Plots...")
    
    count = 0
    for model in MODELS:
        print(f"  Processing {model}...")
        for num_clients in client_counts:
            for num_rounds in round_counts:
                df = load_results(data_dir, num_clients, num_rounds, dataset, model)
                if df is not None:
                    plot_individual_config(df, num_clients, num_rounds, dataset, model, data_dir)
                    count += 1
    
    print(f"✅ Generated {count} individual plots for {dataset}")

if __name__ == "__main__":
    print("=" * 80)
    print("GENERATING INDIVIDUAL CONFIG PLOTS (ALL MODELS)")
    print("=" * 80)
    
    for ds in DATASETS:
        generate_all_plots_for_dataset(ds)
    
    print("\n" + "=" * 80)
    print("✅ ALL PLOTS GENERATED")
    print("=" * 80)
