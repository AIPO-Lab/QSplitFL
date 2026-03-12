"""
Generate detailed "Comparison by Clients" plots for ALL datasets, ALL models, and ALL round settings.
Exhaustively generates plots comparing 5, 10, 100, 200 clients for varying round counts.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATASETS = ['MNIST', 'FMNIST', 'CIFAR10', 'CIFAR-100']
MODELS = ['CNN', 'ResNet50', 'MobileNetV4', 'ConvNeXt']
CLIENTS = [5, 10, 100, 200]
ROUNDS_SETTINGS = [10, 20, 50, 100]

FOLDER_MAP = {
    'MNIST': 'MNIST', 'FMNIST': 'FMNIST',
    'CIFAR10': 'CIFAR10', 'CIFAR-100': 'CIFAR-100'
}

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Blue, Orange, Green, Red

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

# ==============================================================================
# PLOTTING LOGIC
# ==============================================================================

def generate_client_comparison_plot(ds_name, model, n_rounds, output_dir):
    """
    Generates a 2x2 plot comparing performance across client counts (5, 10, 100, 200)
    for a specific Dataset, Model, and Round Count.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{ds_name} ({model}): Client Scale Comparison ({n_rounds} Rounds)', 
                 fontsize=16, fontweight='bold')
    
    data_found = False
    
    for idx, n_clients in enumerate(CLIENTS):
        # Construct filename consistent with perform_final_generation.py
        filename = f"{FOLDER_MAP[ds_name].lower()}_{model.lower()}_results_clients{n_clients}_rounds{n_rounds}.csv"
        filepath = os.path.join(output_dir, filename)
        
        if os.path.exists(filepath):
            data_found = True
            df = pd.read_csv(filepath)
            
            # 1. Accuracy Curve
            axes[0, 0].plot(df['Round'], df['Accuracy'], label=f'{n_clients} Clients', 
                           color=COLORS[idx], linewidth=2, alpha=0.8)
            
            # 2. Loss Curve
            axes[0, 1].plot(df['Round'], df['Loss'], label=f'{n_clients} Clients', 
                           color=COLORS[idx], linewidth=2, alpha=0.8)
            
            # 3. Reward Curve
            axes[1, 0].plot(df['Round'], df['Reward'], label=f'{n_clients} Clients', 
                           color=COLORS[idx], linewidth=1.5, alpha=0.6)
            
            # 4. Split Layer (Scatter + Avg Line)
            axes[1, 1].scatter(df['Round'], df['SplitLayer'], color=COLORS[idx], 
                              s=20, alpha=0.5, label=f'{n_clients} Clients')
            # Add average line for visibility
            avg_split = df['SplitLayer'].mean()
            axes[1, 1].axhline(y=avg_split, color=COLORS[idx], linestyle='--', alpha=0.5)

    if not data_found:
        plt.close()
        return

    # Formatting 1. Accuracy
    axes[0, 0].set_title(f'Accuracy Convergence ({ds_name} - {model})')
    axes[0, 0].set_xlabel('Round'); axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].grid(True, alpha=0.3); axes[0, 0].legend()

    # Formatting 2. Loss
    axes[0, 1].set_title(f'Loss Minimization ({ds_name} - {model})')
    axes[0, 1].set_xlabel('Round'); axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3); axes[0, 1].legend()

    # Formatting 3. Reward
    axes[1, 0].set_title(f'RL Reward Signal ({ds_name} - {model})')
    axes[1, 0].set_xlabel('Round'); axes[1, 0].set_ylabel('Reward')
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(loc='lower right') # Added Legend

    # Formatting 4. Split Layer
    axes[1, 1].set_title(f'Split Point Selection ({ds_name} - {model})')
    axes[1, 1].set_xlabel('Round'); axes[1, 1].set_ylabel('Layer Index')
    
    # DYNAMIC LIMIT BASED ON MODEL DEPTH
    max_depth = get_model_depth(model)
    axes[1, 1].set_ylim(0, max_depth + 5)
    
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(loc='upper right') # Added Legend
    
    plt.tight_layout()
    
    # Save Plot
    out_filename = f"{ds_name}_{model}_comparison_by_clients_{n_rounds}rounds.png"
    plt.savefig(os.path.join(output_dir, out_filename), dpi=150)
    plt.close()
    # print(f"Generated: {out_filename}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("="*80)
    print("GENERATING DETAILED CLIENT COMPARISON PLOTS (ALL COMBINATIONS)")
    print("="*80)
    
    count = 0
    for ds_name in DATASETS:
        folder = f"{FOLDER_MAP[ds_name]}_Results"
        if not os.path.exists(folder):
            print(f"[WARN] Folder for {ds_name} not found, skipping...")
            continue
            
        print(f"Processing {ds_name}...")
        for model in MODELS:
            for n_rounds in ROUNDS_SETTINGS:
                generate_client_comparison_plot(ds_name, model, n_rounds, folder)
                count += 1
                
    print("\n" + "="*80)
    print(f"✅ GENERATION COMPLETE. Created approx {count} plots.")
    print("Check [Dataset]_Results folders.")
    print("="*80)
