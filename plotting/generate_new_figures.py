"""
Generate new client impact analysis figures for QSplitFL paper.
Creates separate Accuracy and Split Point Selection figures with proper headings and legends.
Organizes figures by dataset and model architecture for efficient subfigure merging.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# Use non-interactive backend
matplotlib.use('Agg')

# Configuration
BASE_DIR = Path(r"c:\Users\nshadin\OneDrive - Kennesaw State University\QSplitFL\complete_rl_implementation")
OUTPUT_DIR = BASE_DIR / "Paper_Figures_New"
OUTPUT_DIR.mkdir(exist_ok=True)

# Dataset configurations - updated with correct paths
DATASETS = {
    'MNIST': {
        'results_dir': 'MNIST_Results',
        'prefix': 'mnist',
        'title': 'MNIST'
    },
    'FMNIST': {
        'results_dir': 'FMNIST_Results', 
        'prefix': 'fmnist',
        'title': 'Fashion-MNIST'
    },
    'CIFAR10': {
        'results_dir': 'CIFAR10_Results',
        'prefix': 'cifar10',
        'title': 'CIFAR-10'
    },
    'CIFAR100': {
        'results_dir': 'CIFAR-100_Results',
        'prefix': 'cifar-100',
        'title': 'CIFAR-100'
    }
}

MODELS = ['CNN', 'ResNet50', 'MobileNetV4', 'ConvNeXt']
ROUNDS = [10, 20, 50, 100]
CLIENTS = [5, 10, 100, 200]

# Color scheme for clients
CLIENT_COLORS = {
    5: '#1f77b4',    # Blue
    10: '#ff7f0e',   # Orange
    100: '#2ca02c',  # Green
    200: '#d62728'   # Red
}

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'figure.dpi': 150
})


def load_data(dataset_key, model, num_clients, num_rounds):
    """Load training data from CSV file."""
    dataset = DATASETS[dataset_key]
    model_lower = model.lower()
    
    # Build path
    csv_path = BASE_DIR / dataset['results_dir'] / f"{dataset['prefix']}_{model_lower}_results_clients{num_clients}_rounds{num_rounds}.csv"
    
    if csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        # Try alternative naming
        if dataset_key == 'CIFAR100':
            csv_path = BASE_DIR / dataset['results_dir'] / f"cifar-100_{model_lower}_results_clients{num_clients}_rounds{num_rounds}.csv"
            if csv_path.exists():
                return pd.read_csv(csv_path)
        return None


def generate_combined_accuracy_by_rounds(dataset_key, model):
    """Generate combined accuracy figure showing all rounds for a model (2x2 grid)."""
    dataset = DATASETS[dataset_key]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    has_any_data = False
    
    for idx, num_rounds in enumerate(ROUNDS):
        ax = axes[idx]
        has_data = False
        
        for num_clients in CLIENTS:
            data = load_data(dataset_key, model, num_clients, num_rounds)
            if data is not None:
                ax.plot(data['Round'], data['Accuracy'] * 100, 
                       color=CLIENT_COLORS[num_clients],
                       linewidth=1.8,
                       label=f'{num_clients} Clients')
                has_data = True
                has_any_data = True
        
        ax.set_xlabel('Training Round', fontsize=9)
        ax.set_ylabel('Accuracy (%)', fontsize=9)
        ax.set_title(f'{num_rounds} Rounds', fontsize=10, fontweight='bold')
        ax.set_xlim(0, num_rounds)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        if idx == 1:  # Legend in top-right subplot
            ax.legend(loc='lower right', fontsize=8, frameon=True)
    
    if not has_any_data:
        plt.close(fig)
        return None
    
    fig.suptitle(f'{dataset["title"]} - {model}\nAccuracy Convergence vs. Client Count',
                fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    filename = f"{dataset_key}_{model}_accuracy.png"
    filepath = OUTPUT_DIR / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {filename}")
    return filepath


def generate_combined_splitpoint_by_rounds(dataset_key, model):
    """Generate combined split point figure showing all rounds for a model (2x2 grid)."""
    dataset = DATASETS[dataset_key]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    has_any_data = False
    
    for idx, num_rounds in enumerate(ROUNDS):
        ax = axes[idx]
        
        for num_clients in CLIENTS:
            data = load_data(dataset_key, model, num_clients, num_rounds)
            if data is not None:
                ax.scatter(data['Round'], data['SplitLayer'], 
                          c=CLIENT_COLORS[num_clients],
                          s=12, alpha=0.6,
                          label=f'{num_clients} Clients')
                has_any_data = True
        
        ax.set_xlabel('Training Round', fontsize=9)
        ax.set_ylabel('Selected Layer', fontsize=9)
        ax.set_title(f'{num_rounds} Rounds', fontsize=10, fontweight='bold')
        ax.set_xlim(0, num_rounds)
        ax.grid(True, alpha=0.3)
        
        if idx == 1:  # Legend in top-right subplot
            ax.legend(loc='upper right', fontsize=8, frameon=True, markerscale=1.5)
    
    if not has_any_data:
        plt.close(fig)
        return None
    
    fig.suptitle(f'{dataset["title"]} - {model}\nDynamic Split Point Selection',
                fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    filename = f"{dataset_key}_{model}_splitpoint.png"
    filepath = OUTPUT_DIR / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {filename}")
    return filepath


def generate_model_comparison_accuracy(dataset_key, num_rounds=100):
    """Generate accuracy comparison across all models for a dataset (single figure, 2x2 models)."""
    dataset = DATASETS[dataset_key]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    has_any_data = False
    
    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        has_data = False
        
        for num_clients in CLIENTS:
            data = load_data(dataset_key, model, num_clients, num_rounds)
            if data is not None:
                ax.plot(data['Round'], data['Accuracy'] * 100, 
                       color=CLIENT_COLORS[num_clients],
                       linewidth=1.8,
                       label=f'{num_clients} Clients')
                has_data = True
                has_any_data = True
        
        ax.set_xlabel('Training Round', fontsize=9)
        ax.set_ylabel('Accuracy (%)', fontsize=9)
        ax.set_title(f'{model}', fontsize=10, fontweight='bold')
        ax.set_xlim(0, num_rounds)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        if idx == 1:
            ax.legend(loc='lower right', fontsize=8, frameon=True)
    
    if not has_any_data:
        plt.close(fig)
        return None
    
    fig.suptitle(f'{dataset["title"]} - Model Architecture Comparison\nAccuracy at {num_rounds} Training Rounds',
                fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    filename = f"{dataset_key}_all_models_accuracy_{num_rounds}rounds.png"
    filepath = OUTPUT_DIR / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {filename}")
    return filepath


def generate_model_comparison_splitpoint(dataset_key, num_rounds=100):
    """Generate split point comparison across all models for a dataset."""
    dataset = DATASETS[dataset_key]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    has_any_data = False
    
    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        
        for num_clients in CLIENTS:
            data = load_data(dataset_key, model, num_clients, num_rounds)
            if data is not None:
                ax.scatter(data['Round'], data['SplitLayer'], 
                          c=CLIENT_COLORS[num_clients],
                          s=12, alpha=0.6,
                          label=f'{num_clients} Clients')
                has_any_data = True
        
        ax.set_xlabel('Training Round', fontsize=9)
        ax.set_ylabel('Selected Layer', fontsize=9)
        ax.set_title(f'{model}', fontsize=10, fontweight='bold')
        ax.set_xlim(0, num_rounds)
        ax.grid(True, alpha=0.3)
        
        if idx == 1:
            ax.legend(loc='upper right', fontsize=8, frameon=True, markerscale=1.5)
    
    if not has_any_data:
        plt.close(fig)
        return None
    
    fig.suptitle(f'{dataset["title"]} - Model Architecture Comparison\nSplit Point Selection at {num_rounds} Training Rounds',
                fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    filename = f"{dataset_key}_all_models_splitpoint_{num_rounds}rounds.png"
    filepath = OUTPUT_DIR / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {filename}")
    return filepath


def main():
    print("="*60)
    print("Generating Client Impact Analysis Figures")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    total_figures = 0
    
    # Generate per-model figures (accuracy and split point with all rounds)
    for dataset_key in DATASETS:
        print(f"\n{'='*50}")
        print(f"Dataset: {DATASETS[dataset_key]['title']}")
        print(f"{'='*50}")
        
        for model in MODELS:
            print(f"\n  Model: {model}")
            
            # Generate combined figures showing all rounds in 2x2 grid
            if generate_combined_accuracy_by_rounds(dataset_key, model):
                total_figures += 1
            if generate_combined_splitpoint_by_rounds(dataset_key, model):
                total_figures += 1
    
    # Generate model comparison figures for each dataset (all models at 100 rounds)
    print(f"\n{'='*50}")
    print("Generating Model Comparison Figures")
    print(f"{'='*50}")
    
    for dataset_key in DATASETS:
        print(f"\n  Dataset: {DATASETS[dataset_key]['title']}")
        if generate_model_comparison_accuracy(dataset_key, 100):
            total_figures += 1
        if generate_model_comparison_splitpoint(dataset_key, 100):
            total_figures += 1
    
    print(f"\n{'='*60}")
    print(f"Complete! Generated {total_figures} figures.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
