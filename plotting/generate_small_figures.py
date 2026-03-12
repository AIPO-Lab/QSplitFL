"""
Generate smaller client impact analysis figures for QSplitFL paper.
Creates individual figures per model/round combination.
Includes ALL training rounds: 10, 20, 50, 100.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.use('Agg')

BASE_DIR = Path(r"c:\Users\nshadin\OneDrive - Kennesaw State University\QSplitFL\complete_rl_implementation")
OUTPUT_DIR = BASE_DIR / "Paper_Figures_Small"
OUTPUT_DIR.mkdir(exist_ok=True)

DATASETS = {
    'MNIST': {'results_dir': 'MNIST_Results', 'prefix': 'mnist', 'title': 'MNIST'},
    'FMNIST': {'results_dir': 'FMNIST_Results', 'prefix': 'fmnist', 'title': 'Fashion-MNIST'},
    'CIFAR10': {'results_dir': 'CIFAR10_Results', 'prefix': 'cifar10', 'title': 'CIFAR-10'},
    'CIFAR100': {'results_dir': 'CIFAR-100_Results', 'prefix': 'cifar-100', 'title': 'CIFAR-100'}
}

MODELS = ['CNN', 'ResNet50', 'MobileNetV4', 'ConvNeXt']
ROUNDS = [10, 20, 50, 100]  # All round configurations
CLIENTS = [5, 10, 100, 200]
CLIENT_COLORS = {5: '#1f77b4', 10: '#ff7f0e', 100: '#2ca02c', 200: '#d62728'}

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 7})


def load_data(dataset_key, model, num_clients, num_rounds):
    dataset = DATASETS[dataset_key]
    model_lower = model.lower()
    csv_path = BASE_DIR / dataset['results_dir'] / f"{dataset['prefix']}_{model_lower}_results_clients{num_clients}_rounds{num_rounds}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def generate_accuracy_single(dataset_key, model, num_rounds):
    """Single accuracy plot for one model at specific rounds."""
    dataset = DATASETS[dataset_key]
    
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    has_data = False
    
    for num_clients in CLIENTS:
        data = load_data(dataset_key, model, num_clients, num_rounds)
        if data is not None:
            ax.plot(data['Round'], data['Accuracy'] * 100, 
                   color=CLIENT_COLORS[num_clients], linewidth=1.5,
                   label=f'{num_clients} Clients')
            has_data = True
    
    if not has_data:
        plt.close(fig)
        return None
    
    ax.set_xlabel('Training Round')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'{dataset["title"]} - {model} ({num_rounds}R)', fontweight='bold')
    ax.legend(loc='lower right', frameon=True)
    ax.set_xlim(0, num_rounds)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"{dataset_key}_{model}_acc_{num_rounds}r.png"
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {filename}")
    return OUTPUT_DIR / filename


def generate_splitpoint_single(dataset_key, model, num_rounds):
    """Single split point plot for one model at specific rounds."""
    dataset = DATASETS[dataset_key]
    
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    has_data = False
    
    for num_clients in CLIENTS:
        data = load_data(dataset_key, model, num_clients, num_rounds)
        if data is not None:
            ax.scatter(data['Round'], data['SplitLayer'], 
                      c=CLIENT_COLORS[num_clients], s=8, alpha=0.6,
                      label=f'{num_clients} Clients')
            has_data = True
    
    if not has_data:
        plt.close(fig)
        return None
    
    ax.set_xlabel('Training Round')
    ax.set_ylabel('Split Layer')
    ax.set_title(f'{dataset["title"]} - {model} ({num_rounds}R)', fontweight='bold')
    ax.legend(loc='upper right', frameon=True, markerscale=1.5)
    ax.set_xlim(0, num_rounds)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"{dataset_key}_{model}_split_{num_rounds}r.png"
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {filename}")
    return OUTPUT_DIR / filename


def main():
    print("="*50)
    print("Generating Small Client Impact Figures (All Rounds)")
    print("="*50)
    
    total = 0
    for dataset_key in DATASETS:
        print(f"\n{DATASETS[dataset_key]['title']}:")
        for model in MODELS:
            for num_rounds in ROUNDS:
                if generate_accuracy_single(dataset_key, model, num_rounds):
                    total += 1
                if generate_splitpoint_single(dataset_key, model, num_rounds):
                    total += 1
    
    print(f"\n{'='*50}")
    print(f"Generated {total} figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
