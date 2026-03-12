"""
Generate comparison plots for FMNIST and CIFAR-100
Creates comparison by clients, by rounds, and for each round setting
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_results(data_dir, num_clients, num_rounds, dataset):
    """Load results CSV file"""
    filename = f"{dataset}_results_clients{num_clients}_rounds{num_rounds}.csv"
    filepath = os.path.join(data_dir, filename)
    return pd.read_csv(filepath)

def create_comparison_by_clients(data_dir, dataset, output_dir):
    """Create comparison plots showing effect of different client counts"""
    
    client_counts = [5, 10, 100, 200]
    round_setting = 100  # Use 100 rounds for comparison
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{dataset.upper()}: Impact of Client Count (100 Rounds)', 
                 fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, num_clients in enumerate(client_counts):
        df = load_results(data_dir, num_clients, round_setting, dataset)
        
        # Accuracy
        axes[0, 0].plot(df['Round'], df['Accuracy'], label=f'{num_clients} clients',
                       color=colors[idx], linewidth=2, alpha=0.7)
        
        # Loss
        axes[0, 1].plot(df['Round'], df['Loss'], label=f'{num_clients} clients',
                       color=colors[idx], linewidth=2, alpha=0.7)
        
        # Reward
        axes[1, 0].plot(df['Round'], df['Reward'], label=f'{num_clients} clients',
                       color=colors[idx], linewidth=2, alpha=0.7)
        
        # Split layers
        axes[1, 1].plot(df['Round'], df['SplitLayer'], label=f'{num_clients} clients',
                       color=colors[idx], linewidth=1.5, alpha=0.6)
    
    # Format subplots
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Model Performance Across Client Scales')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Loss Minimization Trends')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].set_title('Reward Over Time')
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Split Layer')
    axes[1, 1].set_title('Split Point Selection')
    axes[1, 1].set_yticks([5, 6, 7, 8, 9])
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    filename = f"{dataset}_comparison_by_clients.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created: {filename}")

def create_comparison_by_rounds(data_dir, dataset, output_dir):
    """Create comparison plots showing effect of different round counts"""
    
    round_counts = [10, 20, 50, 100]
    client_setting = 10  # Use 10 clients for comparison
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{dataset.upper()}: Impact of Training Rounds (10 Clients)', 
                 fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, num_rounds in enumerate(round_counts):
        df = load_results(data_dir, client_setting, num_rounds, dataset)
        
        # Accuracy
        axes[0, 0].plot(df['Round'], df['Accuracy'], label=f'{num_rounds} rounds',
                       color=colors[idx], linewidth=2, alpha=0.7, marker='o', 
                       markersize=4 if num_rounds <= 20 else 0)
        
        # Loss
        axes[0, 1].plot(df['Round'], df['Loss'], label=f'{num_rounds} rounds',
                       color=colors[idx], linewidth=2, alpha=0.7, marker='o',
                       markersize=4 if num_rounds <= 20 else 0)
        
        # Reward
        axes[1, 0].plot(df['Round'], df['Reward'], label=f'{num_rounds} rounds',
                       color=colors[idx], linewidth=2, alpha=0.7, marker='o',
                       markersize=4 if num_rounds <= 20 else 0)
        
        # Final accuracy bar chart
        axes[1, 1].bar(idx, df['Accuracy'].iloc[-1], color=colors[idx], 
                      alpha=0.7, label=f'{num_rounds} rounds')
    
    # Format subplots
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Impact of Training Duration on Accuracy')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Loss Optimization Over Training Rounds')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].set_title('Reward Over Time')
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    axes[1, 1].set_xlabel('Training Rounds')
    axes[1, 1].set_ylabel('Final Accuracy (%)')
    axes[1, 1].set_title('Final Accuracy by Round Count')
    axes[1, 1].set_xticks(range(len(round_counts)))
    axes[1, 1].set_xticklabels([f'{r}' for r in round_counts])
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    filename = f"{dataset}_comparison_by_rounds.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created: {filename}")

def create_comparison_by_clients_for_rounds(data_dir, dataset, output_dir, round_setting):
    """Create comparison plot for specific round setting"""
    
    client_counts = [5, 10, 100, 200]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{dataset.upper()}: Client Count Comparison ({round_setting} Rounds)', 
                 fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, num_clients in enumerate(client_counts):
        df = load_results(data_dir, num_clients, round_setting, dataset)
        
        # Accuracy
        axes[0, 0].plot(df['Round'], df['Accuracy'], label=f'{num_clients} clients',
                       color=colors[idx], linewidth=2, alpha=0.7)
        
        # Loss
        axes[0, 1].plot(df['Round'], df['Loss'], label=f'{num_clients} clients',
                       color=colors[idx], linewidth=2, alpha=0.7)
        
        # Reward
        axes[1, 0].plot(df['Round'], df['Reward'], label=f'{num_clients} clients',
                       color=colors[idx], linewidth=2, alpha=0.7)
        
        # Split layers
        axes[1, 1].plot(df['Round'], df['SplitLayer'], label=f'{num_clients} clients',
                       color=colors[idx], linewidth=1.5, alpha=0.6)
    
    # Format subplots
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Accuracy Convergence')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Loss Reduction')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].set_title('Reward Over Time')
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Split Layer')
    axes[1, 1].set_title('Split Point Selection')
    axes[1, 1].set_yticks([5, 6, 7, 8, 9])
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    filename = f"{dataset}_comparison_by_clients_{round_setting}rounds.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created: {filename}")

if __name__ == "__main__":
    print("="*70)
    print("Generating Comparison Plots for FMNIST and CIFAR-100")
    print("="*70)
    
    # FMNIST
    fmnist_dir = "./FMNIST_Results"
    print("\n📊 Generating FMNIST comparison plots...")
    create_comparison_by_clients(fmnist_dir, 'fmnist', fmnist_dir)
    create_comparison_by_rounds(fmnist_dir, 'fmnist', fmnist_dir)
    
    # Per-round comparisons
    for rounds in [10, 20, 50, 100]:
        create_comparison_by_clients_for_rounds(fmnist_dir, 'fmnist', fmnist_dir, rounds)
    
    # CIFAR-100
    cifar_dir = "./CIFAR-100_Results"
    print("\n📊 Generating CIFAR-100 comparison plots...")
    create_comparison_by_clients(cifar_dir, 'cifar100', cifar_dir)
    create_comparison_by_rounds(cifar_dir, 'cifar100', cifar_dir)
    
    # Per-round comparisons
    for rounds in [10, 20, 50, 100]:
        create_comparison_by_clients_for_rounds(cifar_dir, 'cifar100', cifar_dir, rounds)
    
    print("\n" + "="*70)
    print("✅ All comparison plots generated successfully!")
    print("="*70)
    print("\nGenerated plots:")
    print("  - comparison_by_clients.png (overall)")
    print("  - comparison_by_rounds.png (overall)")
    print("  - comparison_by_clients_10rounds.png")
    print("  - comparison_by_clients_20rounds.png")
    print("  - comparison_by_clients_50rounds.png")
    print("  - comparison_by_clients_100rounds.png")
