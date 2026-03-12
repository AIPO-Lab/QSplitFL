"""
Regenerate comparison plots with updated meaningful titles
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_results(data_dir, num_clients, num_rounds, dataset='mnist'):
    """Load results CSV file"""
    filename = f"{dataset}_results_clients{num_clients}_rounds{num_rounds}.csv"
    filepath = os.path.join(data_dir, filename)
    return pd.read_csv(filepath)

def regenerate_comparison_by_clients(data_dir, dataset, output_dir):
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
    
    # Format subplots with NEW meaningful titles
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
    
    print(f"✓ Regenerated: {filename}")

def regenerate_comparison_by_rounds(data_dir, dataset, output_dir):
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
    
    # Format subplots with NEW meaningful titles
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
    
    print(f"✓ Regenerated: {filename}")

if __name__ == "__main__":
    print("=" * 80)
    print("REGENERATING COMPARISON PLOTS WITH UPDATED TITLES")
    print("=" * 80)
    
    # MNIST
    mnist_dir = "c:/Users/nshadin/OneDrive - Kennesaw State University/QSplitFL/complete_rl_implementation/MNIST_Results"
    print("\n📊 Regenerating MNIST comparison plots...")
    regenerate_comparison_by_clients(mnist_dir, 'mnist', mnist_dir)
    regenerate_comparison_by_rounds(mnist_dir, 'mnist', mnist_dir)
    
    # CIFAR10
    cifar_dir = "c:/Users/nshadin/OneDrive - Kennesaw State University/QSplitFL/complete_rl_implementation/CIFAR10_Results"
    print("\n📊 Regenerating CIFAR10 comparison plots...")
    regenerate_comparison_by_clients(cifar_dir, 'cifar10', cifar_dir)
    regenerate_comparison_by_rounds(cifar_dir, 'cifar10', cifar_dir)
    
    print("\n" + "=" * 80)
    print("✅ ALL COMPARISON PLOTS REGENERATED WITH NEW TITLES")
    print("=" * 80)
    print("\nUpdated titles:")
    print("  - 'Accuracy Convergence' → 'Model Performance Across Client Scales'")
    print("  - 'Loss Reduction' → 'Training Loss Minimization Trends'")
    print("  - 'Accuracy Convergence' (by rounds) → 'Impact of Training Duration on Accuracy'")
    print("  - 'Loss Reduction' (by rounds) → 'Loss Optimization Over Training Rounds'")
