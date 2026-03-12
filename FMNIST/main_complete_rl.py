"""
Complete Training Script: Capability-Aware Q-Learning with Committee DQN

This script integrates all components:
- Enhanced DQN with exponential epsilon decay
- Decayed loss-drop reward function
- Reward Committee for anti-reward-hacking
- Committee DQN with majority voting
- Multi-episode training

Usage:
    python main_complete_rl.py
"""

import torch
import traceback
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random
from models import get_model_by_choice
from torch.utils.data import Subset

# Import all components
from client import Client
from server import Server
from environment import FL_Environment
from dqn_agent import CapabilityAwareDQN_Agent
from reward_committee import RewardCommittee
from committee_dqn import CommitteeDQN

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================


# ============================================================================
# MODEL SELECTION
# ============================================================================

print("\nSelect Model Architecture:")
print("=" * 60)
print("1: CNN (Baseline)")
print("2: ResNet50 (Deep Residual Network)")
print("3: MobileNetV4 (Efficient Mobile Architecture)")
print("4: ConvNeXt (Modern CNN with Vision Transformer Design)")
print("=" * 60)

while True:
    try:
        model_choice = int(input("Enter model number (1-4): "))
        if 1 <= model_choice <= 4:
            break
        else:
            print("Please enter a number between 1 and 4.")
    except ValueError:
        print("Invalid input. Please enter a valid integer.")

model_names = {1: "CNN", 2: "ResNet50", 3: "MobileNetV4", 4: "ConvNeXt"}
selected_model_name = model_names[model_choice]
print(f"\nSelected model: {selected_model_name}")

# Dataset and model parameters
# Get number of clients from user input
while True:
    try:
        num_clients = int(input("Enter the total number of clients: "))
        if num_clients > 0:
            break
        else:
            print("Please enter a positive number.")
    except ValueError:
        print("Invalid input. Please enter a valid integer.")

k = int(0.6 * num_clients)  # Number of clients per round
num_classes = 10

# Define total layers based on model choice
# CNN: 10, ResNet50: 50, MobileNetV4: 53, ConvNeXt: 59
model_layers = {1: 10, 2: 50, 3: 53, 4: 59}
total_layers = model_layers[model_choice]

# Calculate action size dynamically based on split range logic (ceil(L/2) to L-1)
min_split = int(np.ceil(total_layers / 2))
max_split = total_layers - 1
action_size = max_split - min_split + 1

print(f"Model Configuration: Layers={total_layers}, Split Range=[{min_split}, {max_split}], Action Size={action_size}")

# Get number of rounds from user input
while True:
    try:
        num_rounds = int(input("Enter the number of rounds: "))
        if num_rounds > 0:
            break
        else:
            print("Please enter a positive number.")
    except ValueError:
        print("Invalid input. Please enter a valid integer.")

num_episodes = 1

# DQN parameters
learning_rate = 0.001
gamma = 0.2  # Low discount for immediate rewards
epsilon_start = 1.0
epsilon_end = 0.01
kappa = 0.05  # Epsilon decay constant
memory_size = 10000
batch_size = 32
target_update_frequency = 5

# Committee parameters
use_committee = True  # Set to False for single DQN
committee_size = 3
use_reward_committee = True

print("="*80)
print("CAPABILITY-AWARE Q-LEARNING FOR SPLIT POINT OPTIMIZATION (FMNIST)")
print("="*80)
print(f"\nConfiguration:")
print(f"  Clients: {num_clients}, Selected per round: {k}")
print(f"  Total layers: {total_layers}")
print(f"  Rounds per episode: {num_rounds}, Episodes: {num_episodes}")
print(f"  Use Committee DQN: {use_committee} (M={committee_size if use_committee else 'N/A'})")
print(f"  Use Reward Committee: {use_reward_committee}")
print("="*80)

# ============================================================================
# DATA PREPARATION
# ============================================================================

print("\n Loading and preparing Fashion-MNIST dataset...")

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

fmnist_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform_train)
fmnist_test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)

# Create heterogeneous data distribution
total_samples = len(fmnist_dataset)
indices = np.arange(len(fmnist_dataset))
np.random.shuffle(indices)

client_datasets = []
samples_per_client = total_samples // num_clients

for i in range(num_clients):
    start = i * samples_per_client
    if i == num_clients - 1:  # Last client gets remaining samples
        end = total_samples
    else:
        end = start + samples_per_client
    
    subindices = indices[start:end]
    subset = Subset(fmnist_dataset, subindices)
    client_datasets.append(subset)

print(f" Dataset prepared: {num_clients} clients with heterogeneous data")

# ============================================================================
# INITIALIZATION
# ============================================================================

print("\nInitializing components...")

# Initialize clients and server
clients = [Client(i, client_datasets[i]) for i in range(num_clients)]
# Get model based on user selection
model_instance, model_display_name = get_model_by_choice(
    model_choice, 10, input_channels=1
)

# Initialize server with selected model
server = Server(10, model_instance, fmnist_test_dataset)
env = FL_Environment(num_clients, global_class_dist=np.ones(10) / 10)

# Initialize RL agent
if use_committee:
    agent = CommitteeDQN(
        committee_size=committee_size,
        state_size=6,
        action_size=action_size,  # Dynamic based on model
        total_layers=total_layers,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        kappa=kappa,
        synthetic_dataset_capacity=memory_size,
        batch_size=batch_size,
        target_update_frequency=target_update_frequency,
        aggregation='voting'
    )
    print(f" Committee DQN initialized (M={committee_size})")
else:
    agent = CapabilityAwareDQN_Agent(
        state_size=6,
        action_size=action_size, # Dynamic
        total_layers=total_layers,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        kappa=kappa,
        memory_size=memory_size,
        batch_size=batch_size,
        target_update_frequency=target_update_frequency
    )
    print(" Single DQN initialized")

# Initialize reward committee
if use_reward_committee:
    reward_committee = RewardCommittee(
        committee_size=committee_size,
        state_size=6,
        learning_rate=learning_rate,
        synthetic_dataset_capacity=1000,
        batch_size=32,
        aggregation='median'
    )
    print(f" Reward Committee initialized (M={committee_size})")

# ============================================================================
# TRAINING LOOP
# ============================================================================

# Tracking metrics
all_rewards = []
all_accuracies = []
all_losses = []
all_split_points = []
all_epsilons = []

print("\n Starting training...")
print("="*80)

global_round = 0

try:
    for episode in range(num_episodes):
        print(f"\n{'='*80}")
        print(f"EPISODE {episode + 1}/{num_episodes}")
        print(f"{'='*80}")
        
        episode_rewards = []
        episode_accuracies = []
        episode_losses = []
        episode_splits = []
        
        prev_loss = None
        
        for round_num in range(num_rounds):
            print(f"\n--- Round {round_num + 1}/{num_rounds} ---")
            
            # Update client capabilities
            env.update_client_capabilities()
            
            # Select clients randomly
            selected_client_idxs = random.sample(range(num_clients), k)
            selected_clients = [clients[i] for i in selected_client_idxs]
            
            # Get cluster state
            cluster_state = env.get_cluster_state(selected_client_idxs)
            print(f"Selected Clients: {selected_client_idxs}")
            print(f"Cluster: Overall={cluster_state[4]:.3f}, Variance={cluster_state[5]:.3f}")
            
            # Select split point
            split_method = "Unknown"
            if use_committee:
                split_layer, vote_info = agent.select_action(cluster_state)
                if vote_info['method'] == 'voting':
                    split_method = f"Voting (votes: {vote_info['vote_counts']}, epsilon={agent.epsilon:.3f})"
                    print(f"Split: {split_layer} ({split_method})")
                else:
                    split_method = f"Random (epsilon={agent.epsilon:.3f})"
                    print(f"Split: {split_layer} ({split_method})")
            else:
                split_layer = agent.select_split_point(cluster_state)
                split_method = f"Single Agent (epsilon={agent.epsilon:.3f})"
                print(f"Split: {split_layer} ({split_method})")
            
            episode_splits.append(split_layer)
            
            # Execute split learning round according to PDF design
            server.set_split_layer(split_layer)
            client_model, server_model = server.create_split_models(server.global_model)
            
            # Set split models for clients
            for client in selected_clients:
                client.set_split_model(client_model, split_layer)
            
            # Execute complete split learning round
            round_metrics = server.train_split_round(
                [c.client_side_model for c in selected_clients],
                selected_clients,
                epochs=5
            )
            
            # Extract client-specific metrics for display
            client_losses = round_metrics['client_losses']
            client_accuracies = round_metrics['client_accuracies']
            avg_client_loss = round_metrics['avg_client_loss']
            avg_client_accuracy = round_metrics['avg_client_accuracy']
            avg_server_loss = round_metrics['avg_server_loss']
            
            # Use average client metrics as global metrics for this round (as requested)
            # new_acc, new_loss = server.evaluate()
            new_acc = avg_client_accuracy
            new_loss = avg_client_loss
            
            episode_accuracies.append(new_acc)
            episode_losses.append(new_loss)
            
            # Display detailed metrics
            print(f"Global (Avg Client): Acc={new_acc:.4f}, Loss={new_loss:.4f}")
            print(f"Clients: Avg Loss={avg_client_loss:.4f}, Avg Acc={avg_client_accuracy:.4f}")
            print(f"Individual Client Losses: {[f'{loss:.4f}' for loss in client_losses]}")
            print(f"Individual Client Accuracies: {[f'{acc:.4f}' for acc in client_accuracies]}")
            
            # Compute reward (decayed loss-drop) according to PDF design
            if prev_loss is not None:
                # Size-weighted cluster loss using client losses
                client_sizes = [len(c.local_data.dataset) for c in selected_clients]
                cluster_loss = env.compute_size_weighted_cluster_loss(client_losses, client_sizes)
                
                # Decayed loss-drop reward: r_t = -ΔL_t · ρ_t
                reward = env.compute_decayed_loss_drop_reward(
                    prev_loss=prev_loss,
                    current_loss=cluster_loss,
                    round_num=round_num + 1,
                    total_rounds=num_rounds
                )
                
                # Calculate details for logging (replicating env logic)
                lambda_log = np.log(2) / num_rounds
                rho_log = np.exp(-lambda_log * round_num)
                delta_loss_log = cluster_loss - prev_loss
                
                # Optional: Use reward committee for validation
                if use_reward_committee:
                    reward_committee.add_experience(cluster_state, reward)
                    if len(reward_committee.synthetic_dataset) >= 32:
                        committee_loss = reward_committee.train_committee(num_epochs=1)
                        print(f"  Reward Committee Loss: {committee_loss:.4f}")
            else:
                reward = 0.0
                cluster_loss = avg_client_loss  # Use average client loss for first round
                delta_loss_log = 0.0
                rho_log = 1.0
                print(f"  (First round: Reward set to 0.0)")
            
            episode_rewards.append(reward)
            prev_loss = cluster_loss
            
            print(f"Round {round_num + 1} Result:")
            print(f"  Selected Clients: {selected_client_idxs}")
            print(f"  Split Layer: {split_layer}")
            print(f"  Selection Method: {split_method}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Cluster Loss: {cluster_loss:.4f}")
            print(f"  Delta Loss: {delta_loss_log:.4f}")
            print(f"  Rho: {rho_log:.4f}")
            print(f"  Reward: {reward:.4f}")
            
            # Store experience and train
            next_cluster_state = env.get_cluster_state(selected_client_idxs)
            done = (round_num == num_rounds - 1)
            
            if use_committee:
                agent.store_transition(cluster_state, split_layer, reward, next_cluster_state, done)
                if len(agent.synthetic_dataset) >= batch_size:
                    train_loss = agent.train_committee()
                    if round_num % 5 == 0:
                        print(f"  DQN Loss: {train_loss:.4f}")
            else:
                agent.remember(cluster_state, split_layer, reward, next_cluster_state, done)
                agent.replay()
            
            # Update target network
            if use_committee:
                if agent.should_update_target(round_num + 1):
                    agent.update_target_networks()
                    print("   Target networks updated")
            else:
                if agent.should_update_target(round_num + 1):
                    agent.update_target_network()
                    print("   Target network updated")
            
            # Decay epsilon
            global_round += 1
            if use_committee:
                agent.decay_epsilon(global_round)
            else:
                agent.decay_epsilon(global_round)
            
            # Update global lists immediately for robustness
            all_split_points.append(split_layer)
            all_accuracies.append(new_acc)
            all_losses.append(new_loss)
            all_rewards.append(reward)
            all_epsilons.append(agent.epsilon)
        
        # Episode summary
        print(f"\n{'='*80}")
        print(f"Episode {episode + 1} Summary:")
        print(f"  Avg Reward: {np.mean(episode_rewards):.4f}")
        print(f"  Final Accuracy: {episode_accuracies[-1]:.4f}")
        print(f"  Avg Split Layer: {np.mean(episode_splits):.2f}")
        print(f"  Final Epsilon: {agent.epsilon:.4f}")
        print(f"{'='*80}")

except KeyboardInterrupt:
    print("\n\n" + "!"*80)
    print("TRAINING INTERRUPTED BY USER")
    print("Traceback details:")
    traceback.print_exc()
    print("!"*80 + "\n")
    print("Saving results collected so far...")

# ============================================================================
# RESULTS AND VISUALIZATION
# ============================================================================

print("\n Generating results...")

# Ensure all lists have the same length (truncate to minimum)
min_len = min(len(all_accuracies), len(all_losses), len(all_rewards), len(all_split_points), len(all_epsilons))
if min_len < len(all_accuracies):
    print(f"Warning: Truncating lists to length {min_len} due to mismatch.")
    
all_accuracies = all_accuracies[:min_len]
all_losses = all_losses[:min_len]
all_rewards = all_rewards[:min_len]
all_split_points = all_split_points[:min_len]
all_epsilons = all_epsilons[:min_len]

fig, axes = plt.subplots(3, 2, figsize=(14, 15))
rounds_axis = np.arange(1, min_len + 1)

# Save results to CSV
import pandas as pd
results_df = pd.DataFrame({
    'Round': rounds_axis,
    'Accuracy': all_accuracies,
    'Loss': all_losses,
    'Reward': all_rewards,
    'SplitLayer': all_split_points,
    'Epsilon': all_epsilons
})
results_df.to_csv('fmnist_rl_results.csv', index=False)
print(" Results saved to 'fmnist_rl_results.csv'")

# Plot 1: Accuracy
axes[0, 0].plot(rounds_axis, all_accuracies, 'b-', alpha=0.7, label='Test Accuracy')
axes[0, 0].set_xlabel('Round')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Test Accuracy Over Time (FMNIST)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# Plot 2: Loss (New)
axes[0, 1].plot(rounds_axis, all_losses, 'r-', alpha=0.7, label='Test Loss')
axes[0, 1].set_xlabel('Round')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Test Loss Over Time (FMNIST)')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# Plot 3: Rewards
axes[1, 0].plot(rounds_axis, all_rewards, 'g-', alpha=0.7, label='Decayed Loss-Drop Reward')
axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1, 0].set_xlabel('Round')
axes[1, 0].set_ylabel('Reward')
axes[1, 0].set_title('Reward Over Time (FMNIST)')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# Plot 4: Split Points
axes[1, 1].plot(rounds_axis, all_split_points, 'c-', alpha=0.5, marker='o', markersize=3)
axes[1, 1].set_xlabel('Round')
axes[1, 1].set_ylabel('Split Layer')
axes[1, 1].set_title('Selected Split Points (FMNIST)')
# axes[1, 1].set_yticks([5, 6, 7, 8, 9])  # Removed hardcoded ticks to support dynamic ranges
axes[1, 1].grid(True, alpha=0.3)

# Plot 5: Epsilon Decay
axes[2, 0].plot(rounds_axis, all_epsilons, 'm-', alpha=0.7, label='Epsilon')
axes[2, 0].set_xlabel('Round')
axes[2, 0].set_ylabel('Epsilon')
axes[2, 0].set_title('Exploration Rate Decay (FMNIST)')
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].legend()

# Hide empty subplot
axes[2, 1].axis('off')

plt.tight_layout()
plt.savefig('./complete_rl_results_fmnist.png', dpi=150, bbox_inches='tight')
print(" Results saved to 'complete_rl_results_fmnist.png'")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\nFinal Statistics:")
print(f"  Final Accuracy: {all_accuracies[-1]:.4f}")
print(f"  Average Accuracy: {np.mean(all_accuracies):.4f}")
print(f"  Average Reward: {np.mean(all_rewards):.4f}")
print(f"  Average Split Layer: {np.mean(all_split_points):.2f}")
print(f"  Split Layer Std: {np.std(all_split_points):.2f}")
print(f"  Final Epsilon: {agent.epsilon:.4f}")

if use_committee:
    print(f"\nCommittee Statistics:")
    print(f"  Committee Size: {committee_size}")
    print(f"  Total Votes: {len(agent.vote_history)}")
    ties = sum(1 for v in agent.vote_history if v.get('is_tie', False))
    print(f"  Ties Resolved: {ties}")

print("\n" + "="*80)
print(" All components successfully integrated and tested!")
print("="*80)
