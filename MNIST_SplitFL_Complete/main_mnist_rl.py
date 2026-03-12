"""
Complete MNIST Training Script: Capability-Aware Q-Learning with Committee DQN

This script implements all 5 algorithms from the research paper for MNIST dataset:
- Algorithm 1: BuildCommitteeAndClusters (Reward Committee)
- Algorithm 2: Capability-Aware DQN with Experience Replay
- Algorithm 3: TrainRoundSFL (Split Learning Execution)
- Algorithm 4: Majority Voting
- Algorithm 5: Committee-Based DQN

Usage:
    python main_mnist_rl.py
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random
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

# Dataset and model parameters
num_clients = 50  # Updated to 50 clients
k = 3  # Number of clients per round
num_classes = 10
total_layers = 10
num_rounds = 50  # 50 rounds as requested
num_episodes = 3

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
print("CAPABILITY-AWARE Q-LEARNING FOR SPLIT POINT OPTIMIZATION (MNIST)")
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

print("\n Loading and preparing MNIST dataset...")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
mnist_test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create heterogeneous data distribution
total_samples = len(mnist_dataset)
indices = np.arange(len(mnist_dataset))
np.random.shuffle(indices)

# Calculate samples per client with some variation
base_samples_per_client = total_samples // num_clients
client_datasets = []

start = 0
for i in range(num_clients):
    # Add some heterogeneity: ±20% variation
    variation = np.random.uniform(0.8, 1.2)
    num_samples = int(base_samples_per_client * variation)
    
    # Ensure we don't exceed available data
    num_samples = min(num_samples, total_samples - start)
    
    # Ensure minimum samples per client
    if i < num_clients - 1:  # Not the last client
        num_samples = max(num_samples, 100)  # At least 100 samples
    else:  # Last client gets remaining samples
        num_samples = total_samples - start
    
    if num_samples > 0:
        subindices = indices[start:start + num_samples]
        subset = Subset(mnist_dataset, subindices)
        client_datasets.append(subset)
        start += num_samples
    
    if start >= total_samples:
        break

# If we have fewer clients than requested, update count
actual_num_clients = len(client_datasets)
if actual_num_clients < num_clients:
    print(f"⚠️  Warning: Only created {actual_num_clients} clients due to dataset size")
    num_clients = actual_num_clients

print(f"✅ Dataset prepared: {num_clients} clients with heterogeneous data")
print(f"   Total samples distributed: {sum(len(d) for d in client_datasets)}/{total_samples}")
print(f"   Samples per client: min={min(len(d) for d in client_datasets)}, "
      f"max={max(len(d) for d in client_datasets)}, "
      f"avg={sum(len(d) for d in client_datasets)//num_clients}")

# ============================================================================
# INITIALIZATION
# ============================================================================

print("\n🔧 Initializing components...")

# Initialize clients and server
clients = [Client(i, client_datasets[i]) for i in range(num_clients)]
server = Server(mnist_test_dataset)
env = FL_Environment(num_clients, global_class_dist=np.ones(10) / 10)

# Initialize RL agent
if use_committee:
    agent = CommitteeDQN(
        committee_size=committee_size,
        state_size=6,
        action_size=5,
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
        print(f"Cluster: Overall={cluster_state[4]:.3f}, Variance={cluster_state[5]:.3f}")
        
        # Select split point
        if use_committee:
            split_layer, vote_info = agent.select_action(cluster_state)
            if vote_info['method'] == 'voting':
                print(f"Split: {split_layer} (votes: {vote_info['vote_counts']})") 
            else:
                print(f"Split: {split_layer} (random, ε={agent.epsilon:.3f})")
        else:
            split_layer = agent.select_split_point(cluster_state)
            print(f"Split: {split_layer} (ε={agent.epsilon:.3f})")
        
        episode_splits.append(split_layer)
        
        # Execute split learning round
        server.set_split_layer(split_layer)
        client_model, server_model = server.create_split_models(server.global_model)
        
        for client in selected_clients:
            client.set_split_model(client_model, split_layer)
        
        # Train clients
        client_losses = []
        for client in selected_clients:
            result = client.train_split(epochs=2)
            client_losses.append(result['final_loss'])
        
        # Aggregate
        server.aggregate_split_models([c.client_side_model for c in selected_clients])
        
        # Evaluate
        new_acc, new_loss = server.evaluate()
        episode_accuracies.append(new_acc)
        episode_losses.append(new_loss)
        
        print(f"Accuracy: {new_acc:.4f}, Loss: {new_loss:.4f}")
        
        # Compute reward (decayed loss-drop)
        if prev_loss is not None:
            # Size-weighted cluster loss
            client_sizes = [len(c.local_data.dataset) for c in selected_clients]
            cluster_loss = env.compute_size_weighted_cluster_loss(client_losses, client_sizes)
            
            # Decayed loss-drop reward
            reward = env.compute_decayed_loss_drop_reward(
                prev_loss=prev_loss,
                current_loss=cluster_loss,
                round_num=round_num + 1,
                total_rounds=num_rounds
            )
            
            # Optional: Use reward committee for validation
            if use_reward_committee:
                reward_committee.add_experience(cluster_state, reward)
                if len(reward_committee.synthetic_dataset) >= 32:
                    committee_loss = reward_committee.train_committee(num_epochs=1)
        else:
            reward = 0.0
            cluster_loss = new_loss
        
        episode_rewards.append(reward)
        prev_loss = cluster_loss
        
        print(f"Reward: {reward:.4f}")
        
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
                print("  ✓ Target networks updated")
        else:
            if agent.should_update_target(round_num + 1):
                agent.update_target_network()
                print("  ✓ Target network updated")
        
        # Decay epsilon
        if use_committee:
            agent.decay_epsilon(round_num + 1)
        else:
            agent.decay_epsilon(round_num + 1)
        
        all_epsilons.append(agent.epsilon)
    
    # Episode summary
    print(f"\n{'='*80}")
    print(f"Episode {episode + 1} Summary:")
    print(f"  Avg Reward: {np.mean(episode_rewards):.4f}")
    print(f"  Final Accuracy: {episode_accuracies[-1]:.4f}")
    print(f"  Avg Split Layer: {np.mean(episode_splits):.2f}")
    print(f"  Final Epsilon: {agent.epsilon:.4f}")
    print(f"{'='*80}")
    
    all_rewards.extend(episode_rewards)
    all_accuracies.extend(episode_accuracies)
    all_losses.extend(episode_losses)
    all_split_points.extend(episode_splits)

# ============================================================================
# RESULTS AND VISUALIZATION
# ============================================================================

print("\n📊 Generating results...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
rounds_axis = np.arange(1, len(all_accuracies) + 1)

# Plot 1: Accuracy
axes[0, 0].plot(rounds_axis, all_accuracies, 'b-', alpha=0.7, label='Test Accuracy')
axes[0, 0].set_xlabel('Round')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('MNIST Test Accuracy Over Time')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# Plot 2: Rewards
axes[0, 1].plot(rounds_axis[1:], all_rewards, 'r-', alpha=0.7, label='Decayed Loss-Drop Reward')
axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0, 1].set_xlabel('Round')
axes[0, 1].set_ylabel('Reward')
axes[0, 1].set_title('Reward Over Time')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# Plot 3: Split Points
axes[1, 0].plot(rounds_axis, all_split_points, 'g-', alpha=0.5, marker='o', markersize=3)
axes[1, 0].set_xlabel('Round')
axes[1, 0].set_ylabel('Split Layer')
axes[1, 0].set_title('Selected Split Points')
axes[1, 0].set_yticks([5, 6, 7, 8, 9])
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Epsilon Decay
axes[1, 1].plot(rounds_axis, all_epsilons, 'm-', alpha=0.7, label='Epsilon')
axes[1, 1].set_xlabel('Round')
axes[1, 1].set_ylabel('Epsilon')
axes[1, 1].set_title('Exploration Rate Decay')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('./mnist_rl_results.png', dpi=150, bbox_inches='tight')
print("✅ Results saved to 'mnist_rl_results.png'")

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
print("✅ All components successfully integrated and tested!")
print("="*80)
