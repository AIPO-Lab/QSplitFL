"""
Test script to verify the fixes work with reduced complexity
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from torch.utils.data import Subset

# Import components
from client import Client
from server import Server
from environment import FL_Environment

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("="*80)
print("TESTING FIXED SPLIT LEARNING IMPLEMENTATION")
print("="*80)

# Simplified configuration
num_clients = 3
k = 2  # Number of clients per round
num_classes = 10
total_layers = 10
num_rounds = 2  # Reduced for testing
num_episodes = 1

print(f"\nConfiguration:")
print(f"  Clients: {num_clients}, Selected per round: {k}")
print(f"  Total layers: {total_layers}")
print(f"  Rounds per episode: {num_rounds}, Episodes: {num_episodes}")
print("="*80)

# Data preparation
print("\n Loading and preparing CIFAR-10 dataset...")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
cifar_test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Create simplified data distribution
total_samples = len(cifar_dataset)
indices = np.arange(len(cifar_dataset))
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
    subset = Subset(cifar_dataset, subindices)
    client_datasets.append(subset)

print(f" Dataset prepared: {num_clients} clients with heterogeneous data")

# Initialize components
print("\nInitializing components...")

# Initialize clients and server
clients = [Client(i, client_datasets[i]) for i in range(num_clients)]
server = Server(cifar_test_dataset)
env = FL_Environment(num_clients, global_class_dist=np.ones(10) / 10)

print("Components initialized successfully!")

# Test split learning workflow
print("\nTesting split learning workflow...")

for episode in range(num_episodes):
    print(f"\n{'='*80}")
    print(f"EPISODE {episode + 1}/{num_episodes}")
    print(f"{'='*80}")
    
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
        
        # Select split point (fixed for testing)
        split_layer = 6  # Fixed split layer for testing
        print(f"Split: {split_layer}")
        
        # Execute split learning round
        server.set_split_layer(split_layer)
        client_model, server_model = server.create_split_models(server.global_model)
        
        # Set split models for clients
        for client in selected_clients:
            client.set_split_model(client_model, split_layer)
        
        # Train clients locally
        client_losses = []
        client_accuracies = []
        
        for client in selected_clients:
            result = client.train_split(epochs=1)  # Reduced epochs for testing
            client_losses.append(result['final_loss'])
            client_accuracies.append(result['final_accuracy'])
            print(f"Client {client.client_id}: Loss={result['final_loss']:.4f}, Acc={result['final_accuracy']:.4f}")
        
        # Aggregate client models
        server.aggregate_split_models([c.client_side_model for c in selected_clients])
        
        # Evaluate global model
        new_acc, new_loss = server.evaluate()
        print(f"Global: Acc={new_acc:.4f}, Loss={new_loss:.4f}")
        
        # Compute reward
        client_sizes = [len(c.local_data.dataset) for c in selected_clients]
        cluster_loss = env.compute_size_weighted_cluster_loss(client_losses, client_sizes)
        
        if round_num > 0:
            reward = env.compute_decayed_loss_drop_reward(
                prev_loss=prev_loss,
                current_loss=cluster_loss,
                round_num=round_num + 1,
                total_rounds=num_rounds
            )
        else:
            reward = 0.0
        
        prev_loss = cluster_loss
        print(f"Cluster Loss: {cluster_loss:.4f}, Reward: {reward:.4f}")

print("\n" + "="*80)
print("TEST COMPLETED!")
print("="*80)