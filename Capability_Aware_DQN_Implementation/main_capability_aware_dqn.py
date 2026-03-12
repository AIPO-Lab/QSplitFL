"""
Main Training Script: Capability-Aware DQN for Split Learning
===============================================================

This script implements the complete training pipeline combining:
- Algorithm 6: Capability-Aware DQN with Experience Replay
- Algorithm 8: TrainRoundSFL (Split Learning Round Execution)

The system learns to select optimal split points based on client capabilities
to minimize federated learning loss while considering resource constraints.

Workflow:
1. Initialize environment, clients, server, DQN agent
2. FOR each round:
   a. Select clients
   b. Compute capability-aware state (6D)
   c. DQN selects split point (ε-greedy)
   d. Execute split learning round
   e. Compute decayed loss-drop reward
   f. Store transition in replay buffer
   g. Train DQN
   h. Decay exploration
3. Visualize results and analyze learned policy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import random
import logging
from typing import List, Dict
import os

# Import our modules
from environment_capability_aware import CapabilityAwareEnvironment
from dqn_split_agent import CapabilityAwareDQNAgent
from split_learning_utils import SplitModel, evaluate_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    """
    Simple CNN for MNIST/Fashion-MNIST designed for split learning.
    
    Architecture (Sequential layers for easy splitting):
        Layer 0: Conv1 (1→16) + ReLU + Pool
        Layer 1: Conv2 (16→32) + ReLU + Pool + Flatten
        Layer 2: FC1 (1568→256) + ReLU
        Layer 3: FC2 (256→128) + ReLU
        Layer 4: FC3 (128→64) + ReLU
        Layer 5: FC4 (64→10) - Output
    
    Total: 6 layers (can split from layer 3 onwards after flattening)
    """
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Build as Sequential for easy layer extraction
        self.features = nn.Sequential(
            # Layer 0: First conv block
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Layer 1: Second conv block + flatten
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        self.classifier = nn.Sequential(
            # Layer 2: FC1
            nn.Linear(32 * 7 * 7, 256),
            nn.ReLU(),
            
            # Layer 3: FC2
            nn.Linear(256, 128),
            nn.ReLU(),
            
            # Layer 4: FC3
            nn.Linear(128, 64),
            nn.ReLU(),
            
            # Layer 5: Output
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_client_datasets(dataset, num_clients: int, heterogeneous: bool = True):
    """
    Create client datasets with optional non-IID distribution.
    
    Args:
        dataset: Full dataset
        num_clients: Number of clients
        heterogeneous: If True, create non-IID distribution
    
    Returns:
        client_datasets: List of client datasets
    """
    total_samples = len(dataset)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    client_datasets = []
    start = 0
    
    if heterogeneous:
        # Non-IID: variable data sizes
        for i in range(num_clients):
            proportion = np.random.uniform(0.15, 0.25)
            num_samples = int(proportion * total_samples)
            num_samples = min(num_samples, total_samples - start)
            
            subset_indices = indices[start:start + num_samples]
            client_datasets.append(Subset(dataset, subset_indices))
            
            start += num_samples
            if start >= total_samples:
                break
    else:
        # IID: equal data sizes
        samples_per_client = total_samples // num_clients
        for i in range(num_clients):
            subset_indices = indices[i * samples_per_client:(i + 1) * samples_per_client]
            client_datasets.append(Subset(dataset, subset_indices))
    
    return client_datasets


def train_round_sfl(
    clients: List[Dict],
    split_layer: int,
    full_model: nn.Module,
    device: str,
    epochs: int = 1
) -> Dict:
    """
    Execute one Split Federated Learning round (Algorithm 8).
    
    Args:
        clients: List of client dictionaries with data loaders and optimizers
        split_layer: Split point selected by DQN
        full_model: Complete model to split
        device: Device to use
        epochs: Local training epochs
    
    Returns:
        metrics: Dictionary with training metrics and losses
    """
    logger.info(f"Executing SFL round with split layer {split_layer}")
    
    # Create split model
    split_model = SplitModel(full_model, split_layer, device)
    
    # Train each client
    client_losses = []
    client_accuracies = []
    client_models = []
    
    for client_id, client_info in enumerate(clients):
        train_loader = client_info['train_loader']
        optimizer = client_info['optimizer']
        criterion = nn.CrossEntropyLoss()
        
        # Get client-side model
        client_model = split_model.get_client_model()
        server_model = split_model.get_server_model()
        
        # Train client
        client_model.train()
        server_model.eval()
        
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for epoch in range(epochs):
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                
                # Client forward (with gradients)
                smashed_data = client_model(data)
                
                # Server forward (enable gradients for backprop)
                # In real split learning, server would compute this and send gradients back
                server_model.train()  # Enable gradient tracking
                output = server_model(smashed_data)
                
                # Compute loss
                loss = criterion(output, target)
                
                # Backward (gradients flow through both client and server parts)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                epoch_correct += pred.eq(target.view_as(pred)).sum().item()
                epoch_total += data.size(0)
        
        avg_loss = epoch_loss / epoch_total
        avg_accuracy = epoch_correct / epoch_total
        
        client_losses.append(avg_loss)
        client_accuracies.append(avg_accuracy)
        client_models.append(client_model.state_dict())
        
        logger.debug(f"  Client {client_id}: loss={avg_loss:.4f}, acc={avg_accuracy:.4f}")
    
    # FedAvg aggregation (simple average)
    aggregated_state = {}
    for key in client_models[0].keys():
        aggregated_state[key] = torch.stack([
            client_models[i][key].float() for i in range(len(client_models))
        ]).mean(0)
    
    # Update client-side of full model
    split_model.get_client_model().load_state_dict(aggregated_state)
    
    return {
        'client_losses': client_losses,
        'client_accuracies': client_accuracies,
        'avg_loss': np.mean(client_losses),
        'avg_accuracy': np.mean(client_accuracies)
    }


def main():
    """Main training loop."""
    
    # ============================================================
    # CONFIGURATION
    # ============================================================
    
    # General settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    NUM_ROUNDS = 50
    
    # FL settings
    NUM_CLIENTS = 20  # Increased from 9 to 20 clients
    CLIENTS_PER_ROUND = 5  # Increased proportionally (was 3, now ~25% of clients)
    LOCAL_EPOCHS = 2
    
    # Model settings
    # SimpleCNN has: features(conv blocks) + classifier(4 FC layers) = ~8 sequential ops
    # But for split learning, we count as: 2 blocks (features + classifier sublayers)
    TOTAL_LAYERS = 8  # features block + 4 FC layers in classifier = can split at FC layers
    
    # DQN settings
    STATE_SIZE = 6
    LEARNING_RATE = 0.001
    GAMMA = 0.2  # Small discount (immediate focus)
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.95
    BUFFER_CAPACITY = 5000
    BATCH_SIZE = 32
    TARGET_UPDATE_FREQ = 10
    
    # Set seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    logger.info("="*60)
    logger.info("CAPABILITY-AWARE DQN FOR SPLIT LEARNING")
    logger.info("="*60)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Num clients: {NUM_CLIENTS}")
    logger.info(f"Num rounds: {NUM_ROUNDS}")
    logger.info(f"Total layers: {TOTAL_LAYERS}")
    
    # ============================================================
    # DATA PREPARATION
    # ============================================================
    
    logger.info("\n--- Loading MNIST dataset ---")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Create client datasets
    client_datasets = create_client_datasets(train_dataset, NUM_CLIENTS, heterogeneous=True)
    
    logger.info(f"Created {len(client_datasets)} client datasets")
    for i, dataset in enumerate(client_datasets):
        logger.info(f"  Client {i}: {len(dataset)} samples")
    
    # ============================================================
    # INITIALIZATION
    # ============================================================
    
    logger.info("\n--- Initializing components ---")
    
    # Environment
    env = CapabilityAwareEnvironment(
        num_clients=NUM_CLIENTS,
        capability_profiles="mixed",
        seed=SEED
    )
    
    # DQN Agent
    agent = CapabilityAwareDQNAgent(
        state_size=STATE_SIZE,
        total_layers=TOTAL_LAYERS,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
        device=DEVICE
    )
    
    # Global model
    global_model = SimpleCNN().to(DEVICE)
    
    # Client optimizers and data loaders
    # Use actual number of datasets created (might be less than NUM_CLIENTS if data runs out)
    actual_num_clients = len(client_datasets)
    clients = []
    for i in range(actual_num_clients):
        train_loader = DataLoader(
            client_datasets[i], batch_size=64, shuffle=True
        )
        optimizer = optim.SGD(global_model.parameters(), lr=0.01, momentum=0.9)
        
        clients.append({
            'id': i,
            'train_loader': train_loader,
            'optimizer': optimizer
        })
    
    logger.info(f"Initialized {len(clients)} clients")
    
    # Update NUM_CLIENTS to match actual clients created
    NUM_CLIENTS = actual_num_clients
    logger.info(f"Using {NUM_CLIENTS} clients for training")
    
    # ============================================================
    # TRAINING LOOP
    # ============================================================
    
    logger.info("\n" + "="*60)
    logger.info("STARTING TRAINING")
    logger.info("="*60 + "\n")
    
    # Metrics tracking
    metrics = {
        'rounds': [],
        'test_accuracy': [],
        'test_loss': [],
        'rewards': [],
        'split_points': [],
        'epsilon': [],
        'td_loss': [],
        'q_values': []
    }
    
    prev_loss = None
    
    for round_num in range(1, NUM_ROUNDS + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"ROUND {round_num}/{NUM_ROUNDS}")
        logger.info(f"{'='*60}")
        
        # 1. Select clients randomly
        selected_indices = random.sample(range(NUM_CLIENTS), CLIENTS_PER_ROUND)
        selected_clients = [clients[i] for i in selected_indices]
        
        logger.info(f"Selected clients: {selected_indices}")
        
        # 2. Update client capabilities (simulate dynamics)
        if round_num > 1:
            env.update_client_capabilities(selected_indices)
        
        # 3. Compute cluster state (6D capability vector)
        cluster_state = env.get_cluster_state(selected_indices)
        
        logger.info(f"Cluster state:")
        logger.info(f"  CPU={cluster_state[0]:.3f}, Memory={cluster_state[1]:.3f}")
        logger.info(f"  Battery={cluster_state[2]:.3f}, Network={cluster_state[3]:.3f}")
        logger.info(f"  Overall={cluster_state[4]:.3f}, Std={cluster_state[5]:.3f}")
        
        # 4. DQN selects split point (ε-greedy)
        split_layer = agent.select_action(cluster_state, round_num)
        
        logger.info(f"Agent selected split layer: {split_layer}")
        logger.info(f"Exploration rate (ε): {agent.epsilon:.4f}")
        
        # 5. Execute Split FL round (Algorithm 8)
        train_metrics = train_round_sfl(
            selected_clients,
            split_layer,
            global_model,
            DEVICE,
            epochs=LOCAL_EPOCHS
        )
        
        logger.info(f"Training metrics:")
        logger.info(f"  Avg client loss: {train_metrics['avg_loss']:.4f}")
        logger.info(f"  Avg client accuracy: {train_metrics['avg_accuracy']:.4f}")
        
        # 6. Evaluate global model
        test_accuracy, test_loss = evaluate_model(
            global_model, test_loader, nn.CrossEntropyLoss(), DEVICE
        )
        
        logger.info(f"Global model evaluation:")
        logger.info(f"  Test accuracy: {test_accuracy:.4f}")
        logger.info(f"  Test loss: {test_loss:.4f}")
        
        # 7. Compute reward (decayed loss-drop)
        reward = env.compute_decayed_reward(test_loss, round_num)
        
        logger.info(f"Reward: {reward:.6f}")
        
        # 8. Store transition in replay buffer
        next_cluster_state = env.get_cluster_state(selected_indices)
        agent.store_transition(
            cluster_state,
            split_layer,
            reward,
            next_cluster_state,
            done=False
        )
        
        # 9. Train DQN
        td_loss = agent.train_step()
        if td_loss is not None:
            logger.info(f"TD loss: {td_loss:.6f}")
        
        # 10. Update target network periodically
        if round_num % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
        
        # 11. Decay exploration
        agent.decay_epsilon()
        
        # Store metrics
        metrics['rounds'].append(round_num)
        metrics['test_accuracy'].append(test_accuracy)
        metrics['test_loss'].append(test_loss)
        metrics['rewards'].append(reward)
        metrics['split_points'].append(split_layer)
        metrics['epsilon'].append(agent.epsilon)
        if td_loss is not None:
            metrics['td_loss'].append(td_loss)
        
        # Log progress
        logger.info(f"Round {round_num} complete")
        logger.info(f"  Accuracy: {test_accuracy:.4f}")
        logger.info(f"  Loss: {test_loss:.4f}")
        logger.info(f"  Reward: {reward:.6f}")
        logger.info(f"  Split: {split_layer}")
        logger.info(f"  ε: {agent.epsilon:.4f}")
    
    # ============================================================
    # RESULTS AND VISUALIZATION
    # ============================================================
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    
    # Print summaries
    env.print_summary()
    agent.print_summary()
    
    # Visualizations
    logger.info("\nGenerating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Test Accuracy
    axes[0, 0].plot(metrics['rounds'], metrics['test_accuracy'], 'b-', linewidth=2)
    axes[0, 0].set_title('Test Accuracy over Rounds', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Test Loss
    axes[0, 1].plot(metrics['rounds'], metrics['test_loss'], 'r-', linewidth=2)
    axes[0, 1].set_title('Test Loss over Rounds', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Rewards
    axes[0, 2].plot(metrics['rounds'], metrics['rewards'], 'g-', linewidth=2)
    axes[0, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 2].set_title('Rewards over Rounds', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Round')
    axes[0, 2].set_ylabel('Reward')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Split Points
    axes[1, 0].scatter(metrics['rounds'], metrics['split_points'], c=metrics['rounds'], cmap='viridis', s=50)
    axes[1, 0].set_title('Selected Split Points', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('Split Layer')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yticks(agent.valid_splits)
    
    # 5. Exploration Rate
    axes[1, 1].plot(metrics['rounds'], metrics['epsilon'], 'm-', linewidth=2)
    axes[1, 1].set_title('Exploration Rate (ε)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('ε')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. TD Loss
    if metrics['td_loss']:
        axes[1, 2].plot(range(1, len(metrics['td_loss'])+1), metrics['td_loss'], 'c-', linewidth=2)
        axes[1, 2].set_title('TD Loss', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Training Step')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/capability_aware_dqn_results.png', dpi=300, bbox_inches='tight')
    logger.info("Saved results to results/capability_aware_dqn_results.png")
    
    plt.show()
    
    # Final statistics
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    print(f"Initial accuracy: {metrics['test_accuracy'][0]:.4f}")
    print(f"Final accuracy: {metrics['test_accuracy'][-1]:.4f}")
    print(f"Accuracy improvement: {metrics['test_accuracy'][-1] - metrics['test_accuracy'][0]:.4f}")
    print(f"Initial loss: {metrics['test_loss'][0]:.4f}")
    print(f"Final loss: {metrics['test_loss'][-1]:.4f}")
    print(f"Loss reduction: {metrics['test_loss'][0] - metrics['test_loss'][-1]:.4f}")
    print(f"Total reward: {sum(metrics['rewards']):.4f}")
    print(f"Average reward: {np.mean(metrics['rewards']):.4f}")
    print("="*60)
    
    # Save model
    agent.save_model('results/dqn_model.pth')
    logger.info("Saved DQN model to results/dqn_model.pth")
    
    logger.info("\n✅ Training complete! Check results/ directory for outputs.")


if __name__ == "__main__":
    main()

