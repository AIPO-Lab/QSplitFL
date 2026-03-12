"""
Reward Committee for Anti-Reward-Hacking (Algorithm 5)

This module implements a committee of reward selectors to mitigate reward hacking
in the capability-aware Q-learning framework for split point optimization.

Architecture:
- Shared backbone (4 hidden layers): Learns stable, low-level representations
- M personalized heads (2 hidden layers each): Provide diversity in reward estimation
- Aggregation (mean/median): Reduces individual bias and noise

Reference: Algorithm 5 in the LaTeX document
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class SharedBackbone(nn.Module):
    """Shared shallow backbone for all committee members (4 hidden layers)."""
    
    def __init__(self, input_size=6, hidden_sizes=[64, 64, 32, 32]):
        """
        Initialize shared backbone.
        
        Args:
            input_size: Size of input state vector (6 for capability state)
            hidden_sizes: List of hidden layer sizes (default: 4 layers)
        """
        super(SharedBackbone, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        self.network = nn.Sequential(*layers)
        self.output_size = hidden_sizes[-1]
    
    def forward(self, x):
        """Forward pass through shared backbone."""
        return self.network(x)


class PersonalizedHead(nn.Module):
    """Personalized head for each committee member (2 hidden layers)."""
    
    def __init__(self, input_size=32, hidden_sizes=[16, 8], output_size=1):
        """
        Initialize personalized head.
        
        Args:
            input_size: Size of input from shared backbone
            hidden_sizes: List of hidden layer sizes (default: 2 layers)
            output_size: Output size (1 for scalar reward prediction)
        """
        super(PersonalizedHead, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Final output layer (no activation for regression)
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through personalized head."""
        return self.network(x)


class RewardCommittee:
    """
    Committee of reward selectors for anti-reward-hacking (Algorithm 5).
    
    Each committee member consists of:
    - Shared backbone f_s(·; φ_s): 4 hidden layers (fixed after initial training)
    - Personalized head h_{c,m}(·; φ_{c,m}): 2 hidden layers (cluster-specific)
    
    Output: S_{c,m}(s) = h_{c,m}(f_s(s; φ_s); φ_{c,m})
    """
    
    def __init__(self, committee_size=3, state_size=6, learning_rate=0.001,
                 synthetic_dataset_capacity=1000, batch_size=32, 
                 aggregation='median', fix_backbone_after=100):
        """
        Initialize Reward Committee.
        
        Args:
            committee_size: Number of committee members M (odd number recommended)
            state_size: Size of input state vector
            learning_rate: Learning rate for training
            synthetic_dataset_capacity: Max size of synthetic dataset (FIFO)
            batch_size: Batch size for training
            aggregation: Aggregation method ('mean' or 'median')
            fix_backbone_after: Fix shared backbone after this many updates
        """
        assert committee_size % 2 == 1, "Committee size should be odd for tie-breaking"
        
        self.committee_size = committee_size
        self.state_size = state_size
        self.batch_size = batch_size
        self.aggregation = aggregation
        self.fix_backbone_after = fix_backbone_after
        self.update_count = 0
        
        # Shared backbone (will be fixed after initial training)
        self.shared_backbone = SharedBackbone(input_size=state_size)
        self.backbone_optimizer = optim.Adam(self.shared_backbone.parameters(), lr=learning_rate)
        self.backbone_fixed = False
        
        # M personalized heads
        self.heads = nn.ModuleList([
            PersonalizedHead(input_size=self.shared_backbone.output_size)
            for _ in range(committee_size)
        ])
        
        # Optimizers for each head
        self.head_optimizers = [
            optim.Adam(head.parameters(), lr=learning_rate)
            for head in self.heads
        ]
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # Synthetic dataset: FIFO buffer of (state, reward) pairs
        self.synthetic_dataset = deque(maxlen=synthetic_dataset_capacity)
    
    def add_experience(self, state, reward):
        """
        Add (state, reward) pair to synthetic dataset.
        
        Args:
            state: Capability-aware state vector s_t^(c)
            reward: Ground-truth reward y_t^(c)
        """
        self.synthetic_dataset.append((state, reward))
    
    def predict_reward(self, state):
        """
        Predict reward using committee aggregation.
        
        Args:
            state: Capability-aware state vector
            
        Returns:
            aggregated_reward: Committee's aggregated reward prediction
            individual_predictions: List of individual member predictions
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get shared features
        with torch.no_grad():
            shared_features = self.shared_backbone(state_tensor)
        
        # Get predictions from each committee member
        individual_predictions = []
        for head in self.heads:
            with torch.no_grad():
                prediction = head(shared_features).item()
                individual_predictions.append(prediction)
        
        # Aggregate predictions
        if self.aggregation == 'mean':
            aggregated_reward = np.mean(individual_predictions)
        elif self.aggregation == 'median':
            aggregated_reward = np.median(individual_predictions)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
        
        return aggregated_reward, individual_predictions
    
    def train_committee(self, num_epochs=1):
        """
        Train committee on synthetic dataset.
        
        Args:
            num_epochs: Number of training epochs
            
        Returns:
            avg_loss: Average training loss
        """
        if len(self.synthetic_dataset) < self.batch_size:
            return 0.0  # Not enough data yet
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(num_epochs):
            # Sample mini-batch from synthetic dataset
            batch = random.sample(self.synthetic_dataset, self.batch_size)
            states = torch.FloatTensor([s for s, r in batch])
            targets = torch.FloatTensor([r for s, r in batch]).unsqueeze(1)
            
            # Forward pass through shared backbone
            if self.backbone_fixed:
                with torch.no_grad():
                    shared_features = self.shared_backbone(states)
            else:
                shared_features = self.shared_backbone(states)
            
            # Train each committee member
            committee_loss = 0.0
            
            for m in range(self.committee_size):
                # Forward pass through head
                predictions = self.heads[m](shared_features)
                
                # Compute loss
                loss = self.loss_fn(predictions, targets)
                committee_loss += loss
                
                # Backward pass and update head
                self.head_optimizers[m].zero_grad()
                loss.backward(retain_graph=(m < self.committee_size - 1))
                self.head_optimizers[m].step()
            
            # Update shared backbone (if not fixed)
            if not self.backbone_fixed:
                self.backbone_optimizer.zero_grad()
                committee_loss.backward()
                self.backbone_optimizer.step()
            
            total_loss += committee_loss.item()
            num_batches += 1
            
            # Check if we should fix the backbone
            self.update_count += 1
            if self.update_count >= self.fix_backbone_after and not self.backbone_fixed:
                self.backbone_fixed = True
                print(f"Shared backbone fixed after {self.update_count} updates")
        
        avg_loss = total_loss / (num_batches * self.committee_size) if num_batches > 0 else 0.0
        return avg_loss
    
    def get_committee_stats(self, state):
        """
        Get statistics about committee predictions for a given state.
        
        Args:
            state: Capability-aware state vector
            
        Returns:
            dict: Statistics including mean, std, min, max of predictions
        """
        _, individual_predictions = self.predict_reward(state)
        
        return {
            'mean': np.mean(individual_predictions),
            'median': np.median(individual_predictions),
            'std': np.std(individual_predictions),
            'min': np.min(individual_predictions),
            'max': np.max(individual_predictions),
            'predictions': individual_predictions
        }
    
    def save(self, filepath):
        """Save committee models."""
        torch.save({
            'shared_backbone': self.shared_backbone.state_dict(),
            'heads': [head.state_dict() for head in self.heads],
            'backbone_fixed': self.backbone_fixed,
            'update_count': self.update_count
        }, filepath)
    
    def load(self, filepath):
        """Load committee models."""
        checkpoint = torch.load(filepath)
        self.shared_backbone.load_state_dict(checkpoint['shared_backbone'])
        for i, head_state in enumerate(checkpoint['heads']):
            self.heads[i].load_state_dict(head_state)
        self.backbone_fixed = checkpoint['backbone_fixed']
        self.update_count = checkpoint['update_count']


if __name__ == "__main__":
    # Test the Reward Committee
    print("Testing Reward Committee...")
    
    # Initialize committee
    committee = RewardCommittee(committee_size=3, state_size=6)
    
    # Add some synthetic data
    for i in range(100):
        state = np.random.rand(6)
        reward = np.random.randn()  # Simulated reward
        committee.add_experience(state, reward)
    
    # Train committee
    loss = committee.train_committee(num_epochs=10)
    print(f"Training loss: {loss:.4f}")
    
    # Test prediction
    test_state = np.random.rand(6)
    pred_reward, individual_preds = committee.predict_reward(test_state)
    print(f"\nTest state: {test_state}")
    print(f"Individual predictions: {individual_preds}")
    print(f"Aggregated reward: {pred_reward:.4f}")
    
    # Get statistics
    stats = committee.get_committee_stats(test_state)
    print(f"\nCommittee statistics:")
    for key, value in stats.items():
        if key != 'predictions':
            print(f"  {key}: {value:.4f}")
    
    print("\n✅ Reward Committee test completed!")
