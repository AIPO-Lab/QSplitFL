"""
Committee-Based DQN for Robust Action Selection (Algorithm 9)

This module implements an ensemble of M DQNs with shared encoder and separate heads
for robust split point selection with majority voting.

Architecture:
- Shared encoder f_s(·; φ_s): Shallow layers (fixed after initial training)
- M DQN heads g^(m)(·; ψ^(m)): Separate deep heads for diversity
- Q-network: Q^(m)(s,a) = [g^(m)(f_s(s; φ_s); ψ^(m))]_a

Reference: Algorithm 9 in the LaTeX document
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from voting_mechanism import majority_vote


class SharedEncoder(nn.Module):
    """Shared shallow encoder for all committee members."""
    
    def __init__(self, input_size=6, hidden_sizes=[32, 32]):
        """
        Initialize shared encoder.
        
        Args:
            input_size: Size of input state vector
            hidden_sizes: List of hidden layer sizes
        """
        super(SharedEncoder, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        self.network = nn.Sequential(*layers)
        self.output_size = hidden_sizes[-1]
    
    def forward(self, x):
        """Forward pass through shared encoder."""
        return self.network(x)


class DQNHead(nn.Module):
    """DQN head for each committee member."""
    
    def __init__(self, input_size=32, hidden_sizes=[64, 32], action_size=5):
        """
        Initialize DQN head.
        
        Args:
            input_size: Size of input from shared encoder
            hidden_sizes: List of hidden layer sizes
            action_size: Number of actions (split points)
        """
        super(DQNHead, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer: Q-values for each action
        layers.append(nn.Linear(prev_size, action_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through DQN head."""
        return self.network(x)


class CommitteeDQN:
    """
    Committee of M DQNs for robust action selection (Algorithm 9).
    
    Each member consists of:
    - Shared encoder (shallow, fixed)
    - DQN head (deep, personalized)
    
    Action selection uses majority voting with tie-breaking.
    """
    
    def __init__(self, committee_size=3, state_size=6, action_size=5, total_layers=10,
                 learning_rate=0.001, gamma=0.95, epsilon_start=1.0, epsilon_end=0.01,
                 kappa=0.05, synthetic_dataset_capacity=10000, batch_size=32,
                 target_update_frequency=5, aggregation='voting', fix_encoder_after=100):
        """
        Initialize Committee DQN.
        
        Args:
            committee_size: Number of committee members M (odd recommended)
            state_size: Size of state vector
            action_size: Number of actions
            total_layers: Total layers in model (for action space)
            learning_rate: Learning rate (η)
            gamma: Discount factor (γ)
            epsilon_start: Initial exploration (ε_0)
            epsilon_end: Minimum exploration (ε_min)
            kappa: Decay constant (κ)
            synthetic_dataset_capacity: Max synthetic dataset size
            batch_size: Mini-batch size (B)
            target_update_frequency: Sync target every C rounds
            aggregation: Aggregation method ('voting', 'mean', 'median')
            fix_encoder_after: Fix shared encoder after this many updates
        """
        assert committee_size % 2 == 1, "Committee size should be odd"
        
        self.committee_size = committee_size
        self.state_size = state_size
        self.action_size = action_size
        self.total_layers = total_layers
        self.min_split = int(np.ceil(total_layers / 2))
        self.max_split = total_layers - 1
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_0 = epsilon_start
        self.epsilon_min = epsilon_end
        self.kappa = kappa
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.aggregation = aggregation
        self.fix_encoder_after = fix_encoder_after
        self.update_count = 0
        self.encoder_fixed = False
        
        # Shared encoder (will be fixed after initial training)
        self.shared_encoder = SharedEncoder(input_size=state_size)
        self.encoder_optimizer = optim.Adam(self.shared_encoder.parameters(), lr=learning_rate)
        
        # M DQN heads (online networks)
        self.dqn_heads = nn.ModuleList([
            DQNHead(input_size=self.shared_encoder.output_size, action_size=action_size)
            for _ in range(committee_size)
        ])
        
        # M DQN heads (target networks)
        self.target_heads = nn.ModuleList([
            DQNHead(input_size=self.shared_encoder.output_size, action_size=action_size)
            for _ in range(committee_size)
        ])
        
        # Initialize target networks
        for m in range(committee_size):
            self.target_heads[m].load_state_dict(self.dqn_heads[m].state_dict())
        
        # Optimizers for each head
        self.head_optimizers = [
            optim.Adam(self.dqn_heads[m].parameters(), lr=learning_rate)
            for m in range(committee_size)
        ]
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # Synthetic dataset: FIFO buffer of (s, a, r, s', d) transitions
        self.synthetic_dataset = deque(maxlen=synthetic_dataset_capacity)
        
        # Training metrics
        self.training_steps = 0
        self.vote_history = []
    
    def get_q_value(self, state, action):
        """
        Get Q-value for a specific state-action pair (for tie-breaking).
        
        Args:
            state: State vector
            action: Action (split layer)
            
        Returns:
            q_value: Q-value from first committee member
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_idx = action - self.min_split
        
        with torch.no_grad():
            shared_features = self.shared_encoder(state_tensor)
            q_values = self.dqn_heads[0](shared_features)
            return q_values[0, action_idx].item()
    
    def select_action(self, state, use_epsilon_greedy=True):
        """
        Select action using committee with ε-greedy and majority voting.
        
        Args:
            state: Capability-aware state vector
            use_epsilon_greedy: Whether to use ε-greedy exploration
            
        Returns:
            action: Selected split layer
            vote_info: Dictionary with voting information
        """
        # ε-greedy exploration
        if use_epsilon_greedy and np.random.rand() < self.epsilon:
            # Random exploration
            action = np.random.randint(self.min_split, self.max_split + 1)
            return action, {'method': 'random', 'epsilon': self.epsilon}
        
        # Committee-based action selection
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get shared features
        with torch.no_grad():
            if self.encoder_fixed:
                shared_features = self.shared_encoder(state_tensor)
            else:
                shared_features = self.shared_encoder(state_tensor)
        
        # Get proposals from each committee member
        proposals = []
        for m in range(self.committee_size):
            with torch.no_grad():
                q_values = self.dqn_heads[m](shared_features)
                action_idx = q_values.argmax().item()
                action = self.min_split + action_idx
                proposals.append(action)
        
        # Majority voting
        action_space = list(range(self.min_split, self.max_split + 1))
        
        # Create mock DQN objects for voting (with get_q_value method)
        class MockDQN:
            def __init__(self, head, encoder, min_split):
                self.head = head
                self.encoder = encoder
                self.min_split = min_split
            
            def get_q_value(self, state, action):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_idx = action - self.min_split
                with torch.no_grad():
                    features = self.encoder(state_tensor)
                    q_values = self.head(features)
                    return q_values[0, action_idx].item()
        
        mock_dqns = [
            MockDQN(self.dqn_heads[m], self.shared_encoder, self.min_split)
            for m in range(self.committee_size)
        ]
        
        selected_action, vote_counts, is_tie = majority_vote(
            proposals, state, mock_dqns, action_space
        )
        
        vote_info = {
            'method': 'voting',
            'proposals': proposals,
            'vote_counts': vote_counts,
            'is_tie': is_tie,
            'selected_action': selected_action
        }
        
        self.vote_history.append(vote_info)
        
        return selected_action, vote_info
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store transition in synthetic dataset.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Terminal flag
        """
        self.synthetic_dataset.append((state, action, reward, next_state, done))
    
    def train_committee(self):
        """
        Train committee on synthetic dataset (Algorithm 9).
        
        Returns:
            avg_loss: Average training loss across all members
        """
        if len(self.synthetic_dataset) < self.batch_size:
            return 0.0
        
        # Sample mini-batch
        batch = random.sample(self.synthetic_dataset, self.batch_size)
        states = torch.FloatTensor([s for s, a, r, ns, d in batch])
        actions = torch.LongTensor([a - self.min_split for s, a, r, ns, d in batch])
        rewards = torch.FloatTensor([r for s, a, r, ns, d in batch])
        next_states = torch.FloatTensor([ns for s, a, r, ns, d in batch])
        dones = torch.BoolTensor([d for s, a, r, ns, d in batch])
        
        # Get shared features
        if self.encoder_fixed:
            with torch.no_grad():
                shared_features = self.shared_encoder(states)
                next_shared_features = self.shared_encoder(next_states)
        else:
            shared_features = self.shared_encoder(states)
            next_shared_features = self.shared_encoder(next_states)
        
        total_loss = 0.0
        
        # Train each committee member independently
        for m in range(self.committee_size):
            # Compute TD targets
            with torch.no_grad():
                next_q_values = self.target_heads[m](next_shared_features)
                max_next_q = next_q_values.max(1)[0]
                td_targets = rewards + self.gamma * (1 - dones.float()) * max_next_q
            
            # Current Q-values
            current_q_values = self.dqn_heads[m](shared_features).gather(1, actions.unsqueeze(1)).squeeze()
            
            # Compute loss
            loss = self.loss_fn(current_q_values, td_targets)
            total_loss += loss.item()
            
            # Update head
            self.head_optimizers[m].zero_grad()
            loss.backward(retain_graph=(m < self.committee_size - 1 and not self.encoder_fixed))
            self.head_optimizers[m].step()
        
        # Update shared encoder (if not fixed)
        if not self.encoder_fixed:
            # Recompute loss for encoder update
            encoder_loss = 0.0
            shared_features = self.shared_encoder(states)
            
            for m in range(self.committee_size):
                with torch.no_grad():
                    next_shared_features = self.shared_encoder(next_states)
                    next_q_values = self.target_heads[m](next_shared_features)
                    max_next_q = next_q_values.max(1)[0]
                    td_targets = rewards + self.gamma * (1 - dones.float()) * max_next_q
                
                current_q_values = self.dqn_heads[m](shared_features).gather(1, actions.unsqueeze(1)).squeeze()
                encoder_loss += self.loss_fn(current_q_values, td_targets)
            
            self.encoder_optimizer.zero_grad()
            encoder_loss.backward()
            self.encoder_optimizer.step()
        
        # Check if we should fix the encoder
        self.update_count += 1
        if self.update_count >= self.fix_encoder_after and not self.encoder_fixed:
            self.encoder_fixed = True
            print(f"Shared encoder fixed after {self.update_count} updates")
        
        self.training_steps += 1
        
        return total_loss / self.committee_size
    
    def update_target_networks(self):
        """Update all target networks."""
        for m in range(self.committee_size):
            self.target_heads[m].load_state_dict(self.dqn_heads[m].state_dict())
    
    def decay_epsilon(self, round_num):
        """Exponential epsilon decay."""
        self.epsilon = self.epsilon_min + (self.epsilon_0 - self.epsilon_min) * np.exp(-self.kappa * round_num)
    
    def should_update_target(self, round_num):
        """Check if target networks should be updated."""
        return round_num % self.target_update_frequency == 0
    
    def save(self, filepath):
        """Save committee models."""
        torch.save({
            'shared_encoder': self.shared_encoder.state_dict(),
            'dqn_heads': [head.state_dict() for head in self.dqn_heads],
            'target_heads': [head.state_dict() for head in self.target_heads],
            'encoder_fixed': self.encoder_fixed,
            'update_count': self.update_count,
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath):
        """Load committee models."""
        checkpoint = torch.load(filepath)
        self.shared_encoder.load_state_dict(checkpoint['shared_encoder'])
        for m in range(self.committee_size):
            self.dqn_heads[m].load_state_dict(checkpoint['dqn_heads'][m])
            self.target_heads[m].load_state_dict(checkpoint['target_heads'][m])
        self.encoder_fixed = checkpoint['encoder_fixed']
        self.update_count = checkpoint['update_count']
        self.epsilon = checkpoint['epsilon']


if __name__ == "__main__":
    # Test Committee DQN
    print("Testing Committee DQN...")
    
    # Initialize committee
    committee = CommitteeDQN(
        committee_size=3,
        state_size=6,
        action_size=5,
        total_layers=10
    )
    
    # Test action selection
    test_state = np.random.rand(6)
    action, vote_info = committee.select_action(test_state, use_epsilon_greedy=False)
    
    print(f"\nTest state: {test_state}")
    print(f"Selected action: {action}")
    print(f"Vote info: {vote_info}")
    
    # Add some transitions
    for i in range(100):
        state = np.random.rand(6)
        action = np.random.randint(5, 10)
        reward = np.random.randn()
        next_state = np.random.rand(6)
        done = False
        committee.store_transition(state, action, reward, next_state, done)
    
    # Train committee
    loss = committee.train_committee()
    print(f"\nTraining loss: {loss:.4f}")
    
    # Update target networks
    committee.update_target_networks()
    print("Target networks updated")
    
    print("\nCommittee DQN test completed!")
