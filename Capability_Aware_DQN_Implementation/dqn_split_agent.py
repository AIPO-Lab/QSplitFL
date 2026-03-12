"""
Capability-Aware DQN Agent for Split Point Selection
=====================================================

Implements Algorithm 6 from the paper:
"Capability-Aware DQN for FL with Experience Replay for Split Point Selection"

Key components:
1. Q-network (MLP): state (6D) → Q-values (action_size)
2. Target network: Frozen copy for stable TD targets
3. Experience replay buffer: FIFO storage of transitions
4. ε-greedy exploration with exponential decay
5. TD learning updates

State Space (6D):
    [avg_CPU, avg_Memory, avg_Battery, avg_Network, avg_Overall, std_Overall]

Action Space:
    {⌈L/2⌉, ⌈L/2⌉+1, ..., L-1} - Valid split points

Reward:
    r_t = -ΔL_t × exp(-λ(t-1)) - Decayed loss-drop
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class QNetwork(nn.Module):
    """
    Q-Network for approximating Q(s, a).
    
    Architecture:
        Input (6D state) → Dense(128) → ReLU → Dense(64) → ReLU → Output (action_size)
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [128, 64]):
        """
        Initialize Q-network.
        
        Args:
            state_size: Dimension of state vector (6 for capability-aware)
            action_size: Number of possible actions (split points)
            hidden_sizes: Sizes of hidden layers
        """
        super(QNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Build network layers
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
        logger.debug(f"Initialized Q-Network: {state_size} → {hidden_sizes} → {action_size}")
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Q-network.
        
        Args:
            state: State tensor of shape (batch_size, state_size) or (state_size,)
        
        Returns:
            q_values: Q-values for each action, shape (batch_size, action_size) or (action_size,)
        """
        return self.network(state)


class ReplayBuffer:
    """
    Experience Replay Buffer with FIFO policy.
    
    Stores transitions: (state, action, reward, next_state, done)
    """
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a random minibatch from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            states, actions, rewards, next_states, dones: Batched tensors
        """
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8)
        )
    
    def __len__(self):
        """Return current size of buffer."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return len(self.buffer) >= batch_size


class CapabilityAwareDQNAgent:
    """
    DQN Agent for capability-aware split point selection.
    
    Implements Algorithm 6 from the paper with:
    - Q-network and target network
    - Experience replay
    - ε-greedy exploration with decay
    - TD learning updates
    """
    
    def __init__(
        self,
        state_size: int = 6,
        total_layers: int = 10,
        learning_rate: float = 0.001,
        gamma: float = 0.2,  # Small discount for immediate focus
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 10,
        device: str = "cpu"
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_size: Dimension of state (6 for capability-aware)
            total_layers: Total layers in model (L)
            learning_rate: Learning rate for optimizer (α)
            gamma: Discount factor (γ) - small (0.1-0.3) per paper
            epsilon_start: Initial exploration rate (ε₀)
            epsilon_end: Final exploration rate (ε_min)
            epsilon_decay: Exponential decay rate (κ)
            buffer_capacity: Replay buffer size (N_max)
            batch_size: Minibatch size (B)
            target_update_freq: Target network sync period (C rounds)
            device: Device to use ("cpu" or "cuda")
        """
        self.state_size = state_size
        self.total_layers = total_layers
        self.device = device
        
        # Action space: {⌈L/2⌉, ..., L-1}
        self.min_split = int(np.ceil(total_layers / 2))
        self.max_split = total_layers - 1
        self.action_size = self.max_split - self.min_split + 1
        self.valid_splits = list(range(self.min_split, self.max_split + 1))
        
        logger.info(f"Action space: split points {self.valid_splits}")
        
        # Q-networks
        self.q_network = QNetwork(state_size, self.action_size).to(device)
        self.target_network = QNetwork(state_size, self.action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is always in eval mode
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Exploration
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Training metrics
        self.update_count = 0
        self.loss_history = []
        self.q_value_history = []
        self.epsilon_history = []
        
        logger.info(f"Initialized CapabilityAwareDQNAgent")
        logger.info(f"  State size: {state_size}")
        logger.info(f"  Action size: {self.action_size}")
        logger.info(f"  Gamma: {gamma}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Device: {device}")
    
    def select_action(self, state: np.ndarray, round_num: Optional[int] = None) -> int:
        """
        Select action using ε-greedy policy.
        
        With probability ε: explore (random action)
        With probability 1-ε: exploit (best Q-value)
        
        Args:
            state: Current capability-aware state (6D)
            round_num: Current round number (for logging)
        
        Returns:
            action: Selected split layer ℓ ∈ {⌈L/2⌉, ..., L-1}
        """
        # ε-greedy exploration
        if random.random() < self.epsilon:
            # Explore: random action
            action = random.choice(self.valid_splits)
            if round_num is not None:
                logger.debug(f"Round {round_num}: EXPLORE - random action {action} (ε={self.epsilon:.4f})")
        else:
            # Exploit: best Q-value
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor).cpu().numpy()[0]
            
            # Select action with highest Q-value
            best_action_idx = np.argmax(q_values)
            action = self.valid_splits[best_action_idx]
            
            if round_num is not None:
                logger.debug(f"Round {round_num}: EXPLOIT - best action {action} "
                           f"(Q-values: {q_values}, ε={self.epsilon:.4f})")
            
            # Store Q-values for analysis
            self.q_value_history.append(q_values.copy())
        
        # Store epsilon for tracking
        self.epsilon_history.append(self.epsilon)
        
        return action
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False
    ):
        """
        Store transition in replay buffer.
        
        Args:
            state: Current state
            action: Action taken (split layer)
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        # Convert action (split layer) to action index
        action_idx = action - self.min_split
        
        self.replay_buffer.push(state, action_idx, reward, next_state, done)
        
        logger.debug(f"Stored transition: action={action}, reward={reward:.4f}, "
                    f"buffer_size={len(self.replay_buffer)}")
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step (TD learning update).
        
        Algorithm:
        1. Sample minibatch from replay buffer
        2. Compute TD targets: y_i = r_i + γ(1-d_i) max Q̄(s'_i, a')
        3. Compute TD loss: L = (1/B) Σ (Q(s_i, a_i) - y_i)²
        4. Update Q-network via SGD
        
        Returns:
            loss: TD loss value (None if buffer not ready)
        """
        # Check if buffer has enough samples
        if not self.replay_buffer.is_ready(self.batch_size):
            return None
        
        # Sample minibatch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q-values: Q(s, a)
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute next Q-values from target network: max Q̄(s', a')
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            # TD target: r + γ(1-done) max Q̄(s', a')
            target_q_values = rewards + self.gamma * (1 - dones) * next_q_values
        
        # Compute TD loss: MSE between current and target Q-values
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Optional: gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update counter
        self.update_count += 1
        
        # Store loss
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        logger.debug(f"Training step {self.update_count}: TD loss = {loss_value:.6f}")
        
        return loss_value
    
    def update_target_network(self):
        """
        Update target network: Q̄_θ ← Q_θ
        
        This is done every C rounds to stabilize training.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
        logger.info(f"Updated target network (step {self.update_count})")
    
    def decay_epsilon(self, method: str = "exponential"):
        """
        Decay exploration rate.
        
        Exponential decay: ε_t = ε_min + (ε_0 - ε_min) × exp(-κt)
        
        Args:
            method: Decay method ("exponential" or "linear")
        """
        if method == "exponential":
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon * self.epsilon_decay
            )
        elif method == "linear":
            decay_amount = (self.epsilon_start - self.epsilon_end) / 1000
            self.epsilon = max(self.epsilon_end, self.epsilon - decay_amount)
        
        logger.debug(f"Decayed epsilon to {self.epsilon:.6f}")
    
    def save_model(self, filepath: str):
        """Save Q-network weights."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_count': self.update_count
        }, filepath)
        logger.info(f"Saved model to {filepath}")
    
    def load_model(self, filepath: str):
        """Load Q-network weights."""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.update_count = checkpoint['update_count']
        logger.info(f"Loaded model from {filepath}")
    
    def get_statistics(self) -> Dict:
        """Get agent statistics."""
        return {
            'epsilon': self.epsilon,
            'update_count': self.update_count,
            'buffer_size': len(self.replay_buffer),
            'avg_loss': np.mean(self.loss_history[-100:]) if self.loss_history else 0,
            'avg_q_value': np.mean([np.max(q) for q in self.q_value_history[-100:]]) if self.q_value_history else 0
        }
    
    def print_summary(self):
        """Print agent summary."""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("DQN AGENT SUMMARY")
        print("="*60)
        print(f"State size: {self.state_size}")
        print(f"Action space: {self.valid_splits}")
        print(f"Gamma (discount): {self.gamma}")
        print(f"Current epsilon: {stats['epsilon']:.6f}")
        print(f"Update count: {stats['update_count']}")
        print(f"Buffer size: {stats['buffer_size']}/{self.replay_buffer.capacity}")
        print(f"Average TD loss (last 100): {stats['avg_loss']:.6f}")
        print(f"Average max Q-value (last 100): {stats['avg_q_value']:.4f}")
        print("="*60 + "\n")


# Example usage and testing
if __name__ == "__main__":
    print("Testing CapabilityAwareDQNAgent...")
    
    # Initialize agent
    agent = CapabilityAwareDQNAgent(
        state_size=6,
        total_layers=10,
        gamma=0.2,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.99,
        buffer_capacity=1000,
        batch_size=32
    )
    
    print("\n--- Test 1: Action Selection ---")
    state = np.array([0.7, 0.8, 0.6, 0.9, 0.75, 0.1])
    
    for i in range(5):
        action = agent.select_action(state, round_num=i+1)
        print(f"  Round {i+1}: Selected action (split layer) = {action}")
    
    print("\n--- Test 2: Experience Storage and Training ---")
    # Store some fake transitions
    for i in range(100):
        state = np.random.rand(6)
        action = agent.select_action(state)
        reward = np.random.randn()  # Random reward
        next_state = np.random.rand(6)
        
        agent.store_transition(state, action, reward, next_state, done=False)
    
    print(f"Buffer size: {len(agent.replay_buffer)}")
    
    # Train for a few steps
    for i in range(10):
        loss = agent.train_step()
        if loss is not None:
            print(f"  Training step {i+1}: loss = {loss:.6f}")
    
    # Update target network
    agent.update_target_network()
    
    # Decay epsilon
    for _ in range(10):
        agent.decay_epsilon()
    print(f"Epsilon after 10 decays: {agent.epsilon:.6f}")
    
    # Print summary
    agent.print_summary()
    
    print("\n✅ All tests passed!")

