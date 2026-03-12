import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQN(nn.Module):
    """Neural Network for Q-learning with split point selection."""
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class CapabilityAwareDQN_Agent:
    """Capability-Aware DQN Agent for Split Point Selection (Algorithm 6)."""
    def __init__(self, state_size=6, action_size=None, total_layers=10,
                 learning_rate=0.001, gamma=0.95, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=0.995, memory_size=10000, batch_size=64, 
                 target_update_frequency=5, kappa=None):
        """
        Initialize the Capability-Aware DQN Agent (Algorithm 6).
        
        Args:
            state_size: Size of state vector (6 for capability-aware state)
            action_size: Number of possible split points
            total_layers: Total number of layers in the model
            learning_rate: Learning rate for optimizer (α)
            gamma: Discount factor (γ)
            epsilon_start: Initial exploration rate (ε_0)
            epsilon_end: Minimum exploration rate (ε_min)
            epsilon_decay: Legacy decay rate (deprecated, use kappa instead)
            memory_size: Size of experience replay buffer (N_max)
            batch_size: Batch size for training (B)
            target_update_frequency: Sync target network every C rounds
            kappa: Decay constant for exponential epsilon decay (κ)
        """
        if action_size is None:
            # Action space: split points from ceil(L/2) to L-1
            min_split = int(np.ceil(total_layers / 2))
            max_split = total_layers - 1
            action_size = max_split - min_split + 1
            # Ensure action_size is at least 1
            if action_size <= 0:
                action_size = 1
                min_split = total_layers - 1
            
        self.state_size = state_size
        self.action_size = int(action_size)  # Ensure it's an integer
        self.total_layers = total_layers
        self.min_split = int(np.ceil(total_layers / 2))
        self.max_split = total_layers - 1
        
        # DQN parameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_0 = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay  # Legacy (deprecated)
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        
        # Exponential epsilon decay: ε_t = ε_min + (ε_0 - ε_min)e^(-κt)
        if kappa is None:
            # Auto-calculate kappa for reasonable decay over ~50 rounds
            # Want epsilon to reach ~0.1 by round 30
            self.kappa = 0.05
        else:
            self.kappa = kappa
        
        # Neural networks
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        
        # Training metrics
        self.training_step = 0
        self.q_value_history = []
        self.loss_history = []
        self.epsilon_history = []
        
        # Update target network
        self.update_target_network()
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def get_action_from_index(self, action_idx):
        """Convert action index to actual split layer number."""
        return self.min_split + action_idx
    
    def select_split_point(self, cluster_state):
        """
        Select split point using epsilon-greedy strategy.
        
        Args:
            cluster_state: Capability-aware state vector
            
        Returns:
            split_layer: Selected split layer
        """
        if np.random.rand() < self.epsilon:
            # Random exploration
            return np.random.randint(self.min_split, self.max_split + 1)
        
        # Exploitation: choose best action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(cluster_state).unsqueeze(0)
            q_values = self.model(state_tensor)
            action_idx = q_values.argmax().item()
            return self.get_action_from_index(action_idx)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        # Convert split layer to action index
        action_idx = action - self.min_split
        self.memory.append((state, action_idx, reward, next_state, done))
    
    def replay(self):
        """Train the model using experience replay."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        # Get current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values from target network
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Track metrics
        self.training_step += 1
        self.loss_history.append(loss.item())
        self.q_value_history.append(current_q_values.mean().item())
        self.epsilon_history.append(self.epsilon)
    
    def decay_epsilon(self, round_num):
        """
        Exponential epsilon decay as per Algorithm 6.
        
        Formula: ε_t = ε_min + (ε_0 - ε_min)e^(-κt)
        
        Args:
            round_num: Current round number (t)
        """
        self.epsilon = self.epsilon_min + (self.epsilon_0 - self.epsilon_min) * np.exp(-self.kappa * round_num)
    
    def should_update_target(self, round_num):
        """
        Check if target network should be updated.
        
        Args:
            round_num: Current round number
            
        Returns:
            bool: True if round_num is multiple of target_update_frequency
        """
        return round_num % self.target_update_frequency == 0
    
    def get_metrics(self):
        """
        Get training metrics.
        
        Returns:
            dict: Dictionary containing training metrics
        """
        return {
            'q_values': self.q_value_history,
            'losses': self.loss_history,
            'epsilons': self.epsilon_history,
            'training_steps': self.training_step
        }

class DQN_Agent:
    """Legacy DQN Agent for Client Selection (backward compatibility)."""
    def __init__(self, state_size, action_size):
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()
        self.epsilon = 0.5

    def select_clients(self, state, num_clients, k=3):
        """Select clients using epsilon-greedy strategy."""
        if np.random.rand() < self.epsilon:
            return random.sample(range(num_clients), k)
        with torch.no_grad():
            q_values = self.model(torch.tensor(state, dtype=torch.float32))
            return q_values.argsort(descending=True)[:k].tolist()

    def train(self, state, action, reward, next_state):
        """Train the Q-network."""
        state, next_state = torch.tensor(state, dtype=torch.float32), torch.tensor(next_state, dtype=torch.float32)
        state = torch.tensor(state, dtype=torch.float32)  # Convert to float32
        action = torch.tensor(action, dtype=torch.long)   # Actions should be long for indexing
        reward = torch.tensor(reward, dtype=torch.float32) # Convert reward to float32
        next_state = torch.tensor(next_state, dtype=torch.float32)
        target_q = reward + 0.9 * self.target_model(next_state).max().item()
        q_value = self.model(state)[action]
        
        loss = self.loss_fn(q_value, torch.tensor(target_q))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
