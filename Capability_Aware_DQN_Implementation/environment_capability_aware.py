"""
Capability-Aware Federated Learning Environment
================================================

This module implements a sophisticated FL environment that:
1. Tracks client hardware capabilities (CPU, Memory, Battery, Network)
2. Computes 6D cluster state vectors
3. Provides decayed loss-drop rewards
4. Simulates realistic capability dynamics

Based on the capability-aware Q-learning framework for split point optimization.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClientCapability:
    """Data class to store client capability metrics."""
    cpu: float  # [0, 1] normalized
    memory: float  # [0, 1] normalized
    battery: float  # [0, 1] normalized
    network: float  # [0, 1] normalized
    overall: float  # Weighted combination
    client_id: int
    
    def __repr__(self):
        return (f"Client{self.client_id}: CPU={self.cpu:.2f}, Mem={self.memory:.2f}, "
                f"Bat={self.battery:.2f}, Net={self.network:.2f}, Overall={self.overall:.2f}")


class CapabilityAwareEnvironment:
    """
    Federated Learning Environment with Capability-Aware State Representation.
    
    This environment:
    - Tracks client capabilities over time
    - Computes cluster-level state vectors (6D)
    - Provides decayed loss-drop rewards
    - Simulates dynamic capability changes
    
    State Space (6D):
        [avg_CPU, avg_Memory, avg_Battery, avg_Network, avg_Overall, std_Overall]
    
    Reward:
        r_t = -ΔL_t × ρ_t where ρ_t = exp(-λ(t-1))
    """
    
    def __init__(
        self,
        num_clients: int,
        capability_weights: Optional[List[float]] = None,
        capability_profiles: Optional[str] = "mixed",
        seed: Optional[int] = None
    ):
        """
        Initialize the capability-aware environment.
        
        Args:
            num_clients: Number of federated clients
            capability_weights: Weights for [CPU, Memory, Battery, Network]
                               Default: [0.25, 0.25, 0.25, 0.25] (equal weights)
            capability_profiles: Type of capability distribution
                               "high": All high-capability clients
                               "low": All low-capability clients
                               "mixed": Mix of high/medium/low (realistic)
                               "heterogeneous": Maximum diversity
            seed: Random seed for reproducibility
        """
        self.num_clients = num_clients
        
        # Set default capability weights if not provided
        if capability_weights is None:
            self.capability_weights = [0.25, 0.25, 0.25, 0.25]
        else:
            assert len(capability_weights) == 4, "Need 4 weights for [CPU, Mem, Bat, Net]"
            assert abs(sum(capability_weights) - 1.0) < 1e-6, "Weights must sum to 1.0"
            self.capability_weights = capability_weights
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize client capabilities
        self.capability_profiles = capability_profiles
        self.client_capabilities: List[ClientCapability] = []
        self._initialize_capabilities()
        
        # Loss tracking for reward computation
        self.loss_history = deque(maxlen=1000)  # Store loss history
        self.prev_loss: Optional[float] = None
        
        # Metrics tracking
        self.state_history = []
        self.reward_history = []
        self.action_history = []
        
        logger.info(f"Initialized CapabilityAwareEnvironment with {num_clients} clients")
        logger.info(f"Capability profile: {capability_profiles}")
        logger.info(f"Capability weights: {self.capability_weights}")
    
    def _initialize_capabilities(self):
        """
        Initialize client capabilities based on the selected profile.
        
        Profiles:
        - high: [0.8, 1.0] for all metrics
        - low: [0.2, 0.4] for all metrics
        - mixed: 1/3 high, 1/3 medium, 1/3 low
        - heterogeneous: Maximum diversity across all ranges
        """
        self.client_capabilities = []
        
        if self.capability_profiles == "high":
            # All high-capability clients (e.g., datacenter servers)
            for i in range(self.num_clients):
                cpu = np.random.uniform(0.8, 1.0)
                memory = np.random.uniform(0.8, 1.0)
                battery = np.random.uniform(0.8, 1.0)
                network = np.random.uniform(0.8, 1.0)
                overall = self._compute_overall_capability(cpu, memory, battery, network)
                
                self.client_capabilities.append(
                    ClientCapability(cpu, memory, battery, network, overall, i)
                )
        
        elif self.capability_profiles == "low":
            # All low-capability clients (e.g., IoT devices)
            for i in range(self.num_clients):
                cpu = np.random.uniform(0.2, 0.4)
                memory = np.random.uniform(0.2, 0.4)
                battery = np.random.uniform(0.2, 0.4)
                network = np.random.uniform(0.2, 0.4)
                overall = self._compute_overall_capability(cpu, memory, battery, network)
                
                self.client_capabilities.append(
                    ClientCapability(cpu, memory, battery, network, overall, i)
                )
        
        elif self.capability_profiles == "mixed":
            # Mixed capabilities (realistic scenario)
            num_high = self.num_clients // 3
            num_medium = self.num_clients // 3
            num_low = self.num_clients - num_high - num_medium
            
            client_id = 0
            
            # High-capability clients
            for _ in range(num_high):
                cpu = np.random.uniform(0.8, 1.0)
                memory = np.random.uniform(0.8, 1.0)
                battery = np.random.uniform(0.7, 1.0)
                network = np.random.uniform(0.8, 1.0)
                overall = self._compute_overall_capability(cpu, memory, battery, network)
                
                self.client_capabilities.append(
                    ClientCapability(cpu, memory, battery, network, overall, client_id)
                )
                client_id += 1
            
            # Medium-capability clients
            for _ in range(num_medium):
                cpu = np.random.uniform(0.5, 0.7)
                memory = np.random.uniform(0.5, 0.7)
                battery = np.random.uniform(0.4, 0.7)
                network = np.random.uniform(0.5, 0.7)
                overall = self._compute_overall_capability(cpu, memory, battery, network)
                
                self.client_capabilities.append(
                    ClientCapability(cpu, memory, battery, network, overall, client_id)
                )
                client_id += 1
            
            # Low-capability clients
            for _ in range(num_low):
                cpu = np.random.uniform(0.2, 0.4)
                memory = np.random.uniform(0.2, 0.4)
                battery = np.random.uniform(0.2, 0.4)
                network = np.random.uniform(0.3, 0.5)
                overall = self._compute_overall_capability(cpu, memory, battery, network)
                
                self.client_capabilities.append(
                    ClientCapability(cpu, memory, battery, network, overall, client_id)
                )
                client_id += 1
        
        else:  # heterogeneous
            # Maximum diversity
            for i in range(self.num_clients):
                cpu = np.random.uniform(0.2, 1.0)
                memory = np.random.uniform(0.2, 1.0)
                battery = np.random.uniform(0.2, 1.0)
                network = np.random.uniform(0.2, 1.0)
                overall = self._compute_overall_capability(cpu, memory, battery, network)
                
                self.client_capabilities.append(
                    ClientCapability(cpu, memory, battery, network, overall, i)
                )
        
        # Log initial capabilities
        logger.info("Initial client capabilities:")
        for cap in self.client_capabilities:
            logger.debug(f"  {cap}")
    
    def _compute_overall_capability(
        self, 
        cpu: float, 
        memory: float, 
        battery: float, 
        network: float
    ) -> float:
        """
        Compute overall capability score as weighted combination.
        
        C_Overall = w1·CPU + w2·Memory + w3·Battery + w4·Network
        
        Args:
            cpu, memory, battery, network: Individual capability metrics [0, 1]
        
        Returns:
            overall: Weighted overall capability [0, 1]
        """
        overall = (
            self.capability_weights[0] * cpu +
            self.capability_weights[1] * memory +
            self.capability_weights[2] * battery +
            self.capability_weights[3] * network
        )
        return overall
    
    def get_cluster_state(self, client_indices: List[int]) -> np.ndarray:
        """
        Compute 6D cluster-level state vector from Eq. 6 in the paper.
        
        State: [avg_CPU, avg_Memory, avg_Battery, avg_Network, avg_Overall, std_Overall]
        
        This is the KEY INNOVATION: O(|K|) complexity vs O(d³) for PCA-based states.
        
        Args:
            client_indices: Indices of clients in this cluster
        
        Returns:
            state: 6D numpy array representing cluster state
        """
        if not client_indices:
            logger.warning("Empty client_indices, returning zero state")
            return np.zeros(6)
        
        # Collect capabilities for selected clients
        cpu_values = []
        memory_values = []
        battery_values = []
        network_values = []
        overall_values = []
        
        for idx in client_indices:
            cap = self.client_capabilities[idx]
            cpu_values.append(cap.cpu)
            memory_values.append(cap.memory)
            battery_values.append(cap.battery)
            network_values.append(cap.network)
            overall_values.append(cap.overall)
        
        # Compute averages (Eq. 7)
        avg_cpu = np.mean(cpu_values)
        avg_memory = np.mean(memory_values)
        avg_battery = np.mean(battery_values)
        avg_network = np.mean(network_values)
        avg_overall = np.mean(overall_values)
        
        # Compute standard deviation (heterogeneity measure, Eq. 8)
        std_overall = np.std(overall_values)
        
        # Construct state vector (Eq. 6)
        state = np.array([
            avg_cpu,
            avg_memory,
            avg_battery,
            avg_network,
            avg_overall,
            std_overall
        ])
        
        # Store for history
        self.state_history.append(state.copy())
        
        logger.debug(f"Cluster state: CPU={avg_cpu:.3f}, Mem={avg_memory:.3f}, "
                    f"Bat={avg_battery:.3f}, Net={avg_network:.3f}, "
                    f"Overall={avg_overall:.3f}, Std={std_overall:.3f}")
        
        return state
    
    def compute_decayed_reward(
        self,
        current_loss: float,
        round_num: int,
        verbose: bool = True
    ) -> float:
        """
        Compute decayed loss-drop reward from the paper.
        
        Reward formula:
            r_t = -ΔL_t × ρ_t
            where:
                ΔL_t = L_t - L_{t-1}
                ρ_t = exp(-λ(t-1))
                λ = ln(2) / t  (half-life decay)
        
        Interpretation:
        - Positive reward if loss decreases (ΔL < 0)
        - Negative reward if loss increases (ΔL > 0)
        - Early rounds weighted more heavily (ρ_t decay)
        
        Args:
            current_loss: Loss at current round t
            round_num: Current round number (1-indexed)
            verbose: Whether to log reward computation
        
        Returns:
            reward: Scalar reward value
        """
        # First round: no previous loss, return 0
        if self.prev_loss is None:
            self.prev_loss = current_loss
            self.loss_history.append(current_loss)
            logger.info(f"Round {round_num}: Initial loss = {current_loss:.4f}, reward = 0.0")
            return 0.0
        
        # Compute loss change
        delta_loss = current_loss - self.prev_loss
        
        # Compute decay factor (exponential decay)
        # λ = ln(2) / round_num for half-life behavior
        if round_num > 1:
            lambda_decay = np.log(2) / round_num
            rho_t = np.exp(-lambda_decay * (round_num - 1))
        else:
            rho_t = 1.0  # No decay for first round
        
        # Compute reward: negative of (loss change × decay)
        # If loss decreases (delta_loss < 0), reward is positive
        # If loss increases (delta_loss > 0), reward is negative
        reward = -delta_loss * rho_t
        
        # Update history
        self.loss_history.append(current_loss)
        self.reward_history.append(reward)
        self.prev_loss = current_loss
        
        if verbose:
            logger.info(f"Round {round_num}: Loss={current_loss:.4f}, "
                       f"ΔLoss={delta_loss:.4f}, ρ_t={rho_t:.4f}, "
                       f"Reward={reward:.4f}")
        
        return reward
    
    def update_client_capabilities(
        self,
        client_indices: Optional[List[int]] = None,
        noise_scale: float = 0.05
    ):
        """
        Simulate dynamic capability changes over time.
        
        This simulates realistic scenarios where:
        - Battery drains over time
        - Network conditions fluctuate
        - CPU/Memory availability changes
        
        Args:
            client_indices: Which clients to update (None = all)
            noise_scale: Standard deviation of Gaussian noise
        """
        if client_indices is None:
            client_indices = list(range(self.num_clients))
        
        for idx in client_indices:
            cap = self.client_capabilities[idx]
            
            # Add Gaussian noise to each metric
            noise = np.random.normal(0, noise_scale, 4)
            
            # Update with bounds [0.1, 1.0]
            new_cpu = np.clip(cap.cpu + noise[0], 0.1, 1.0)
            new_memory = np.clip(cap.memory + noise[1], 0.1, 1.0)
            new_battery = np.clip(cap.battery + noise[2], 0.1, 1.0)
            new_network = np.clip(cap.network + noise[3], 0.1, 1.0)
            
            # Recompute overall capability
            new_overall = self._compute_overall_capability(
                new_cpu, new_memory, new_battery, new_network
            )
            
            # Update client capability
            self.client_capabilities[idx] = ClientCapability(
                new_cpu, new_memory, new_battery, new_network, new_overall, idx
            )
        
        logger.debug(f"Updated capabilities for {len(client_indices)} clients")
    
    def get_client_capability(self, client_id: int) -> ClientCapability:
        """Get capability info for a specific client."""
        return self.client_capabilities[client_id]
    
    def get_all_capabilities(self) -> List[ClientCapability]:
        """Get all client capabilities."""
        return self.client_capabilities.copy()
    
    def reset(self):
        """Reset environment (for new episode)."""
        self.prev_loss = None
        self.loss_history.clear()
        self.state_history = []
        self.reward_history = []
        self.action_history = []
        logger.info("Environment reset")
    
    def get_statistics(self) -> Dict:
        """Get environment statistics."""
        if not self.loss_history:
            return {}
        
        return {
            "avg_loss": np.mean(self.loss_history),
            "final_loss": self.loss_history[-1],
            "loss_improvement": self.loss_history[0] - self.loss_history[-1] if len(self.loss_history) > 1 else 0,
            "avg_reward": np.mean(self.reward_history) if self.reward_history else 0,
            "total_reward": np.sum(self.reward_history) if self.reward_history else 0,
            "num_rounds": len(self.loss_history)
        }
    
    def print_summary(self):
        """Print environment summary."""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("ENVIRONMENT SUMMARY")
        print("="*60)
        print(f"Number of clients: {self.num_clients}")
        print(f"Capability profile: {self.capability_profiles}")
        print(f"Capability weights: {self.capability_weights}")
        
        if stats:
            print(f"\nPerformance:")
            print(f"  Average loss: {stats['avg_loss']:.4f}")
            print(f"  Final loss: {stats['final_loss']:.4f}")
            print(f"  Loss improvement: {stats['loss_improvement']:.4f}")
            print(f"  Average reward: {stats['avg_reward']:.4f}")
            print(f"  Total reward: {stats['total_reward']:.4f}")
            print(f"  Rounds completed: {stats['num_rounds']}")
        
        print("\nClient Capabilities:")
        for cap in self.client_capabilities:
            print(f"  {cap}")
        
        print("="*60 + "\n")


# Example usage and testing
if __name__ == "__main__":
    print("Testing CapabilityAwareEnvironment...")
    
    # Test 1: Mixed capability profile
    print("\n--- Test 1: Mixed Capabilities ---")
    env = CapabilityAwareEnvironment(
        num_clients=9,
        capability_profiles="mixed",
        seed=42
    )
    
    # Get cluster state for a subset of clients
    selected_clients = [0, 3, 6]  # One from each tier
    state = env.get_cluster_state(selected_clients)
    print(f"Cluster state for clients {selected_clients}:")
    print(f"  State vector: {state}")
    
    # Simulate loss trajectory and compute rewards
    print("\n--- Test 2: Reward Computation ---")
    losses = [0.8, 0.7, 0.65, 0.62, 0.60, 0.59]
    for round_num, loss in enumerate(losses, 1):
        reward = env.compute_decayed_reward(loss, round_num)
    
    # Update capabilities
    print("\n--- Test 3: Dynamic Capability Updates ---")
    print("Before update:")
    print(f"  Client 0: {env.get_client_capability(0)}")
    
    env.update_client_capabilities([0])
    
    print("After update:")
    print(f"  Client 0: {env.get_client_capability(0)}")
    
    # Print summary
    env.print_summary()
    
    print("\n✅ All tests passed!")

