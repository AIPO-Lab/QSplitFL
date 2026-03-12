import numpy as np
import torch.nn.functional as F
from scipy.stats import entropy

class FL_Environment:
    """Federated Learning Environment for Capability-Aware Q-Learning."""
    def __init__(self, num_clients, global_class_dist, alpha=0.3, beta=0.2, capability_weights=None):
        self.num_clients = num_clients
        self.global_class_dist = global_class_dist
        # Balancing factors for the reward function
        self.alpha = alpha  # KL divergence balancing factor
        self.beta = beta    # Participation frequency balancing factor
        
        # Capability weights for overall capability score
        if capability_weights is None:
            self.capability_weights = [0.25, 0.25, 0.25, 0.25]  # Equal weights for CPU, Memory, Battery, Network
        else:
            self.capability_weights = capability_weights
            
        # Initialize client capabilities (will be updated during training)
        self.client_capabilities = self._initialize_client_capabilities()
        
    def _initialize_client_capabilities(self):
        """Initialize random client capabilities for simulation."""
        capabilities = []
        for _ in range(self.num_clients):
            # Simulate different client capabilities
            cpu_cap = np.random.uniform(0.3, 1.0)  # CPU capability
            memory_cap = np.random.uniform(0.4, 1.0)  # Memory capability
            battery_cap = np.random.uniform(0.2, 1.0)  # Battery capability
            network_cap = np.random.uniform(0.3, 1.0)  # Network capability
            
            capabilities.append({
                'CPU': cpu_cap,
                'Memory': memory_cap,
                'Battery': battery_cap,
                'Network': network_cap
            })
        return capabilities
    
    def compute_overall_capability(self, client_idx):
        """Compute overall capability score for a client."""
        caps = self.client_capabilities[client_idx]
        overall = (
            self.capability_weights[0] * caps['CPU'] +
            self.capability_weights[1] * caps['Memory'] +
            self.capability_weights[2] * caps['Battery'] +
            self.capability_weights[3] * caps['Network']
        )
        return overall
    
    def get_cluster_state(self, client_indices):
        """
        Compute cluster-level state vector as per Equation 36 in the paper.
        State vector: [avg_CPU, avg_Memory, avg_Battery, avg_Network, avg_Overall, capability_variance]
        """
        if not client_indices:
            return np.zeros(6)
            
        # Collect capabilities for clients in the cluster
        cluster_caps = {
            'CPU': [], 'Memory': [], 'Battery': [], 'Network': [], 'Overall': []
        }
        
        for idx in client_indices:
            caps = self.client_capabilities[idx]
            cluster_caps['CPU'].append(caps['CPU'])
            cluster_caps['Memory'].append(caps['Memory'])
            cluster_caps['Battery'].append(caps['Battery'])
            cluster_caps['Network'].append(caps['Network'])
            cluster_caps['Overall'].append(self.compute_overall_capability(idx))
        
        # Compute averages
        avg_cpu = np.mean(cluster_caps['CPU'])
        avg_memory = np.mean(cluster_caps['Memory'])
        avg_battery = np.mean(cluster_caps['Battery'])
        avg_network = np.mean(cluster_caps['Network'])
        avg_overall = np.mean(cluster_caps['Overall'])
        
        # Compute capability variance (heterogeneity within cluster)
        capability_variance = np.std(cluster_caps['Overall'])
        
        # Return state vector
        return np.array([avg_cpu, avg_memory, avg_battery, avg_network, avg_overall, capability_variance])
    
    def get_state(self):
        """Return the global class distribution (for backward compatibility)."""
        return self.global_class_dist
    
    def update_client_capabilities(self, client_idx=None):
        """Simulate dynamic capability changes over time."""
        if client_idx is None:
            # Update all clients
            for i in range(self.num_clients):
                self._update_single_client_capability(i)
        else:
            # Update specific client
            self._update_single_client_capability(client_idx)
    
    def _update_single_client_capability(self, client_idx):
        """Update capability for a single client with small random changes."""
        caps = self.client_capabilities[client_idx]
        
        # Add small random changes to simulate dynamic conditions
        caps['CPU'] = np.clip(caps['CPU'] + np.random.uniform(-0.05, 0.05), 0.1, 1.0)
        caps['Memory'] = np.clip(caps['Memory'] + np.random.uniform(-0.05, 0.05), 0.1, 1.0)
        caps['Battery'] = np.clip(caps['Battery'] + np.random.uniform(-0.05, 0.05), 0.1, 1.0)
        caps['Network'] = np.clip(caps['Network'] + np.random.uniform(-0.05, 0.05), 0.1, 1.0)

    def compute_kl_divergence(self, client_dist, global_dist):
        """
        Compute KL divergence between client and global distributions.
        KL(P||Q) = sum(P * log(P/Q))
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        client_dist = np.array(client_dist) + epsilon
        global_dist = np.array(global_dist) + epsilon
        
        # Normalize distributions
        client_dist = client_dist / np.sum(client_dist)
        global_dist = global_dist / np.sum(global_dist)
        
        # Compute KL divergence
        kl_div = np.sum(client_dist * np.log(client_dist / global_dist))
        return kl_div

    def compute_capability_aware_reward(self, prev_acc, new_acc, client_indices, split_layer, total_layers):
        """
        Calculate capability-aware reward as per Algorithm 6.
        
        The reward function considers:
        1. Accuracy improvement
        2. Resource efficiency based on split layer and client capabilities
        3. Penalty for resource-intensive split points for low-capability clusters
        
        Args:
            prev_acc: Previous global accuracy
            new_acc: New global accuracy
            client_indices: Indices of selected clients
            split_layer: The split layer used in this round
            total_layers: Total number of layers in the model
            
        Returns:
            reward: Computed reward value
        """
        # Calculate accuracy improvement
        delta_acc = new_acc - prev_acc
        
        # Get cluster state for selected clients
        cluster_state = self.get_cluster_state(client_indices)
        avg_overall_capability = cluster_state[4]  # Index 4 is avg_overall
        
        # Calculate resource efficiency factor
        # Deeper split points (higher layer numbers) are more resource-intensive
        split_depth_ratio = split_layer / total_layers
        
        # Resource efficiency: higher capability clusters can handle deeper splits
        # Lower capability clusters should use shallower splits
        resource_efficiency = avg_overall_capability * (1 - split_depth_ratio)
        
        # Compute reward: accuracy improvement weighted by resource efficiency
        # This encourages accuracy improvement while penalizing resource-intensive splits for low-capability clusters
        reward = delta_acc * (1 + resource_efficiency)
        
        return reward
    
    def compute_decayed_loss_drop_reward(self, prev_loss, current_loss, round_num, total_rounds=None):
        """
        Calculate decayed loss-drop reward as per Algorithm 6 in the paper.
        
        Formula: r_t = -ΔL_t · ρ_t = -(L_t - L_{t-1}) · e^(-λ(t-1))
        
        where:
            ΔL_t = L_t - L_{t-1} (loss change)
            ρ_t = e^(-λ(t-1)) (exponential decay factor)
            λ = ln(2)/T (decay constant, where T is total rounds)
        
        Early rounds receive larger credit, later rounds receive exponentially less.
        
        Args:
            prev_loss: Previous round's cluster loss L_{t-1}
            current_loss: Current round's cluster loss L_t
            round_num: Current round number t (1-indexed)
            total_rounds: Total number of rounds T (optional, for adaptive λ)
            
        Returns:
            reward: Decayed loss-drop reward
        """
        # Calculate loss change (negative means improvement)
        delta_loss = current_loss - prev_loss
        
        # Calculate decay constant λ
        if total_rounds is not None and total_rounds > 0:
            # Adaptive: λ = ln(2)/T ensures ρ_T/2 ≈ 0.5
            lambda_decay = np.log(2) / total_rounds
        else:
            # Fixed: reasonable default for ~50 rounds
            lambda_decay = 0.05
        
        # Calculate exponential decay factor ρ_t = e^(-λ(t-1))
        # Note: t-1 ensures ρ_1 = 1.0 (full weight at first round)
        rho_t = np.exp(-lambda_decay * (round_num - 1))
        
        # Compute reward: -ΔL_t · ρ_t
        # If loss decreases (ΔL_t < 0), reward is positive
        # If loss increases (ΔL_t > 0), reward is negative
        reward = -delta_loss * rho_t
        
        return reward
    
    def compute_size_weighted_cluster_loss(self, client_losses, client_sizes):
        """
        Calculate size-weighted cluster loss.
        
        Formula: L_t = Σ_{k=1}^K ω_k · L_t^(k)
        where ω_k = n_k / Σ_j n_j (proportion of validation samples)
        
        Args:
            client_losses: List of per-client losses [L_t^(1), L_t^(2), ..., L_t^(K)]
            client_sizes: List of client dataset sizes [n_1, n_2, ..., n_K]
            
        Returns:
            cluster_loss: Size-weighted average loss
        """
        client_losses = np.array(client_losses)
        client_sizes = np.array(client_sizes)
        
        # Calculate weights ω_k
        total_samples = np.sum(client_sizes)
        if total_samples == 0:
            # Fallback to uniform weights
            weights = np.ones(len(client_sizes)) / len(client_sizes)
        else:
            weights = client_sizes / total_samples
        
        # Weighted average
        cluster_loss = np.sum(weights * client_losses)
        
        return cluster_loss
    
    def compute_reward(self, prev_acc, new_acc, client_class_dist, client_part_freq, client_size):
        """
        Legacy reward function for backward compatibility.
        """
        # Calculate accuracy improvement
        delta_acc = new_acc - prev_acc
        
        # Calculate KL divergence between client and global distributions
        kl_divergence = self.compute_kl_divergence(client_class_dist, self.global_class_dist)
        
        participation_factor = 1 + self.beta * client_part_freq
        kl_factor = 1 + self.alpha * kl_divergence
        size_factor = np.log(1 + client_size)
        
        denominator = participation_factor * kl_factor * size_factor
        
        # Avoid division by zero
        if denominator == 0:
            denominator = 1e-8
            
        reward = delta_acc / denominator
        
        return reward

    def step(self, selected_client_indexes, prev_acc, new_acc, client_distributions, participation_freq, client_sizes, split_layer=None, total_layers=None):
        """
        Simulate FL training step and compute reward.
        
        Args:
            selected_client_indexes: Indices of selected clients
            prev_acc: Previous global accuracy
            new_acc: New global accuracy
            client_distributions: Class distributions for all clients
            participation_freq: Participation frequency for all clients
            client_sizes: Dataset sizes for all clients
            split_layer: Split layer used (for capability-aware reward)
            total_layers: Total layers in model (for capability-aware reward)
            
        Returns:
            reward: Computed reward value
        """
        if split_layer is not None and total_layers is not None:
            # Use capability-aware reward function
            return self.compute_capability_aware_reward(
                prev_acc, new_acc, selected_client_indexes, split_layer, total_layers
            )
        else:
            # Use legacy reward function for backward compatibility
            rewards = []
            for c in selected_client_indexes:
                reward = self.compute_reward(
                    prev_acc,
                    new_acc,
                    client_distributions[c],
                    participation_freq[c],
                    client_sizes[c]
                )
                rewards.append(reward)
            
            return np.mean(rewards)
