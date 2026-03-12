# Current Implementation: State Space, Action Space, and Reward Function

## Overview
This document describes the **current implementation** of the RL framework in your codebase, showing exactly what state, action, and reward representations are being used.

---

## 1. State Space (S)

### **Representation**: 6-Dimensional Capability Vector

The state is a **cluster-level capability vector** that captures both the average hardware capabilities and the heterogeneity of selected clients.

### **Mathematical Definition**
```
s_t^(c) = [C̄_CPU, C̄_Memory, C̄_Battery, C̄_Network, C̄_Overall, σ_c]
```

### **Dimensions** (all normalized to [0, 1])

| Index | Dimension | Description | Formula |
|-------|-----------|-------------|---------|
| 0 | `avg_cpu` | Average CPU capability | `mean(C_CPU^(k))` for k in cluster |
| 1 | `avg_memory` | Average memory capability | `mean(C_Memory^(k))` for k in cluster |
| 2 | `avg_battery` | Average battery level | `mean(C_Battery^(k))` for k in cluster |
| 3 | `avg_network` | Average network capability | `mean(C_Network^(k))` for k in cluster |
| 4 | `avg_overall` | Average overall capability | `mean(C_Overall^(k))` for k in cluster |
| 5 | `capability_variance` | Capability heterogeneity | `std(C_Overall^(k))` for k in cluster |

### **Individual Client Capabilities** (normalized to [0, 1])

Each client k has four capability metrics:

```python
C_CPU^(k) = CPU_available^(k) / CPU_max
C_Memory^(k) = Memory_available^(k) / Memory_max
C_Battery^(k) = Battery_level^(k) / Battery_max
C_Network^(k) = 1 - (current_latency_k / max_acceptable_latency)
```

### **Overall Capability Score** (weighted combination)

```python
C_Overall^(k) = w1·C_CPU + w2·C_Memory + w3·C_Battery + w4·C_Network

where:
  w1 = w2 = w3 = w4 = 0.25 (equal weights by default)
  Σw_i = 1
```

### **Implementation** (`environment.py`)

```python
def get_cluster_state(self, client_indices):
    """
    Compute cluster-level state vector.
    State vector: [avg_CPU, avg_Memory, avg_Battery, avg_Network, 
                   avg_Overall, capability_variance]
    """
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
    
    # Return 6D state vector
    return np.array([avg_cpu, avg_memory, avg_battery, avg_network, 
                     avg_overall, capability_variance])
```

### **Example State Values**

```python
# High-capability, homogeneous cluster
state_high = [0.9, 0.85, 0.9, 0.8, 0.86, 0.04]
#             ↑    ↑     ↑    ↑    ↑     ↑
#             CPU  Mem   Batt Net  Ovr   Var (low heterogeneity)

# Medium-capability, heterogeneous cluster
state_medium = [0.6, 0.55, 0.5, 0.65, 0.58, 0.25]
#               ↑    ↑     ↑    ↑    ↑     ↑
#               CPU  Mem   Batt Net  Ovr   Var (high heterogeneity)

# Low-capability, homogeneous cluster
state_low = [0.3, 0.35, 0.25, 0.4, 0.33, 0.05]
#            ↑    ↑     ↑     ↑   ↑     ↑
#            CPU  Mem   Batt  Net Ovr   Var (low heterogeneity)
```

### **Interpretation**

- **High avg_overall (0.8-1.0)**: Cluster can handle deep splits (layers 8-9)
- **Medium avg_overall (0.5-0.7)**: Cluster should use medium splits (layers 6-7)
- **Low avg_overall (0.2-0.4)**: Cluster needs shallow splits (layers 5-6)
- **High variance (>0.3)**: Heterogeneous cluster, be conservative
- **Low variance (<0.1)**: Homogeneous cluster, can be aggressive

---

## 2. Action Space (A)

### **Representation**: Discrete Split Layer Selection

The action is the **layer index** where the model is split between client and server.

### **Mathematical Definition**
```
A = {ℓ ∈ ℕ : ℓ_min ≤ ℓ ≤ ℓ_max}

where:
  ℓ_min = ⌈L/2⌉  (minimum split point)
  ℓ_max = L - 1  (maximum split point, cannot split after output layer)
```

### **Example** (for L=10 total layers)

```
A = {5, 6, 7, 8, 9}

Action space size: |A| = 5
```

### **Action Interpretation**

| Action (ℓ) | Client Layers | Server Layers | Client Computation | Server Computation |
|------------|---------------|---------------|--------------------|--------------------|
| 5 | 1-5 (50%) | 6-10 (50%) | Low | High |
| 6 | 1-6 (60%) | 7-10 (40%) | Medium-Low | Medium-High |
| 7 | 1-7 (70%) | 8-10 (30%) | Medium | Medium |
| 8 | 1-8 (80%) | 9-10 (20%) | Medium-High | Medium-Low |
| 9 | 1-9 (90%) | 10 (10%) | High | Low |

### **Implementation** (`dqn_agent.py`)

```python
class CapabilityAwareDQN_Agent:
    def __init__(self, state_size=6, action_size=None, total_layers=10, ...):
        if action_size is None:
            # Action space: split points from ceil(L/2) to L-1
            min_split = int(np.ceil(total_layers / 2))  # ⌈10/2⌉ = 5
            max_split = total_layers - 1                 # 10 - 1 = 9
            action_size = max_split - min_split + 1      # 9 - 5 + 1 = 5
        
        self.state_size = state_size        # 6
        self.action_size = action_size      # 5
        self.total_layers = total_layers    # 10
        self.min_split = min_split          # 5
        self.max_split = max_split          # 9
```

### **Action Selection** (ε-greedy)

```python
def select_split_point(self, cluster_state):
    if np.random.rand() < self.epsilon:
        # Exploration: random split point
        return np.random.randint(self.min_split, self.max_split + 1)
    else:
        # Exploitation: best Q-value
        state_tensor = torch.FloatTensor(cluster_state).unsqueeze(0)
        q_values = self.model(state_tensor)
        action_idx = q_values.argmax().item()
        return self.get_action_from_index(action_idx)  # Convert to actual layer
```

### **Mapping Between Action Index and Layer**

```python
# Action index → Actual layer
action_idx = 0 → split_layer = 5
action_idx = 1 → split_layer = 6
action_idx = 2 → split_layer = 7
action_idx = 3 → split_layer = 8
action_idx = 4 → split_layer = 9

# Formula: split_layer = min_split + action_idx
```

---

## 3. Reward Function (R)

### **Current Implementation**: Three Reward Functions Available

The environment supports **three different reward functions** depending on the use case:

---

### **3.1 Capability-Aware Reward** (Current Default)

**Purpose**: Balance accuracy improvement with resource efficiency

**Formula**:
```
r_t = Δ_accuracy × (1 + resource_efficiency)

where:
  Δ_accuracy = new_accuracy - prev_accuracy
  resource_efficiency = avg_capability × (1 - split_depth_ratio)
  split_depth_ratio = split_layer / total_layers
```

**Implementation** (`environment.py`):
```python
def compute_capability_aware_reward(self, prev_acc, new_acc, client_indices, 
                                    split_layer, total_layers):
    # Calculate accuracy improvement
    delta_acc = new_acc - prev_acc
    
    # Get cluster state for selected clients
    cluster_state = self.get_cluster_state(client_indices)
    avg_overall_capability = cluster_state[4]  # Index 4 is avg_overall
    
    # Calculate resource efficiency factor
    split_depth_ratio = split_layer / total_layers
    resource_efficiency = avg_overall_capability * (1 - split_depth_ratio)
    
    # Compute reward
    reward = delta_acc * (1 + resource_efficiency)
    
    return reward
```

**Example Scenarios**:

```python
# Scenario 1: High capability + Deep split + Good accuracy
prev_acc = 0.75, new_acc = 0.78
avg_capability = 0.9, split_layer = 8, total_layers = 10

Δ_accuracy = 0.03
split_depth_ratio = 0.8
resource_efficiency = 0.9 × (1 - 0.8) = 0.18
reward = 0.03 × (1 + 0.18) = 0.0354 ✓ Good!

# Scenario 2: Low capability + Deep split + Small accuracy gain
prev_acc = 0.75, new_acc = 0.76
avg_capability = 0.3, split_layer = 9, total_layers = 10

Δ_accuracy = 0.01
split_depth_ratio = 0.9
resource_efficiency = 0.3 × (1 - 0.9) = 0.03
reward = 0.01 × (1 + 0.03) = 0.0103 ✗ Low reward (bad match)

# Scenario 3: Low capability + Shallow split + Good accuracy
prev_acc = 0.75, new_acc = 0.78
avg_capability = 0.3, split_layer = 5, total_layers = 10

Δ_accuracy = 0.03
split_depth_ratio = 0.5
resource_efficiency = 0.3 × (1 - 0.5) = 0.15
reward = 0.03 × (1 + 0.15) = 0.0345 ✓ Good match!
```

---

### **3.2 Decayed Loss-Drop Reward** (NEW - Just Implemented!)

**Purpose**: Emphasize early improvements, de-emphasize late-stage noise

**Formula**:
```
r_t = -ΔL_t × ρ_t = -(L_t - L_{t-1}) × e^(-λ(t-1))

where:
  ΔL_t = L_t - L_{t-1} (loss change)
  ρ_t = e^(-λ(t-1)) (exponential decay factor)
  λ = ln(2)/T (decay constant, T = total rounds)
```

**Implementation** (`environment.py`):
```python
def compute_decayed_loss_drop_reward(self, prev_loss, current_loss, 
                                     round_num, total_rounds=None):
    # Calculate loss change (negative means improvement)
    delta_loss = current_loss - prev_loss
    
    # Calculate decay constant λ
    if total_rounds is not None and total_rounds > 0:
        lambda_decay = np.log(2) / total_rounds
    else:
        lambda_decay = 0.05  # Default for ~50 rounds
    
    # Calculate exponential decay factor ρ_t = e^(-λ(t-1))
    rho_t = np.exp(-lambda_decay * (round_num - 1))
    
    # Compute reward: -ΔL_t · ρ_t
    reward = -delta_loss * rho_t
    
    return reward
```

**Example Scenarios**:

```python
# Early big improvement (round 2)
prev_loss = 0.80, current_loss = 0.68, round_num = 2, total_rounds = 20

ΔL = -0.12 (improvement!)
λ = ln(2)/20 = 0.0347
ρ_2 = e^(-0.0347×1) = 0.966
reward = -(-0.12) × 0.966 = +0.116 ✓ Strong positive!

# Late small improvement (round 15)
prev_loss = 0.55, current_loss = 0.54, round_num = 15, total_rounds = 20

ΔL = -0.01 (improvement!)
λ = 0.0347
ρ_15 = e^(-0.0347×14) = 0.620
reward = -(-0.01) × 0.620 = +0.0062 ✓ Small positive (decayed)

# Early degradation (round 2)
prev_loss = 0.70, current_loss = 0.74, round_num = 2, total_rounds = 20

ΔL = +0.04 (degradation!)
ρ_2 = 0.966
reward = -(+0.04) × 0.966 = -0.0386 ✗ Negative penalty!
```

**Decay Schedule** (for T=20 rounds):

| Round | ρ_t | Weight |
|-------|-----|--------|
| 1 | 1.000 | 100% |
| 5 | 0.871 | 87% |
| 10 | 0.707 | 71% |
| 15 | 0.620 | 62% |
| 20 | 0.500 | 50% |

---

### **3.3 Size-Weighted Cluster Loss** (NEW - Just Implemented!)

**Purpose**: Compute fair cluster loss considering dataset sizes

**Formula**:
```
L_t = Σ_{k=1}^K ω_k · L_t^(k)

where:
  ω_k = n_k / Σ_j n_j (proportion of validation samples)
  L_t^(k) = loss of client k at round t
```

**Implementation** (`environment.py`):
```python
def compute_size_weighted_cluster_loss(self, client_losses, client_sizes):
    client_losses = np.array(client_losses)
    client_sizes = np.array(client_sizes)
    
    # Calculate weights ω_k
    total_samples = np.sum(client_sizes)
    if total_samples == 0:
        weights = np.ones(len(client_sizes)) / len(client_sizes)
    else:
        weights = client_sizes / total_samples
    
    # Weighted average
    cluster_loss = np.sum(weights * client_losses)
    
    return cluster_loss
```

**Example**:
```python
# Three clients with different dataset sizes
client_losses = [0.5, 0.6, 0.7]
client_sizes = [1000, 500, 200]

total_samples = 1700
weights = [1000/1700, 500/1700, 200/1700] = [0.588, 0.294, 0.118]

cluster_loss = 0.588×0.5 + 0.294×0.6 + 0.118×0.7
             = 0.294 + 0.176 + 0.083
             = 0.553

# Client 1 has most influence (largest dataset)
```

---

## 4. Complete MDP Summary

### **Markov Decision Process Components**

```
State Space (S):
  - Type: Continuous
  - Dimensions: 6
  - Range: [0, 1]^6
  - Representation: Capability vector

Action Space (A):
  - Type: Discrete
  - Size: 5 (for L=10)
  - Range: {5, 6, 7, 8, 9}
  - Representation: Split layer index

Reward Function (R):
  - Type: Continuous
  - Range: ℝ (can be positive or negative)
  - Options: 
    1. Capability-aware (accuracy + efficiency)
    2. Decayed loss-drop (time-weighted improvement)
    3. Size-weighted cluster loss

Transition Function (P):
  - Deterministic given action
  - Capabilities update after each round
  - Stochastic due to training dynamics

Discount Factor (γ):
  - Current: 0.95 (standard)
  - Recommended for loss-based: 0.1-0.3 (focus on immediate)
```

---

## 5. Current vs. Planned Implementations

### **Currently Implemented** ✅
- ✅ 6D capability-aware state representation
- ✅ Discrete action space (split points)
- ✅ Capability-aware reward (accuracy + efficiency)
- ✅ Decayed loss-drop reward (NEW!)
- ✅ Size-weighted cluster loss (NEW!)
- ✅ Basic DQN with experience replay
- ✅ Target network
- ✅ ε-greedy exploration

### **Planned Enhancements** 🚧
- 🚧 Enhanced epsilon decay (exponential)
- 🚧 Reward Committee (Algorithm 5)
- 🚧 Committee DQN (Algorithm 9)
- 🚧 Majority Voting (Algorithm 8)
- 🚧 Multi-episode training
- 🚧 Multi-cluster support

---

## 6. Usage Example

```python
from environment import FL_Environment
from dqn_agent import CapabilityAwareDQN_Agent

# Initialize environment
env = FL_Environment(num_clients=5, global_class_dist=np.ones(10)/10)

# Initialize DQN agent
agent = CapabilityAwareDQN_Agent(
    state_size=6,           # 6D capability state
    total_layers=10,        # 10-layer model
    learning_rate=0.001,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_end=0.01
)

# Training loop
for round_num in range(num_rounds):
    # Get state
    selected_clients = [0, 1, 2]
    state = env.get_cluster_state(selected_clients)
    # state = [0.7, 0.65, 0.8, 0.6, 0.69, 0.15]
    
    # Select action
    split_layer = agent.select_split_point(state)
    # split_layer = 7 (for example)
    
    # Execute training round
    # ... (train clients with split at layer 7)
    
    # Compute reward (option 1: capability-aware)
    reward = env.compute_capability_aware_reward(
        prev_acc=0.75, new_acc=0.78,
        client_indices=selected_clients,
        split_layer=split_layer,
        total_layers=10
    )
    # reward = 0.0345
    
    # OR compute reward (option 2: decayed loss-drop)
    reward = env.compute_decayed_loss_drop_reward(
        prev_loss=0.80, current_loss=0.68,
        round_num=round_num+1,
        total_rounds=50
    )
    # reward = +0.116 (early improvement)
    
    # Get next state
    env.update_client_capabilities()
    next_state = env.get_cluster_state(selected_clients)
    
    # Store experience and train
    agent.remember(state, split_layer, reward, next_state, done=False)
    agent.replay()
```

---

## Summary

**State**: 6D capability vector capturing hardware/network metrics and heterogeneity  
**Action**: Split layer selection from {5, 6, 7, 8, 9} for 10-layer model  
**Reward**: Three options - capability-aware, decayed loss-drop, or size-weighted loss  

This design enables the DQN to learn an adaptive split point selection policy that balances model accuracy with client resource constraints!
