# Reward Function Explanation: Capability-Aware Q-Learning for Split Point Optimization

## Overview

The reward function is the **core signal** that guides the DQN agent to learn optimal split point selection. It balances two critical objectives:
1. **Model accuracy improvement** (primary goal)
2. **Resource efficiency** (matching split depth to client capabilities)

---

## The Reward Function Formula

### Mathematical Definition

```
r_t = Δ_accuracy × (1 + resource_efficiency)

where:
  Δ_accuracy = new_accuracy - previous_accuracy
  resource_efficiency = avg_capability × (1 - split_depth_ratio)
  split_depth_ratio = split_layer / total_layers
```

### Implementation (from `environment.py`)

```python
def compute_capability_aware_reward(self, prev_acc, new_acc, client_indices, 
                                    split_layer, total_layers):
    # 1. Calculate accuracy improvement
    delta_acc = new_acc - prev_acc
    
    # 2. Get average capability of selected clients
    cluster_state = self.get_cluster_state(client_indices)
    avg_overall_capability = cluster_state[4]  # Overall capability score
    
    # 3. Calculate how deep the split is (0 to 1 scale)
    split_depth_ratio = split_layer / total_layers
    
    # 4. Resource efficiency: capability × (1 - depth)
    resource_efficiency = avg_overall_capability * (1 - split_depth_ratio)
    
    # 5. Final reward: accuracy × (1 + efficiency)
    reward = delta_acc * (1 + resource_efficiency)
    
    return reward
```

---

## Component Breakdown

### 1. Accuracy Improvement (Δ_accuracy)

**What it measures:**
- Change in global model accuracy after this training round
- Primary indicator of learning progress

**Values:**
- **Positive**: Model improved (good!)
- **Negative**: Model degraded (bad!)
- **Zero**: No change in accuracy

**Example:**
```
Previous accuracy: 0.75
New accuracy: 0.78
Δ_accuracy = 0.78 - 0.75 = +0.03 (3% improvement)
```

---

### 2. Split Depth Ratio

**What it measures:**
- How deep in the model we're splitting
- Normalized to [0, 1] range

**Formula:**
```
split_depth_ratio = split_layer / total_layers
```

**Example (10-layer model):**
```
Split at layer 5: 5/10 = 0.5 (50% depth - shallow split)
Split at layer 7: 7/10 = 0.7 (70% depth - medium split)
Split at layer 9: 9/10 = 0.9 (90% depth - deep split)
```

**Interpretation:**
- **Low ratio (0.5)**: Shallow split → Client does less work, server does more
- **High ratio (0.9)**: Deep split → Client does more work, server does less

---

### 3. Average Overall Capability

**What it measures:**
- Combined hardware/network capability of selected clients
- Weighted average of CPU, Memory, Battery, Network

**Formula:**
```
avg_capability = mean([C_overall^(k) for k in selected_clients])

where:
  C_overall^(k) = w1×CPU + w2×Memory + w3×Battery + w4×Network
```

**Example:**
```
Client 0: CPU=0.8, Memory=0.7, Battery=0.9, Network=0.6
Client 1: CPU=0.5, Memory=0.6, Battery=0.4, Network=0.7
Client 2: CPU=0.9, Memory=0.8, Battery=0.8, Network=0.9

With equal weights (0.25 each):
  C_overall^(0) = 0.25×0.8 + 0.25×0.7 + 0.25×0.9 + 0.25×0.6 = 0.75
  C_overall^(1) = 0.25×0.5 + 0.25×0.6 + 0.25×0.4 + 0.25×0.7 = 0.55
  C_overall^(2) = 0.25×0.9 + 0.25×0.8 + 0.25×0.8 + 0.25×0.9 = 0.85

avg_capability = (0.75 + 0.55 + 0.85) / 3 = 0.72
```

---

### 4. Resource Efficiency

**What it measures:**
- How well the split point matches client capabilities
- Encourages deep splits for capable clients, shallow splits for weak clients

**Formula:**
```
resource_efficiency = avg_capability × (1 - split_depth_ratio)
```

**Key Insight:**
- **(1 - split_depth_ratio)** represents "how much work the server does"
- Multiplying by **avg_capability** creates the matching incentive

**Example Scenarios:**

#### Scenario A: High-Capability Cluster + Deep Split ✓
```
avg_capability = 0.9 (strong clients)
split_layer = 9, total_layers = 10
split_depth_ratio = 0.9
resource_efficiency = 0.9 × (1 - 0.9) = 0.9 × 0.1 = 0.09
```
**Low efficiency bonus** - but that's okay! Strong clients can handle deep splits.

#### Scenario B: High-Capability Cluster + Shallow Split ✓✓
```
avg_capability = 0.9 (strong clients)
split_layer = 5, total_layers = 10
split_depth_ratio = 0.5
resource_efficiency = 0.9 × (1 - 0.5) = 0.9 × 0.5 = 0.45
```
**High efficiency bonus** - great! Underutilizing strong clients but very efficient.

#### Scenario C: Low-Capability Cluster + Shallow Split ✓✓
```
avg_capability = 0.3 (weak clients)
split_layer = 5, total_layers = 10
split_depth_ratio = 0.5
resource_efficiency = 0.3 × (1 - 0.5) = 0.3 × 0.5 = 0.15
```
**Medium efficiency bonus** - good match! Weak clients do less work.

#### Scenario D: Low-Capability Cluster + Deep Split ✗
```
avg_capability = 0.3 (weak clients)
split_layer = 9, total_layers = 10
split_depth_ratio = 0.9
resource_efficiency = 0.3 × (1 - 0.9) = 0.3 × 0.1 = 0.03
```
**Very low efficiency bonus** - bad match! Weak clients overburdened.

---

## Complete Reward Calculation Examples

### Example 1: Good Match - High Capability + Deep Split

```
Inputs:
  prev_acc = 0.75
  new_acc = 0.78
  avg_capability = 0.9
  split_layer = 8
  total_layers = 10

Calculation:
  Δ_accuracy = 0.78 - 0.75 = 0.03
  split_depth_ratio = 8/10 = 0.8
  resource_efficiency = 0.9 × (1 - 0.8) = 0.18
  
  reward = 0.03 × (1 + 0.18) = 0.03 × 1.18 = 0.0354

Interpretation: Positive reward! Accuracy improved AND split matched capabilities.
```

### Example 2: Bad Match - Low Capability + Deep Split

```
Inputs:
  prev_acc = 0.75
  new_acc = 0.76
  avg_capability = 0.3
  split_layer = 9
  total_layers = 10

Calculation:
  Δ_accuracy = 0.76 - 0.75 = 0.01
  split_depth_ratio = 9/10 = 0.9
  resource_efficiency = 0.3 × (1 - 0.9) = 0.03
  
  reward = 0.01 × (1 + 0.03) = 0.01 × 1.03 = 0.0103

Interpretation: Low reward despite accuracy improvement. Poor capability match.
```

### Example 3: Excellent Match - Low Capability + Shallow Split

```
Inputs:
  prev_acc = 0.75
  new_acc = 0.78
  avg_capability = 0.3
  split_layer = 5
  total_layers = 10

Calculation:
  Δ_accuracy = 0.78 - 0.75 = 0.03
  split_depth_ratio = 5/10 = 0.5
  resource_efficiency = 0.3 × (1 - 0.5) = 0.15
  
  reward = 0.03 × (1 + 0.15) = 0.03 × 1.15 = 0.0345

Interpretation: Good reward! Accuracy improved AND weak clients not overburdened.
```

### Example 4: Negative Reward - Accuracy Degradation

```
Inputs:
  prev_acc = 0.75
  new_acc = 0.73
  avg_capability = 0.9
  split_layer = 7
  total_layers = 10

Calculation:
  Δ_accuracy = 0.73 - 0.75 = -0.02
  split_depth_ratio = 7/10 = 0.7
  resource_efficiency = 0.9 × (1 - 0.7) = 0.27
  
  reward = -0.02 × (1 + 0.27) = -0.02 × 1.27 = -0.0254

Interpretation: Negative reward! Accuracy decreased, so penalized regardless of efficiency.
```

---

## Reward Function Design Rationale

### Why This Design?

1. **Accuracy First**: The `Δ_accuracy` term ensures accuracy improvement is the primary goal
   - Positive accuracy change → positive reward
   - Negative accuracy change → negative reward (penalty)

2. **Efficiency Bonus**: The `(1 + resource_efficiency)` term acts as a multiplier
   - Good capability match → higher multiplier (1.2-1.5×)
   - Poor capability match → lower multiplier (1.0-1.1×)
   - Never makes reward negative if accuracy improves

3. **Adaptive Behavior**: Encourages the agent to:
   - Use deep splits when clients are capable
   - Use shallow splits when clients are weak
   - Balance accuracy and resource constraints

---

## What the Agent Learns

Over time, the DQN learns these patterns:

### Pattern 1: Capability-Split Matching
```
IF avg_capability > 0.8 THEN prefer split_layer ∈ {8, 9}
IF avg_capability ≈ 0.5 THEN prefer split_layer ∈ {6, 7}
IF avg_capability < 0.3 THEN prefer split_layer ∈ {5, 6}
```

### Pattern 2: Variance Consideration
```
IF capability_variance > 0.3 THEN be more conservative (shallower split)
  → Heterogeneous cluster, some weak clients
```

### Pattern 3: Accuracy Prioritization
```
IF accuracy is improving THEN continue similar split strategy
IF accuracy is degrading THEN explore different split points
```

---

## Comparison with Alternative Reward Functions

### Alternative 1: Pure Accuracy (No Efficiency)
```python
reward = delta_acc  # Simple but ignores resources
```
**Problem**: Agent might choose splits that overburden weak clients

### Alternative 2: Separate Penalties
```python
reward = delta_acc - penalty_for_bad_match
```
**Problem**: Can make reward negative even when accuracy improves

### Alternative 3: Multi-Objective
```python
reward = alpha × delta_acc + beta × efficiency
```
**Problem**: Hard to tune alpha and beta weights

### Current Design (Multiplicative)
```python
reward = delta_acc × (1 + resource_efficiency)
```
**Advantages**:
- Accuracy always dominates (multiplicative factor)
- Efficiency provides bonus/penalty through multiplier
- No negative rewards when accuracy improves
- Self-balancing (no hyperparameters to tune)

---

## Practical Implications

### For Training:
- **Early rounds**: Agent explores different split points randomly
- **Mid training**: Agent starts matching splits to capabilities
- **Late training**: Agent consistently chooses optimal splits for each capability level

### For Deployment:
- **High-capability clients**: Automatically get deeper splits (utilize their resources)
- **Low-capability clients**: Automatically get shallow splits (reduce their burden)
- **Mixed clusters**: Balanced split based on average capability

### For System Performance:
- **Better resource utilization**: Strong clients do more work
- **Fairness**: Weak clients not overburdened
- **Adaptability**: Automatically adjusts to changing client capabilities
- **Efficiency**: Reduces communication overhead by optimal splitting

---

## Summary

The capability-aware reward function:

```
reward = Δ_accuracy × (1 + avg_capability × (1 - split_depth_ratio))
```

**Encourages**:
✓ Accuracy improvement (primary objective)
✓ Deep splits for capable clients
✓ Shallow splits for weak clients
✓ Resource-efficient federated learning

**Discourages**:
✗ Accuracy degradation
✗ Overburdening weak clients
✗ Underutilizing strong clients
✗ Resource-inefficient splits

This design enables the DQN agent to learn an adaptive split point selection policy that balances model performance with client resource constraints!
