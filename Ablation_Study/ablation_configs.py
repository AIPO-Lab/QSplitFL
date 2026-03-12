"""
Ablation Study Configurations

Each AblationConfig defines one experimental variant that isolates a single
design choice of the QSplitFL framework.  All configurations are evaluated
on CIFAR-10, ResNet50, 10 clients, 100 rounds.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class AblationConfig:
    """One ablation configuration."""
    name: str
    description: str

    # RL controller settings
    committee_size: int = 3        # M: number of DQN heads (1, 3, or 5)
    use_rl: bool = True            # False → random/fixed split (no RL)

    # Reward function
    lambda_decay: float = 0.05    # λ in ρ_t = exp(-λ(t-1)); 0 = flat reward

    # State representation
    state_mode: str = 'full'      # 'full' (6D) or 'cpu_only' (CPU metric only)
    capability_weights: List[float] = field(
        default_factory=lambda: [0.4, 0.3, 0.2, 0.1]
    )                              # [w_CPU, w_Memory, w_Battery, w_Network]


# ---------------------------------------------------------------------------
# 8 ablation configurations
# ---------------------------------------------------------------------------

ABLATION_CONFIGS: List[AblationConfig] = [

    # 1. Full QSplitFL — the baseline / proposed method
    AblationConfig(
        name="Full QSplitFL",
        description="Full model: committee M=3, decayed reward (λ=0.05), "
                    "capability-aware 6D state with tuned weights",
        committee_size=3,
        use_rl=True,
        lambda_decay=0.05,
        state_mode='full',
        capability_weights=[0.4, 0.3, 0.2, 0.1],
    ),

    # 2. Single-head DQN (ablate committee)
    AblationConfig(
        name="Single-head DQN (M=1)",
        description="Single DQN head — no committee voting; vulnerable to "
                    "reward hacking",
        committee_size=1,
        use_rl=True,
        lambda_decay=0.05,
        state_mode='full',
        capability_weights=[0.4, 0.3, 0.2, 0.1],
    ),

    # 3. Larger committee M=5
    AblationConfig(
        name="Committee M=5",
        description="Larger committee of 5 DQN heads with majority voting",
        committee_size=5,
        use_rl=True,
        lambda_decay=0.05,
        state_mode='full',
        capability_weights=[0.4, 0.3, 0.2, 0.1],
    ),

    # 4. No reward decay (ablate decay factor)
    AblationConfig(
        name="No Decay (λ=0)",
        description="Flat reward — λ=0 so ρ_t=1 for all rounds; all rounds "
                    "weighted equally",
        committee_size=3,
        use_rl=True,
        lambda_decay=0.0,
        state_mode='full',
        capability_weights=[0.4, 0.3, 0.2, 0.1],
    ),

    # 5. High reward decay
    AblationConfig(
        name="High Decay (λ=0.1)",
        description="Aggressive exponential decay — early rounds dominate, "
                    "later rounds heavily discounted",
        committee_size=3,
        use_rl=True,
        lambda_decay=0.1,
        state_mode='full',
        capability_weights=[0.4, 0.3, 0.2, 0.1],
    ),

    # 6. Equal capability weights (ablate tuned weights)
    AblationConfig(
        name="Equal Capability Weights",
        description="Uniform weights wᵢ=0.25 for all four capability metrics; "
                    "no domain-specific prioritization",
        committee_size=3,
        use_rl=True,
        lambda_decay=0.05,
        state_mode='full',
        capability_weights=[0.25, 0.25, 0.25, 0.25],
    ),

    # 7. CPU-only state (ablate rich state representation)
    AblationConfig(
        name="CPU-only State",
        description="State uses only CPU metric; memory/battery/network "
                    "dimensions set to zero — tests value of rich state",
        committee_size=3,
        use_rl=True,
        lambda_decay=0.05,
        state_mode='cpu_only',
        capability_weights=[1.0, 0.0, 0.0, 0.0],
    ),

    # 8. Random split — no RL at all (lower bound baseline)
    AblationConfig(
        name="Random Split (No RL)",
        description="No RL — split layer fixed at ⌈L/2⌉=25 every round; "
                    "tests contribution of adaptive RL policy",
        committee_size=3,
        use_rl=False,
        lambda_decay=0.05,
        state_mode='full',
        capability_weights=[0.4, 0.3, 0.2, 0.1],
    ),
]

# Short labels for plotting
CONFIG_LABELS = [
    "Full QSplitFL",
    "Single-head (M=1)",
    "Committee M=5",
    "No Decay (λ=0)",
    "High Decay (λ=0.1)",
    "Equal Weights",
    "CPU-only State",
    "Random Split",
]

