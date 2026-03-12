"""
Configurable FL Environment for Ablation Study

Supports all state-representation and reward-function variants used in the ablation study:
  - state_mode='full'     → 6D capability vector (Eq. in paper §4.1)
  - state_mode='cpu_only' → CPU metric only; other dims zeroed
  - capability_weights    → [w_CPU, w_Mem, w_Bat, w_Net] for Overall score
  - lambda_decay          → λ in ρ_t = exp(-λ(t-1)); 0 = flat (no decay)
"""

import math
import numpy as np


class FL_Environment:
    """
    Simulates the federated client cluster environment.

    Each 'client' is represented by a dictionary of 4 capability metrics,
    all normalised to [0, 1]:
        CPU      — CPU availability  (higher = more free compute)
        Memory   — available memory
        Battery  — remaining battery level
        Network  — network quality  (1 − normalised_latency)

    The cluster-level state vector s_t (6D) is:
        [avg_CPU, avg_Memory, avg_Battery, avg_Network, avg_Overall, σ_Overall]

    For state_mode='cpu_only', Memory/Battery/Network slots are set to 0.
    """

    def __init__(
        self,
        num_clients: int,
        state_mode: str = 'full',
        capability_weights=None,
        lambda_decay: float = 0.05,
        seed: int = 42,
    ):
        """
        Args:
            num_clients:         Number of clients in the cluster.
            state_mode:          'full' or 'cpu_only'.
            capability_weights:  [w_CPU, w_Mem, w_Bat, w_Net] summing to 1.
            lambda_decay:        λ for exponential reward decay; 0 = flat.
            seed:                RNG seed for capability initialisation.
        """
        self.num_clients = num_clients
        self.state_mode  = state_mode
        self.lambda_decay = lambda_decay

        if capability_weights is None:
            self.capability_weights = [0.4, 0.3, 0.2, 0.1]
        else:
            assert len(capability_weights) == 4
            self.capability_weights = list(capability_weights)

        self.rng = np.random.default_rng(seed)
        self.capabilities = self._init_capabilities()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_capabilities(self):
        """Draw initial capabilities uniformly at random."""
        caps = []
        for _ in range(self.num_clients):
            caps.append({
                'CPU':     float(self.rng.uniform(0.3, 1.0)),
                'Memory':  float(self.rng.uniform(0.4, 1.0)),
                'Battery': float(self.rng.uniform(0.2, 1.0)),
                'Network': float(self.rng.uniform(0.3, 1.0)),
            })
        return caps

    # ------------------------------------------------------------------
    # Capability helpers
    # ------------------------------------------------------------------

    def _overall(self, client_idx: int) -> float:
        """Weighted overall capability score for one client."""
        c = self.capabilities[client_idx]
        w = self.capability_weights
        return (w[0] * c['CPU'] + w[1] * c['Memory'] +
                w[2] * c['Battery'] + w[3] * c['Network'])

    def update_capabilities(self, delta: float = 0.05):
        """
        Simulate dynamic capability fluctuations between rounds.
        Each metric drifts by a small random amount, clamped to [0.1, 1.0].
        """
        for cap in self.capabilities:
            for key in ('CPU', 'Memory', 'Battery', 'Network'):
                noise = float(self.rng.uniform(-delta, delta))
                cap[key] = float(np.clip(cap[key] + noise, 0.1, 1.0))

    # ------------------------------------------------------------------
    # State construction  (Algorithm 1, Steps 1–3 in paper)
    # ------------------------------------------------------------------

    def get_cluster_state(self, client_indices=None) -> np.ndarray:
        """
        Compute the 6-dimensional cluster state vector s_t.

        Args:
            client_indices: List of client indices in the cluster.
                            Defaults to all clients.

        Returns:
            state: np.ndarray of shape (6,), values in [0, 1].
        """
        if client_indices is None:
            client_indices = list(range(self.num_clients))

        cpus     = [self.capabilities[i]['CPU']     for i in client_indices]
        mems     = [self.capabilities[i]['Memory']  for i in client_indices]
        bats     = [self.capabilities[i]['Battery'] for i in client_indices]
        nets     = [self.capabilities[i]['Network'] for i in client_indices]
        overalls = [self._overall(i)                for i in client_indices]

        avg_cpu      = float(np.mean(cpus))
        avg_memory   = float(np.mean(mems))
        avg_battery  = float(np.mean(bats))
        avg_network  = float(np.mean(nets))
        avg_overall  = float(np.mean(overalls))
        std_overall  = float(np.std(overalls))

        if self.state_mode == 'cpu_only':
            # Ablation: only expose CPU information; zero-out other dims
            std_cpu = float(np.std(cpus))
            state = np.array([
                avg_cpu,    # dim 0: CPU
                0.0,        # dim 1: Memory  → masked
                0.0,        # dim 2: Battery → masked
                0.0,        # dim 3: Network → masked
                avg_cpu,    # dim 4: Overall = CPU only
                std_cpu,    # dim 5: Heterogeneity = CPU std
            ], dtype=np.float32)
        else:
            # Full 6D state (default)
            state = np.array([
                avg_cpu,
                avg_memory,
                avg_battery,
                avg_network,
                avg_overall,
                std_overall,
            ], dtype=np.float32)

        return state

    # ------------------------------------------------------------------
    # Reward computation  (paper §4.2)
    # ------------------------------------------------------------------

    def compute_reward(
        self,
        current_loss: float,
        prev_loss: float,
        round_num: int,
    ) -> float:
        """
        Decayed loss-drop reward:
            ρ_t = exp(-λ (t−1))
            r_t = −(L_t − L_{t−1}) × ρ_t

        With lambda_decay=0 (no-decay ablation): ρ_t = 1 for all t.
        With lambda_decay=0.1 (high-decay ablation): ρ_t decays quickly.

        Args:
            current_loss: L_t  (cluster loss this round).
            prev_loss:    L_{t−1} (cluster loss last round).
            round_num:    t  (1-indexed).

        Returns:
            Scalar reward.
        """
        rho = math.exp(-self.lambda_decay * (round_num - 1))
        return -(current_loss - prev_loss) * rho

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def capability_summary(self) -> str:
        """Human-readable summary of current mean capabilities."""
        idxs = list(range(self.num_clients))
        s = self.get_cluster_state(idxs)
        return (
            f"CPU={s[0]:.2f} Mem={s[1]:.2f} Bat={s[2]:.2f} "
            f"Net={s[3]:.2f} Overall={s[4]:.2f} σ={s[5]:.2f}"
        )


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing FL_Environment…")

    env_full = FL_Environment(num_clients=10, state_mode='full',
                              capability_weights=[0.4, 0.3, 0.2, 0.1],
                              lambda_decay=0.05)
    s = env_full.get_cluster_state()
    print(f"Full state: {s}")

    env_cpu = FL_Environment(num_clients=10, state_mode='cpu_only',
                             lambda_decay=0.0)
    s2 = env_cpu.get_cluster_state()
    print(f"CPU-only state: {s2}")

    for lam, t in [(0.0, 5), (0.05, 5), (0.1, 5)]:
        env = FL_Environment(10, lambda_decay=lam)
        r = env.compute_reward(current_loss=0.60, prev_loss=0.70, round_num=t)
        rho = math.exp(-lam * (t - 1))
        print(f"  λ={lam}, t={t}: ρ={rho:.4f}  reward={r:.5f}")

    print("✓ FL_Environment OK")
