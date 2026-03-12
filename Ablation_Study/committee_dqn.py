"""
Committee-Based DQN Agent (Algorithm 1 / Algorithm 3 in QSplitFL paper)

Supports all committee-size ablations:
  M=1  — single DQN head (no voting; shows reward-hacking vulnerability)
  M=3  — full QSplitFL (default, odd → no ties)
  M=5  — larger committee

Architecture:
  Shared encoder  f_s(·; φ_s)        : 6 → 32 → 32
  M independent heads g^(m)(·; ψ^(m)): 32 → 64 → 32 → |A|

Q-value: Q^(m)(s,a) = g^(m)(f_s(s))_a
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from voting_mechanism import majority_vote


# ---------------------------------------------------------------------------
# Network modules
# ---------------------------------------------------------------------------

class SharedEncoder(nn.Module):
    """Shallow shared encoder — same feature space for all committee members."""

    def __init__(self, input_size: int = 6, hidden_sizes=(32, 32)):
        super().__init__()
        layers, prev = [], input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.net         = nn.Sequential(*layers)
        self.output_size = prev

    def forward(self, x):
        return self.net(x)


class DQNHead(nn.Module):
    """Personalised head for one committee member."""

    def __init__(self, input_size: int = 32, hidden_sizes=(64, 32),
                 action_size: int = 25):
        super().__init__()
        layers, prev = [], input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, action_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Committee DQN
# ---------------------------------------------------------------------------

class CommitteeDQN:
    """
    Committee of M DQN heads with shared encoder and majority-voting
    action selection.  Supports M=1 (single head, no real voting).
    """

    def __init__(
        self,
        committee_size: int  = 3,
        state_size: int      = 6,
        action_size: int     = 25,
        total_layers: int    = 50,
        learning_rate: float = 1e-3,
        gamma: float         = 0.2,
        epsilon_start: float = 1.0,
        epsilon_end: float   = 0.01,
        kappa: float         = 0.05,
        buffer_capacity: int = 10_000,
        batch_size: int      = 32,
        target_update_freq: int = 5,
    ):
        """
        Args:
            committee_size:    M (1, 3, or 5).
            state_size:        Dimension of state vector (6).
            action_size:       Number of discrete split actions (|A| = 25).
            total_layers:      L (50 for ResNet50).
            learning_rate:     η.
            gamma:             Discount factor γ.
            epsilon_start:     ε₀.
            epsilon_end:       ε_min.
            kappa:             Decay constant κ for ε-greedy schedule.
            buffer_capacity:   Replay buffer size N_max.
            batch_size:        Mini-batch size B.
            target_update_freq: Sync target nets every C rounds.
        """
        # M=1 is valid (single-head ablation); only require M is a positive integer
        assert committee_size >= 1, "committee_size must be >= 1"

        self.M          = committee_size
        self.state_size = state_size
        self.action_size = action_size
        self.min_split  = int(np.ceil(total_layers / 2))   # 25
        self.max_split  = total_layers - 1                 # 49
        self.gamma      = gamma
        self.epsilon    = epsilon_start
        self.eps0       = epsilon_start
        self.eps_min    = epsilon_end
        self.kappa      = kappa
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Shared encoder
        self.encoder   = SharedEncoder(input_size=state_size)
        self.enc_opt   = optim.Adam(self.encoder.parameters(), lr=learning_rate)

        # M online heads + M target heads
        self.heads        = nn.ModuleList(
            [DQNHead(self.encoder.output_size, action_size=action_size)
             for _ in range(self.M)]
        )
        self.target_heads = nn.ModuleList(
            [DQNHead(self.encoder.output_size, action_size=action_size)
             for _ in range(self.M)]
        )
        for m in range(self.M):
            self.target_heads[m].load_state_dict(self.heads[m].state_dict())

        # Per-head optimisers
        self.head_opts = [
            optim.Adam(self.heads[m].parameters(), lr=learning_rate)
            for m in range(self.M)
        ]

        self.loss_fn = nn.MSELoss()

        # Experience replay buffer
        self.buffer       = deque(maxlen=buffer_capacity)
        self.train_steps  = 0

    # ------------------------------------------------------------------
    # Q-value helper (used in tie-breaking inside majority_vote)
    # ------------------------------------------------------------------

    def get_q_value(self, state, action: int) -> float:
        """Mean Q-value across all heads for tie-breaking."""
        s = torch.FloatTensor(state).unsqueeze(0)
        idx = action - self.min_split
        qs = []
        with torch.no_grad():
            enc = self.encoder(s)
            for m in range(self.M):
                qs.append(self.heads[m](enc)[0, idx].item())
        return float(np.mean(qs))

    # ------------------------------------------------------------------
    # Action selection  (ε-greedy + committee vote)
    # ------------------------------------------------------------------

    def select_action(self, state) -> tuple:
        """
        Select a split layer.

        Returns:
            (action, info_dict)
        """
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.min_split, self.max_split + 1)
            return action, {'method': 'random', 'epsilon': self.epsilon}

        s = torch.FloatTensor(state).unsqueeze(0)
        proposals = []
        with torch.no_grad():
            enc = self.encoder(s)
            for m in range(self.M):
                idx    = self.heads[m](enc).argmax().item()
                action = self.min_split + idx
                proposals.append(action)

        if self.M == 1:
            # No voting needed for a single head
            action = proposals[0]
            return action, {'method': 'greedy', 'proposals': proposals}

        action_space = list(range(self.min_split, self.max_split + 1))

        # Build lightweight wrapper objects for majority_vote tie-breaking
        class _Wrapper:
            def __init__(w, head, encoder, min_split):
                w.head, w.enc, w.min_split = head, encoder, min_split
            def get_q_value(w, state, action):
                s_t  = torch.FloatTensor(state).unsqueeze(0)
                idx  = action - w.min_split
                with torch.no_grad():
                    f = w.enc(s_t)
                    return w.head(f)[0, idx].item()

        wrappers = [_Wrapper(self.heads[m], self.encoder, self.min_split)
                    for m in range(self.M)]
        action, vote_counts, is_tie = majority_vote(
            proposals, state, wrappers, action_space
        )
        return action, {'method': 'vote', 'proposals': proposals,
                        'vote_counts': vote_counts, 'is_tie': is_tie}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def train_step(self) -> float:
        """One gradient step on a random mini-batch. Returns avg TD loss."""
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch  = random.sample(self.buffer, self.batch_size)
        states      = torch.FloatTensor([b[0] for b in batch])
        actions     = torch.LongTensor( [b[1] - self.min_split for b in batch])
        rewards     = torch.FloatTensor([b[2] for b in batch])
        next_states = torch.FloatTensor([b[3] for b in batch])
        dones       = torch.BoolTensor( [b[4] for b in batch])

        # Shared encoder features
        enc_feat      = self.encoder(states)
        enc_feat_next = self.encoder(next_states).detach()

        total_loss = 0.0
        for m in range(self.M):
            # TD target  y = r + γ (1−d) max_a Q̄(s', a)
            with torch.no_grad():
                max_next_q = self.target_heads[m](enc_feat_next).max(1)[0]
                td_target  = rewards + self.gamma * (~dones).float() * max_next_q

            # Current Q estimate
            q_pred = self.heads[m](enc_feat).gather(
                1, actions.unsqueeze(1)
            ).squeeze(1)

            loss = self.loss_fn(q_pred, td_target)
            total_loss += loss.item()

            self.head_opts[m].zero_grad()
            retain = (m < self.M - 1)   # retain graph for intermediate heads
            loss.backward(retain_graph=retain)
            self.head_opts[m].step()

        # Encoder gradient pass (use last head's loss surface)
        enc_feat2 = self.encoder(states)
        enc_loss  = sum(
            self.loss_fn(
                self.heads[m](enc_feat2).gather(1, actions.unsqueeze(1)).squeeze(1),
                rewards + self.gamma * (~dones).float() *
                self.target_heads[m](self.encoder(next_states).detach()).max(1)[0].detach()
            )
            for m in range(self.M)
        )
        self.enc_opt.zero_grad()
        enc_loss.backward()
        self.enc_opt.step()

        self.train_steps += 1
        return total_loss / self.M

    def update_target_networks(self):
        """Hard-copy online heads to target heads."""
        for m in range(self.M):
            self.target_heads[m].load_state_dict(self.heads[m].state_dict())

    def decay_epsilon(self, round_num: int):
        """ε_t = ε_min + (ε₀ − ε_min) exp(−κ t)."""
        self.epsilon = self.eps_min + (self.eps0 - self.eps_min) * \
                       np.exp(-self.kappa * round_num)

    def should_update_target(self, round_num: int) -> bool:
        return round_num % self.target_update_freq == 0


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing CommitteeDQN…")
    for M in (1, 3, 5):
        agent = CommitteeDQN(committee_size=M, action_size=25, total_layers=50)
        state = np.random.rand(6).astype(np.float32)
        action, info = agent.select_action(state)
        print(f"  M={M}: action={action}  info={info['method']}")

        # Fill buffer and train
        for _ in range(50):
            a = np.random.randint(25, 50)
            agent.store_transition(state, a, np.random.randn(),
                                   np.random.rand(6), False)
        loss = agent.train_step()
        print(f"         train_loss={loss:.4f}")

    print("✓ CommitteeDQN OK")
