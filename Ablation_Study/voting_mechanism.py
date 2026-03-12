"""
Majority Voting Mechanism for Committee-Based Action Selection (Algorithm 3)

Each committee member proposes its preferred split layer.
Majority vote selects the final action; ties are broken by mean Q-value.
"""

import numpy as np
from collections import Counter


def majority_vote(proposals, state, committee_members, action_space):
    """
    Select split layer via majority vote.

    Args:
        proposals:         List[int] — one proposed split layer per member.
        state:             Current state vector (for tie-breaking Q-values).
        committee_members: List of objects with a get_q_value(state, action) method.
        action_space:      List[int] — all valid split layers.

    Returns:
        selected_action (int), vote_counts (dict), is_tie (bool)
    """
    vote_counts = Counter(proposals)
    max_votes   = max(vote_counts.values())
    winners     = [a for a, v in vote_counts.items() if v == max_votes]

    if len(winners) == 1:
        return winners[0], dict(vote_counts), False

    # Tie-breaking: choose the winner with the highest mean Q-value
    mean_q = {}
    for action in winners:
        qs = [m.get_q_value(state, action) for m in committee_members]
        mean_q[action] = float(np.mean(qs))

    selected = max(mean_q, key=mean_q.get)
    return selected, dict(vote_counts), True


def vote_confidence(vote_counts: dict, committee_size: int) -> float:
    """Fraction of committee members that voted for the winning action."""
    if not vote_counts:
        return 0.0
    return max(vote_counts.values()) / committee_size
