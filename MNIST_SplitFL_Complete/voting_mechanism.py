"""
Majority Voting Mechanism for Committee-Based Action Selection (Algorithm 8)

This module implements majority voting for selecting split layers from a committee
of DQN agents, with tie-breaking using mean Q-values.

Reference: Algorithm 8 in the LaTeX document
"""

import numpy as np
from collections import Counter


def majority_vote(committee_proposals, state, committee_dqns, action_space):
    """
    Select split layer using majority voting from committee proposals.
    
    Algorithm 8: Majority voting to choose the split layer
    
    Args:
        committee_proposals: List of proposed actions from each committee member
        state: Current state (for tie-breaking)
        committee_dqns: List of committee DQN models
        action_space: List of valid actions (split layers)
        
    Returns:
        selected_action: Chosen split layer after voting
        vote_counts: Dictionary of vote counts for each action
        is_tie: Boolean indicating if there was a tie
    """
    # Step 1: Count votes for each action
    vote_counts = Counter(committee_proposals)
    
    # Step 2: Find action(s) with maximum votes
    max_votes = max(vote_counts.values())
    winners = [action for action, count in vote_counts.items() if count == max_votes]
    
    # Step 3: Check if there's a unique winner
    if len(winners) == 1:
        # Unique winner
        return winners[0], dict(vote_counts), False
    
    # Step 4: Tie-breaking using mean Q-values
    print(f"  Tie detected among actions: {winners}")
    
    # Compute mean Q-values for tied actions
    mean_q_values = {}
    for action in winners:
        q_values = []
        for dqn in committee_dqns:
            q_val = dqn.get_q_value(state, action)
            q_values.append(q_val)
        mean_q_values[action] = np.mean(q_values)
    
    # Select action with highest mean Q-value
    selected_action = max(mean_q_values, key=mean_q_values.get)
    
    print(f"  Tie-breaking Q-values: {mean_q_values}")
    print(f"  Selected action: {selected_action}")
    
    return selected_action, dict(vote_counts), True


def get_vote_confidence(vote_counts, committee_size):
    """
    Calculate confidence score based on vote distribution.
    
    Args:
        vote_counts: Dictionary of vote counts
        committee_size: Total number of committee members
        
    Returns:
        confidence: Confidence score (0-1), higher means stronger consensus
    """
    if not vote_counts:
        return 0.0
    
    max_votes = max(vote_counts.values())
    confidence = max_votes / committee_size
    
    return confidence


def get_vote_entropy(vote_counts, committee_size):
    """
    Calculate entropy of vote distribution (measure of disagreement).
    
    Args:
        vote_counts: Dictionary of vote counts
        committee_size: Total number of committee members
        
    Returns:
        entropy: Entropy value, higher means more disagreement
    """
    if not vote_counts:
        return 0.0
    
    # Convert to probabilities
    probabilities = [count / committee_size for count in vote_counts.values()]
    
    # Calculate entropy
    entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probabilities)
    
    return entropy


if __name__ == "__main__":
    # Test majority voting
    print("Testing Majority Voting Mechanism...")
    
    # Mock DQN class for testing
    class MockDQN:
        def __init__(self, q_values):
            self.q_values = q_values
        
        def get_q_value(self, state, action):
            return self.q_values.get(action, 0.0)
    
    # Test case 1: Clear winner
    print("\n=== Test Case 1: Clear Winner ===")
    proposals = [7, 7, 7, 8, 6]  # Layer 7 wins with 3 votes
    committee_size = 5
    action_space = [5, 6, 7, 8, 9]
    
    # Mock DQNs (not needed for clear winner)
    dqns = [MockDQN({}) for _ in range(committee_size)]
    
    selected, votes, is_tie = majority_vote(proposals, None, dqns, action_space)
    print(f"Proposals: {proposals}")
    print(f"Vote counts: {votes}")
    print(f"Selected action: {selected}")
    print(f"Is tie: {is_tie}")
    
    confidence = get_vote_confidence(votes, committee_size)
    entropy = get_vote_entropy(votes, committee_size)
    print(f"Confidence: {confidence:.2f}")
    print(f"Entropy: {entropy:.2f}")
    
    # Test case 2: Tie-breaking
    print("\n=== Test Case 2: Tie-Breaking ===")
    proposals = [7, 7, 8, 8, 6]  # Tie between 7 and 8
    
    # Mock DQNs with Q-values for tie-breaking
    dqns = [
        MockDQN({7: 0.5, 8: 0.6}),
        MockDQN({7: 0.4, 8: 0.7}),
        MockDQN({7: 0.6, 8: 0.5}),
        MockDQN({7: 0.3, 8: 0.8}),
        MockDQN({7: 0.5, 8: 0.6})
    ]
    
    selected, votes, is_tie = majority_vote(proposals, None, dqns, action_space)
    print(f"Proposals: {proposals}")
    print(f"Vote counts: {votes}")
    print(f"Selected action: {selected}")
    print(f"Is tie: {is_tie}")
    
    confidence = get_vote_confidence(votes, committee_size)
    entropy = get_vote_entropy(votes, committee_size)
    print(f"Confidence: {confidence:.2f}")
    print(f"Entropy: {entropy:.2f}")
    
    # Test case 3: Three-way tie
    print("\n=== Test Case 3: Three-Way Tie ===")
    proposals = [7, 7, 8, 8, 9]  # Multiple ties
    
    selected, votes, is_tie = majority_vote(proposals, None, dqns, action_space)
    print(f"Proposals: {proposals}")
    print(f"Vote counts: {votes}")
    print(f"Selected action: {selected}")
    print(f"Is tie: {is_tie}")
    
    print("\n✅ Majority Voting test completed!")
