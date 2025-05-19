# src/adaptive_play/environment.py

from collections import Counter

# Payoff matrix for the Stag-Hunt game:
# (your action, majority neighbour action) → reward
PAYOFF = {
    ("A", "A"): 3,
    ("A", "B"): 0,
    ("B", "A"): 0,
    ("B", "B"): 2,
}


def majority(neigh_actions: list[str]) -> str:
    """Return the most common action; ties go to 'A'."""
    counts = Counter(neigh_actions)
    return "A" if counts["A"] >= counts["B"] else "B"


def reward(my_action: str, neighbour_actions: list[str]) -> float:
    """Compute the reward for my_action given the neighbours’ actions."""
    opp = majority(neighbour_actions)
    return PAYOFF[(my_action, opp)]
