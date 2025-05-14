from __future__ import annotations
import numpy as np
from numpy.random import default_rng
_rng = default_rng()

class Agent:
    def __init__(self, agent_id: str, actions=None):
        self.agent_id = agent_id
        self.actions  = actions or ["A", "B"]
        p = 1/len(self.actions)
        self.probabilities = {a: p for a in self.actions}
        self._prev_prob    = self.probabilities.copy()

    def choose_action(self):
        actions, probs = zip(*self.probabilities.items())
        return _rng.choice(actions, p=probs)

    def update_strategy(self, global_actions):
        self._prev_prob = self.probabilities.copy()
        counts = {a: 0 for a in self.actions}
        for a in global_actions.values():
            counts[a] += 1
        best = max(counts, key=counts.get)
        alpha = 0.1
        for a in self.actions:
            if a == best:
                self.probabilities[a] = min(self.probabilities[a] + alpha, 1)
            else:
                self.probabilities[a] = max(self.probabilities[a] - alpha/(len(self.actions)-1), 0)
        s = sum(self.probabilities.values())
        for a in self.actions:
            self.probabilities[a] /= s

    def delta_max(self):
        return max(abs(self.probabilities[a]-self._prev_prob[a]) for a in self.actions)
