import numpy as np
from src.config import Config


class Agent:
    """One MAS agent with ε-greedy action selection and TD-style learning."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.actions = ["A", "B"]

        # random non-uniform start so learning dynamics are visible
        rnd = np.random.rand(2)
        rnd /= rnd.sum()
        self.probabilities = dict(zip(self.actions, rnd))
        self._prev_prob = self.probabilities.copy()

    # ───────────────────────────────────────────────
    def renorm(self):
        s = sum(self.probabilities.values())
        for a in self.actions:
            self.probabilities[a] /= s

    # ───────────────────────────────────────────────
    def choose_action(self, step: int) -> str:
        eps = max(Config.EPS_MIN, Config.EPS_START * (Config.EPS_DECAY ** step))
        if np.random.rand() < eps:
            return np.random.choice(self.actions)
        return max(self.probabilities, key=self.probabilities.get)

    # ───────────────────────────────────────────────
    def learn(self, action: str, reward: float, step: int):
        """TD-like update: move prob mass toward the chosen action."""
        alpha = max(
            Config.ALPHA_MIN, Config.ALPHA_START * (Config.ALPHA_DECAY ** step)
        )
        for a in self.actions:
            target = 1.0 if a == action else 0.0
            self.probabilities[a] += alpha * (target - self.probabilities[a])
        self.renorm()

    # ───────────────────────────────────────────────
    def update_strategy_from_vector(self, neigh: dict[str, dict[str, float]], step: int):
        """Blend neighbour vectors (after LLM parse)."""
        alpha = max(
            Config.ALPHA_MIN, Config.ALPHA_START * (Config.ALPHA_DECAY ** step)
        )
        agg, valid = {a: 0.0 for a in self.actions}, 0
        for probs in neigh.values():
            if not isinstance(probs, dict):
                continue
            if probs and all(a in probs for a in self.actions):
                valid += 1
                for a in self.actions:
                    agg[a] += probs[a]
        if not valid:
            return
        for a in agg:
            agg[a] /= valid

        self._prev_prob = self.probabilities.copy()
        for a in self.actions:
            self.probabilities[a] = (1 - alpha) * self.probabilities[a] + alpha * agg[a]
        self.renorm()

    def delta_max(self) -> float:
        return max(
            abs(self.probabilities[a] - self._prev_prob[a]) for a in self.actions
        )
