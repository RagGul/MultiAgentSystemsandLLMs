import numpy as np
import pybullet as p
from src.sim.pybullet_env import PyBulletEnv

class RobotController:
    """Colour encodes prob(A); forward velocity encodes prob(B)."""
    def __init__(self, env: PyBulletEnv, agent_id: str):
        self.agent_id = agent_id
        self.env = env
        xy = np.random.uniform(-2, 2, size=2)
        env.spawn_robot(agent_id, pos=(xy[0], xy[1], 0.2))

    def update(self, probs: dict[str, float]):
        pA = probs.get("A", 0.5)
        rgba = [1 - pA, 0.2, pA, 1.0]             # red ↔ blue
        self.env.set_robot_colour(self.agent_id, rgba)

        pB = probs.get("B", 0.5)
        bid = self.env.robots[self.agent_id]
        lin = [0, pB * 0.3, 0]
        p.resetBaseVelocity(bid, linearVelocity=lin, angularVelocity=[0, 0, 0])
