import asyncio
from typing import List

from src.config import Config
from src.utils.logger import get_logger
from src.adaptive_play.environment import reward as env_reward
from src.llm_integration.vector_text_pipeline import VectorTextPipeline
from src.llm_integration.anomaly_detection import detect_anomalies
from src.sim.pybullet_env import PyBulletEnv
from src.sim.robot_controller import RobotController

log = get_logger(__name__)


class AdaptivePlay:
    """MAS loop with ε-greedy exploration, reward learning, LLM translation,
    anomaly detection, and optional PyBullet visual layer."""

    def __init__(self, agents: List):
        self.agents = agents
        self.history: list[dict[str, dict[str, float]]] = []
        self.anomalies_seen: list[str] = []
        self.vtp = VectorTextPipeline()

        # PyBullet layer (GUI only if Config.PBULLET_GUI)
        self.env = PyBulletEnv(gui=Config.PBULLET_GUI)
        self.ctrl = {
            ag.agent_id: RobotController(self.env, ag.agent_id) for ag in agents
        }

    # ───────────────────────────────────────────────
    async def _step_llm_exchange(self, step: int):
        # numeric → English
        text = {
            ag.agent_id: await self.vtp.vector_to_text(ag.agent_id, ag.probabilities)
            for ag in self.agents
        }
        for aid, sent in text.items():
            log.info(f"[{aid}] {sent}")

        # English → numeric
        parsed = {aid: await self.vtp.text_to_vector(s) for aid, s in text.items()}
        for ag in self.agents:
            ag.update_strategy_from_vector(parsed, step)

        self.history.append(parsed)

    # ───────────────────────────────────────────────
    async def run(self, steps: int):
        for t in range(steps):
            # 1. LLM round on interval
            if t % Config.EXPLANATION_INTERVAL == 0:
                await self._step_llm_exchange(t)
            else:
                self.history.append({ag.agent_id: {} for ag in self.agents})

            # 2. each agent chooses an action (ε-greedy)
            actions = {ag.agent_id: ag.choose_action(t) for ag in self.agents}

            # 3. compute rewards & learn
            for ag in self.agents:
                neigh_actions = [
                    actions[nid] for nid in actions if nid != ag.agent_id
                ]
                r = env_reward(actions[ag.agent_id], neigh_actions)
                ag.learn(actions[ag.agent_id], r, t)

            # 4. anomaly detection
            for aid, note in detect_anomalies(self.agents).items():
                line = f"{aid}: {note}"
                if line not in self.anomalies_seen:
                    self.anomalies_seen.append(line)
                    log.warning(f"ANOMALY {line}")

            # 5. update visual layer & step physics
            for ag in self.agents:
                self.ctrl[ag.agent_id].update(ag.probabilities)
            self.env.step()
