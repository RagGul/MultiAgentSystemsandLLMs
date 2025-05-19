"""
Run PyBullet GUI in its own process so Streamlit (or Jupyter) can stay alive.
"""

from src.config import Config
from src.sim.pybullet_env import PyBulletEnv
from src.sim.robot_controller import RobotController
from src.adaptive_play.agent import Agent
from src.adaptive_play.adaptive_play_algorithm import AdaptivePlay
import asyncio, numpy as np

async def main():
    agents = [Agent(f"Agent_{i}") for i in range(Config.NUM_AGENTS)]
    env    = PyBulletEnv(gui=True)
    ctrl   = {ag.agent_id: RobotController(env, ag.agent_id) for ag in agents}
    play   = AdaptivePlay(agents)

    for step in range(Config.NUM_ITERATIONS):
        await play._step_llm_exchange(step)           # LLM every step here
        for ag in agents:
            ctrl[ag.agent_id].update(ag.probabilities)
        env.step()

if __name__ == "__main__":
    asyncio.run(main())
