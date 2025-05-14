import asyncio
from typing import List
from tqdm import tqdm
from src.config import Config
from src.utils.logger import get_logger
from .agent import Agent
from src.llm_integration.ollama_realtime import OllamaRealtime
from src.llm_integration.anomaly_detection import detect_anomalies

log = get_logger(__name__)

class AdaptivePlay:
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.history = []
        self.anomalies_seen: list[str] = []
        self.llm = OllamaRealtime()
        self.pending: set[asyncio.Task] = set()

    async def _explain_agents(self, step: int):
        for ag in self.agents:
            self.pending.add(asyncio.create_task(
                self.llm.explain(ag.agent_id, ag.probabilities, step)
            ))
        if self.pending:
            done, self.pending = await asyncio.wait(self.pending, timeout=0)
            for d in done:
                log.info(d.result())

    async def run(self, steps: int):
        for t in tqdm(range(steps), desc="Simulation", ncols=70):
            actions = {ag.agent_id: ag.choose_action() for ag in self.agents}
            for ag in self.agents:
                ag.update_strategy(actions)
            self.history.append({ag.agent_id: ag.probabilities.copy() for ag in self.agents})
            anns = detect_anomalies(self.agents)
            for aid, note in anns.items():
                self.anomalies_seen.append(f"{aid}: {note}")
                log.warning(f"ANOMALY {aid}: {note}")
            if t % Config.EXPLANATION_INTERVAL == 0:
                await self._explain_agents(t)
        if self.pending:
            await asyncio.gather(*self.pending)