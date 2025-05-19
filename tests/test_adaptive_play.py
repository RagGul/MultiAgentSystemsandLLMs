import pytest, asyncio
from src.adaptive_play.agent import Agent
from src.adaptive_play.adaptive_play_algorithm import AdaptivePlay

@pytest.mark.asyncio
async def test_probabilities_sum_to_one():
    agents = [Agent("T0"), Agent("T1")]
    play = AdaptivePlay(agents)
    await play.run(3)
    for ag in agents:
        assert abs(sum(ag.probabilities.values()) - 1.0) < 1e-6
