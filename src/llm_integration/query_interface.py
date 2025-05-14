from __future__ import annotations
import asyncio, json
from src.llm_integration.ollama_realtime import OllamaRealtime

class QueryInterface:
    def __init__(self, history_log: list[dict[str, dict[str,float]]]):
        self.history = history_log
        self.llm     = OllamaRealtime()

    async def answer(self, question: str) -> str:
        # include last 5 timesteps context
        ctx = self.history[-5:]
        ctx_txt = json.dumps(ctx, indent=2)
        prompt = (
            "Given the following recent probability updates in a multi‑agent system:\n" +
            ctx_txt + "\nQuestion: " + question +
            "\nAnswer briefly in plain English."
        )
        return await self.llm._generate(prompt)
