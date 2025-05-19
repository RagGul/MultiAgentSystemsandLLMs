import asyncio, json
from src.llm_integration.ollama_realtime import OllamaRealtime

class QueryInterface:
    """Ask LLM questions about recent MAS history."""

    def __init__(self, history):
        self.history = history
        self.llm = OllamaRealtime()

    async def answer(self, question: str) -> str:
        ctx = json.dumps(self.history[-5:], indent=2)
        prompt = (
            "Given this recent MAS history:\n" + ctx +
            "\nQuestion: " + question + "\nAnswer succinctly."
        )
        return await self.llm._generate(prompt)
