# src/llm_integration/vector_text_pipeline.py

import asyncio, aiohttp, json, async_timeout
from src.config import Config

class VectorTextPipeline:
    """Translate between numeric policy vectors and natural-language summaries via Ollama."""

    def __init__(self):
        # point at Ollama’s generate endpoint and use your chosen model
        self.url = "http://localhost:11434/api/generate"
        self.model = Config.LOCAL_LLM_MODEL
        self.sema = asyncio.Semaphore(Config.MAX_PARALLEL_LLM)
        self.headers = {"Content-Type": "application/json"}

    async def _generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            # you can add "max_tokens" or "temperature" here if desired
        }
        async with self.sema, aiohttp.ClientSession() as sess:
            try:
                # overall timeout for the call
                async with async_timeout.timeout(60):
                    async with sess.post(self.url, json=payload, headers=self.headers) as resp:
                        data = await resp.json()
                        # Ollama returns {"response": "..."} for non-stream
                        return data.get("response", "").strip()
            except Exception:
                return ""  # on error, return empty so caller can handle

    async def vector_to_text(self, aid: str, probs: dict[str, float]) -> str:
        """Ask the LLM to turn a two-action distribution into a one- or two-sentence summary."""
        pA, pB = probs.get("A", 0.0), probs.get("B", 0.0)
        prompt = (
            f"You are an analyst summarizing agent behavior.\n"
            f"Agent {aid} has a probability distribution over two actions:\n"
            f"  • action A = {pA:.2f}\n"
            f"  • action B = {pB:.2f}\n\n"
            "In one or two clear sentences (prefixed with “[INFO]”), explain what this tells us "
            "about the agent’s bias and likely decision. For example:\n"
            "[INFO] At step 0, Agent_3’s policy heavily favors action A (1.0) over B (0.0), indicating it will choose A.\n"
            "Do not output JSON or bullet points—just the plain INFO line."
        )
        return await self._generate(prompt)

    async def text_to_vector(self, text: str) -> dict[str, float]:
        """Parse a back-translated JSON or fallback to regex extraction of A=... and B=...."""
        # first, try JSON
        try:
            vec = json.loads(text)
            if all(k in vec for k in ("A", "B")):
                return {"A": float(vec["A"]), "B": float(vec["B"])}
        except json.JSONDecodeError:
            pass

        # fallback: look for “A=0.75” and “B=0.25” in the text
        import re
        m = re.search(r"A\s*=\s*([0-9.]+).+B\s*=\s*([0-9.]+)", text)
        if m:
            return {"A": float(m.group(1)), "B": float(m.group(2))}
        return {}
