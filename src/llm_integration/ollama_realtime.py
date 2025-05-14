import aiohttp, asyncio, json
from src.config import Config
from src.utils.logger import get_logger
log = get_logger(__name__)

class OllamaRealtime:
    def __init__(self):
        self.url   = f"http://localhost:11434/api/generate"
        self.model = Config.LOCAL_LLM_MODEL
        self.sema  = asyncio.Semaphore(Config.MAX_PARALLEL_LLM)

    async def _generate(self, prompt):
        headers = {"Content-Type": "application/json"}
        payload = {"model": self.model, "prompt": prompt, "stream": True}
        async with self.sema, aiohttp.ClientSession() as sess:
            async with sess.post(self.url, json=payload, headers=headers) as resp:
                parts = []
                async for raw in resp.content:
                    try:
                        parts.append(json.loads(raw.decode()).get("response", ""))
                    except json.JSONDecodeError:
                        continue
                return "".join(parts).strip()

    async def explain(self, aid, probs, step):
        prompt = (
            f"Agent {aid} distribution at step {step}: {json.dumps(probs)}\n"
            "Explain briefly (<=2 sentences)."
        )
        txt = await self._generate(prompt)
        return f"[{aid}] {txt}"