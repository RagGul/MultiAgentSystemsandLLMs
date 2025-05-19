import aiohttp, asyncio, json, async_timeout
from src.config import Config

class OllamaRealtime:
    """Thin async wrapper around your local Ollama server."""

    def __init__(self):
        self.url   = "http://localhost:11434/api/generate"
        self.model = Config.LOCAL_LLM_MODEL
        self.sema  = asyncio.Semaphore(Config.MAX_PARALLEL_LLM)

    async def _generate(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        payload = {"model": self.model, "prompt": prompt, "stream": True}

        async with self.sema, aiohttp.ClientSession() as sess:
            try:
                async with async_timeout.timeout(180):          # overall cap
                    async with sess.post(
                        self.url,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=60)  # idle cap
                    ) as resp:
                        parts = []
                        async for raw in resp.content:
                            try:
                                parts.append(json.loads(raw.decode()).get("response", ""))
                            except json.JSONDecodeError:
                                continue
                        return "".join(parts).strip()
            except (asyncio.TimeoutError, aiohttp.ClientError):
                return ""          # let caller ignore on timeout
