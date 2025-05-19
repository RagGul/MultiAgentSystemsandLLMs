"""Order‑bias and compression‑bias checks"""
import asyncio, numpy as np
from numpy.random import default_rng
from src.llm_integration.vector_text_pipeline import VectorTextPipeline

actions = ["A", "B"]
_rng = default_rng(1)

async def run(n=200):
    vtp = VectorTextPipeline()
    first_counts = {a:0 for a in actions}
    comps = []
    for _ in range(n):
        vec = _rng.dirichlet(np.ones(len(actions))).tolist()
        txt = await vtp.vector_to_text("T", dict(zip(actions, vec)))
        first = txt.split()[0].strip(".,")
        if first in first_counts: first_counts[first]+=1
        back = await vtp.text_to_vector(txt)
        gap_raw   = max(vec) - min(vec)
        gap_parsed= max(back.values()) - min(back.values())
        if gap_raw: comps.append(gap_parsed/gap_raw)
    print("Order bias counts", first_counts)
    print("Avg compression ratio", np.mean(comps))

if __name__ == "__main__":
    asyncio.run(run())