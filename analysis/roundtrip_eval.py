"""Compute MAE / MSE / KL / Corr for vector→text→vector round‑trip"""
import asyncio, json, numpy as np
from numpy.random import default_rng
from src.llm_integration.vector_text_pipeline import VectorTextPipeline

actions = ["A", "B"]
_rng = default_rng(0)

async def run(n=500):
    vtp = VectorTextPipeline()
    maes = []; mses = []; kls = []; cors = []
    for _ in range(n):
        vec = _rng.dirichlet(np.ones(len(actions))).tolist()
        vec_dict = dict(zip(actions, vec))
        txt  = await vtp.vector_to_text("T", vec_dict)
        back = await vtp.text_to_vector(txt)
        a = np.array(vec)
        b = np.array([back[a] for a in actions])
        maes.append(np.mean(np.abs(a-b)))
        mses.append(np.mean((a-b)**2))
        kls.append(np.sum(a * np.log(a/(b+1e-12))))
        cors.append(np.corrcoef(a,b)[0,1])
    print("MAE", np.mean(maes), "±", np.std(maes))
    print("MSE", np.mean(mses), "±", np.std(mses))
    print("KL ", np.mean(kls),  "±", np.std(kls))
    print("Corr", np.mean(cors))

if __name__ == "__main__":
    asyncio.run(run())