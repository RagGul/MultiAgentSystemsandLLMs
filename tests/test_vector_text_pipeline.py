import pytest, asyncio, json
from src.llm_integration.vector_text_pipeline import VectorTextPipeline

class EchoVTP(VectorTextPipeline):
    async def vector_to_text(self, aid, probs):
        return json.dumps(probs)
    async def text_to_vector(self, text):
        return json.loads(text)

@pytest.mark.asyncio
async def test_round_trip_echo():
    vtp = EchoVTP()
    vec = {"A":0.6,"B":0.4}
    txt = await vtp.vector_to_text("X", vec)
    back= await vtp.text_to_vector(txt)
    assert back == vec