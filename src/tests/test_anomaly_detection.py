from src.llm_integration.anomaly_detection import detect_anomalies
from src.adaptive_play.agent import Agent

def test_detector():
    ag = Agent("X")
    ag.probabilities = {"A":1.0, "B":0.0}
    ag._prev_prob     = {"A":0.0, "B":1.0}
    anomalies = detect_anomalies([ag])
    assert "X" in anomalies