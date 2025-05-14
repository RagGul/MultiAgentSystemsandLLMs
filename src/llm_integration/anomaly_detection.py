from src.config import Config

def detect_anomalies(agents):
    out = {}
    for ag in agents:
        if ag.delta_max() >= Config.PROB_JUMP_THRESH:
            out[ag.agent_id] = f"Δp>{Config.PROB_JUMP_THRESH:.2f}"
    return out
