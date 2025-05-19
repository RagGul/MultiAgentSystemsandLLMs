from src.config import Config

def detect_anomalies(agents):
    """Return {agent_id: note} for jumps ≥ threshold."""
    out = {}
    thresh = Config.PROB_JUMP_THRESH
    for ag in agents:
        if ag.delta_max() >= thresh:
            out[ag.agent_id] = f"Δp>{thresh:.2f}"
    return out
