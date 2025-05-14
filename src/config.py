class Config:
    # ----- MAS size & timing -----
    NUM_AGENTS          = 20
    NUM_ITERATIONS      = 60
    EXPLANATION_INTERVAL = 1        # every step → real‑time

    # ----- LLM / Ollama -----
    LOCAL_LLM_MODEL     = "llama2:7b"  # default full‑size model  # small & fast (ollama pull phi:2.7b)
    MAX_PARALLEL_LLM    = 4           # safer concurrency for 7‑B model on 16‑GB RAM           # safe on 16‑GB RAM

    # ----- Anomaly detection -----
    PROB_JUMP_THRESH    = 0.25        # flag if |Δp| > 0.25

    # ----- Physics (unused in toy) -----
    SIM_DT              = 1/120
    GUI                 = False