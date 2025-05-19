class Config:
    # ---------- MAS size & runtime ----------
    NUM_AGENTS           = 6
    NUM_ITERATIONS       = 80
    EXPLANATION_INTERVAL = 1       # still real‑time enough but faster demo

    # ---------- LLM ----------
    LOCAL_LLM_MODEL      = "llama2:7b"   # or "llama2:7b-q2_k" for speed
    MAX_PARALLEL_LLM     = 4            # safe on 16‑GB M‑series

    # ---------- Exploration (ε‑greedy) ----------
    EPS_START        = 1.0
    EPS_DECAY        = 0.97
    EPS_MIN          = 0.05
    EPSILON_MIN  = EPS_MIN 

    # ---------- Learning‑rate decay ----------
    ALPHA_START          = 0.40
    ALPHA_DECAY          = 0.95
    ALPHA_MIN            = 0.05

    # ---------- Anomaly detection ----------
    PROB_JUMP_THRESH     = 0.25

    # ---------- Visual / physics ----------
    SIM_DT               = 1/120
    GUI                  = False

    # --- PyBullet visual layer ---
    PBULLET_GUI  = False          # Streamlit run should stay headless
    # (visualizer script always requests gui=True)