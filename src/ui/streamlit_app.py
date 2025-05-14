import streamlit as st, asyncio
from src.config import Config
from src.adaptive_play.agent import Agent
from src.adaptive_play.adaptive_play_algorithm import AdaptivePlay
from src.llm_integration.query_interface import QueryInterface

def run_sim_with_bar():
    """Synchronous wrapper so we don't nest asyncio.run inside Streamlit."""
    agents = [Agent(f"Agent_{i}") for i in range(Config.NUM_AGENTS)]
    play   = AdaptivePlay(agents)

    bar = st.progress(0.0)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for step in range(Config.NUM_ITERATIONS):
        loop.run_until_complete(play.run(1))   # one iteration at a time
        bar.progress((step + 1) / Config.NUM_ITERATIONS)
    loop.close()
    return agents, play


def app():
    st.title("MAS Adaptive‑Play + LLM Interpretability")

    if "run_done" not in st.session_state:
        st.session_state["run_done"] = False

    if not st.session_state.run_done:
        if st.button("Run Simulation"):
            agents, play = run_sim_with_bar()
            st.session_state["agents"] = agents
            st.session_state["history"] = play.history
            st.session_state["anoms"] = play.anomalies_seen
            st.session_state.run_done = True
            st.experimental_rerun()
    else:
        st.success("Simulation finished!")
        st.subheader("Final probabilities (first 10 agents)")
        for ag in st.session_state["agents"][:10]:
            st.write(f"{ag.agent_id} → {ag.probabilities}")

        # show anomalies if any
        st.subheader("Anomalies detected during run")
        if st.session_state["anoms"]:
            for note in st.session_state["anoms"]:
                st.warning(note)
        else:
            st.info("No anomalies exceeded the threshold.")

        # interactive query box
        st.markdown("---")
        st.subheader("Ask a question about recent behaviour")
        question = st.text_input("Your question")
        if st.button("Ask") and question:
            qi = QueryInterface(st.session_state["history"])
            # if Streamlit is already in an async loop use ensure_future
            loop = asyncio.get_event_loop()
            answer = loop.run_until_complete(qi.answer(question)) if not loop.is_running() else asyncio.ensure_future(qi.answer(question))
            st.info(answer)

        # reset button for a fresh run
        if st.button("Reset and run again"):
            for key in ("run_done", "agents", "history", "anoms"):
                if key in st.session_state:
                    del st.session_state[key]
            st.experimental_rerun()