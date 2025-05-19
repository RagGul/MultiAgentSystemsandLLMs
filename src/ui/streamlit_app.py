import streamlit as st
import asyncio
from src.config import Config
from src.adaptive_play.agent import Agent
from src.adaptive_play.adaptive_play_algorithm import AdaptivePlay
from src.llm_integration.query_interface import QueryInterface

# ------------------------------------------------------------------
# Run MAS and stream live log (no graph)
# ------------------------------------------------------------------
def _run_sim_live():
    agents = [Agent(f"Agent_{i}") for i in range(Config.NUM_AGENTS)]
    play   = AdaptivePlay(agents)

    bar  = st.progress(0.0)
    log  = st.empty()

    # set up a fresh event loop for our async calls
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for step in range(Config.NUM_ITERATIONS):
        active = agents[step % len(agents)].agent_id
        # advance one iteration
        loop.run_until_complete(play.run(1))

        # grab the latest distribution for this agent
        dist = play.history[-1].get(active, {}) if play.history else {}

        # every EXPLANATION_INTERVAL steps, ask the LLM to turn it into human-friendly text
        if dist and step % Config.EXPLANATION_INTERVAL == 0:
            qi = QueryInterface(play.history)
            explanation = loop.run_until_complete(
                qi.answer(
                    f"Step {step}: Agent {active} has distribution {dist}. "
                    "Generate a human-friendly INFO log entry describing these probabilities."
                )
            )
            log.markdown(f"[INFO] [{active}] {explanation}")
        else:
            # show raw dict for debugging when not explaining
            log.markdown(f"[DEBUG] [{active}] {dist}")

        # update progress bar
        bar.progress((step + 1) / Config.NUM_ITERATIONS)

    loop.close()
    return agents, play

# ------------------------------------------------------------------
# Streamlit main
# ------------------------------------------------------------------
def app():
    st.set_page_config(page_title="MAS + LLM", layout="centered")
    st.title("THESIS PROJECT")

    # initialize our session state
    if "done" not in st.session_state:
        st.session_state.done = False

    # first view: run the sim
    if not st.session_state.done:
        if st.button("Run Simulation", type="primary"):
            agents, play = _run_sim_live()
            st.session_state["done"]    = True
            st.session_state["agents"]  = agents
            st.session_state["history"] = play.history
            st.session_state["anoms"]   = play.anomalies_seen
        else:
            st.info("Click the button to start. Tweak parameters as needed.")

    # after running: show results
    if st.session_state.done:
        st.success("Simulation complete")
        st.subheader("Final probabilities")
        for ag in st.session_state.agents:
            st.write(f"{ag.agent_id}: {ag.probabilities}")

        st.subheader("Anomalies")
        if st.session_state.anoms:
            for note in st.session_state.anoms:
                st.warning(note)
        else:
            st.info("None detected")

        st.markdown("---")
        q = st.text_input("Ask about the run")
        if st.button("Ask") and q:
            qi   = QueryInterface(st.session_state.history)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            ans  = loop.run_until_complete(qi.answer(q))
            loop.close()
            st.info(ans)

        # allow one more full re-run
        if st.button("Run again"):
            for k in ("done", "agents", "history", "anoms"):
                st.session_state.pop(k, None)

