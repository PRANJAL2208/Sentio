"""
app.py — CogniFlow

Run with: streamlit run app.py
"""

import os
import uuid
import time

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI as OpenAIClient

from agent.graph import build_graph, run_agent
from memory.store import (
    get_chroma_client,
    get_or_create_collection,
    store_topic,
    extract_topics_from_response,
    update_topic_on_review,
    get_forgotten_topics,
)
from ui.timing import inject_timing_tracker, get_pause_seconds, update_clarification_count

load_dotenv()


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CogniFlow",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 CogniFlow")
st.caption("An AI tutor that adapts to your cognitive state in real time.")


# ── Init session state ────────────────────────────────────────────────────────

def init_session():
    defaults = {
        "session_id":            str(uuid.uuid4()),
        "chat_history":          [],        # [{role, content}]
        "clarification_window":  [],        # last 3 turns, bool
        "js_pause_seconds":      None,      # populated by JS component
        "_last_interaction_time": time.time(),
        "load_state_history":    [],        # for the sidebar debug panel
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()


# ── Inject JS timing tracker ──────────────────────────────────────────────────
inject_timing_tracker()


# ── Setup clients (cached so they don't rebuild on every rerun) ───────────────

# @st.cache_resource
# def get_agent():
#     api_key = os.getenv("OPENAI_API_KEY", "")
#     if not api_key:
#         st.error("OPENAI_API_KEY not found. Add it to .env file.")
#         st.stop()
#     return build_graph(openai_api_key=api_key, model="gpt-4o-mini")


# And update get_agent():
@st.cache_resource
def get_agent():
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        st.error("GEMINI_API_KEY not found. Add it to .env file.")
        st.stop()
    return build_graph(gemini_api_key=api_key)

@st.cache_resource
def get_memory(user_id: str = "default"):
    client = get_chroma_client()
    return get_or_create_collection(client, user_id=user_id)

# @st.cache_resource
# def get_openai_client():
#     return OpenAIClient(api_key=os.getenv("OPENAI_API_KEY", ""))


@st.cache_resource
def get_openai_client():
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0,
    )

compiled_graph   = get_agent()
memory_collection = get_memory()
openai_client    = get_openai_client()

# ── Sidebar: debug panel ──────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🔬 Cognitive Load Monitor")
    st.caption("What the system detected for your last message.")

    if st.session_state["load_state_history"]:
        last = st.session_state["load_state_history"][-1]

        state_colors = {
            "OVERLOADED":   "🔴",
            "OPTIMAL":      "🟢",
            "UNDERLOADED":  "🔵",
        }
        emoji = state_colors.get(last["state"], "⚪")
        st.markdown(f"**State:** {emoji} {last['state']}")
        st.markdown(f"**Confidence:** {last['confidence']:.0%}")

        with st.expander("Raw signals"):
            signals = last["signals"]
            st.markdown(f"- Pause: `{signals['pause_seconds']:.1f}s`")
            st.markdown(f"- Message length: `{signals['message_length']} chars`")
            st.markdown(f"- Clarification: `{signals['is_clarification']}`")
            st.markdown(f"- Recent clarifications: `{signals['recent_clarification_count']}`")

    st.divider()
    st.markdown("### 🗃️ Memory")
    st.caption("Topics the system thinks you might have forgotten.")

    forgotten = get_forgotten_topics(memory_collection, retention_threshold=0.7)
    if forgotten:
        for item in forgotten:
            pct = int(item["retention"] * 100)
            st.markdown(f"- {item['summary'][:60]}… *(~{pct}% retained)*")
    else:
        st.caption("Nothing flagged for review yet.")

    st.divider()
    st.markdown("### ⚙️ Settings")
    show_load_badge = st.toggle("Show load state in chat", value=True)

    if st.button("Clear session memory"):
        st.session_state["chat_history"] = []
        st.session_state["clarification_window"] = []
        st.session_state["load_state_history"] = []
        st.rerun()


# ── Render existing chat history ──────────────────────────────────────────────

for turn in st.session_state["chat_history"]:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if show_load_badge and turn["role"] == "assistant" and "load_state" in turn:
            badge = {"OVERLOADED": "🔴 simplified", "OPTIMAL": "🟢 normal", "UNDERLOADED": "🔵 enriched"}
            st.caption(badge.get(turn["load_state"], ""))


# ── Chat input ────────────────────────────────────────────────────────────────

if user_input := st.chat_input("Ask me anything..."):

    # 1. Get pause duration (JS primary, fallback secondary)
    pause = get_pause_seconds(st.session_state)

    # 2. Count recent clarifications
    clarification_count = update_clarification_count(st.session_state, user_input)

    # 3. Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # 4. Run the agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = run_agent(
                compiled_graph=compiled_graph,
                user_message=user_input,
                pause_seconds=pause,
                chat_history=st.session_state["chat_history"],
                memory_collection=memory_collection,
                recent_clarification_count=clarification_count,
            )

        st.markdown(result["response"])

        if show_load_badge:
            badge = {
                "OVERLOADED":  "🔴 simplified response",
                "OPTIMAL":     "🟢 normal response",
                "UNDERLOADED": "🔵 enriched response",
            }
            st.caption(badge.get(result["load_state"], ""))

    # 5. Update session state
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    st.session_state["chat_history"].append({
        "role":       "assistant",
        "content":    result["response"],
        "load_state": result["load_state"],
    })
    st.session_state["load_state_history"].append({
        "state":      result["load_state"],
        "confidence": result["load_confidence"],
        "signals":    result["load_signals"],
    })

    # 6. Store topics in memory (background, non-blocking)
    try:
        topics = extract_topics_from_response(openai_client, result["response"])
        for topic in topics:
            store_topic(
                collection=memory_collection,
                topic_summary=topic,
                session_id=st.session_state["session_id"],
            )

        # If any forgotten topics were resurfaced, update their stability
        forgotten_now = get_forgotten_topics(memory_collection)
        for item in forgotten_now:
            update_topic_on_review(memory_collection, item["topic_id"])

    except Exception:
        pass  # memory is enhancement — never block the chat


# ── First-time welcome message ────────────────────────────────────────────────

if not st.session_state["chat_history"]:
    st.info(
        "👋 Start chatting about any topic you're learning. "
        "CogniFlow will detect when you're overwhelmed or bored and adapt automatically. "
        "Watch the **Load Monitor** in the sidebar."
    )
