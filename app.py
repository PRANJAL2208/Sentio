"""
app.py — Sentio

Run with: streamlit run app.py
"""

import os
import uuid
import time

import streamlit as st
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

from agent.graph import build_graph, run_agent
from memory.store import (
    get_chroma_client,
    get_or_create_collection,
    store_topic,
    extract_topics_from_response,
    update_topic_on_review,
    get_forgotten_topics,
)
from ui.timing import inject_timing_tracker, get_typing_telemetry, update_clarification_count

load_dotenv()


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sentio",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 Sentio")
st.caption("An AI tutor that adapts to your cognitive state in real time.")


# ── Init session state ────────────────────────────────────────────────────────

def init_session():
    defaults = {
        "session_id":            str(uuid.uuid4()),
        "chat_history":          [],        # [{role, content}]
        "clarification_window":  [],        # last 3 turns, bool
        "telemetry_input":       "",        # populated by JS telemetry collector
        "_last_interaction_time": time.time(),
        "load_state_history":    [],        # for the sidebar debug panel
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()


# ── Inject JS timing tracker & Telemetry Bridge ────────────────────────────────

# Hidden text input to sync telemetry JSON from JS component
st.text_input("telemetry_data", label_visibility="collapsed", key="telemetry_input")

# Hide the telemetry text input via CSS
st.markdown("""
<style>
div[data-testid="stTextInput"]:has(input[aria-label="telemetry_data"]) {
    display: none;
}
</style>
""", unsafe_allow_html=True)

inject_timing_tracker()


# ── Setup Memory (needs to be initialized early for sidebar rendering) ────────

@st.cache_resource
def get_memory(user_id: str = "default"):
    client = get_chroma_client()
    return get_or_create_collection(client, user_id=user_id)

memory_collection = get_memory()


# ── Model configuration utilities ─────────────────────────────────────────────

@st.cache_resource
def get_llm_instance(provider: str, model_name: str, api_key: str):
    if provider == "Gemini (Google)":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.7,
        )
    elif provider == "OpenAI":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0.7,
        )
    elif provider == "Anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name,
            anthropic_api_key=api_key,
            temperature=0.7,
        )
    elif provider == "Groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model_name,
            api_key=api_key,
            temperature=0.7,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

@st.cache_resource
def get_agent(_llm):
    return build_graph(llm=_llm)

@st.cache_resource
def get_extractor_llm(provider: str, model_name: str, api_key: str):
    if provider == "Gemini (Google)":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.0,
        )
    elif provider == "OpenAI":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0.0,
        )
    elif provider == "Anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name,
            anthropic_api_key=api_key,
            temperature=0.0,
        )
    elif provider == "Groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model_name,
            api_key=api_key,
            temperature=0.0,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ── Sidebar: LLM Configuration & Monitor ─────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ LLM Settings")
    provider = st.selectbox(
        "Provider",
        ["Gemini (Google)", "OpenAI", "Anthropic", "Groq"],
        index=0,
    )
    
    default_models = {
        "Gemini (Google)": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        "OpenAI": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        "Anthropic": ["claude-3-5-sonnet-latest", "claude-3-haiku-20240307"],
        "Groq": ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    }
    
    model_name = st.selectbox(
        "Model",
        default_models[provider],
        index=0,
    )
    
    custom_model = st.text_input("Custom model name (optional)")
    if custom_model.strip():
        model_name = custom_model.strip()
        
    api_key_input = st.text_input(
        "API Key (Leave blank to use .env)",
        type="password",
    )

    resolved_api_key = api_key_input.strip()
    if not resolved_api_key:
        env_keys = {
            "Gemini (Google)": "GEMINI_API_KEY",
            "OpenAI": "OPENAI_API_KEY",
            "Anthropic": "ANTHROPIC_API_KEY",
            "Groq": "GROQ_API_KEY",
        }
        resolved_api_key = os.getenv(env_keys[provider], "")

    st.divider()

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
            if signals.get("avg_flight_ms", 0.0) > 0:
                st.markdown(f"- Avg Flight: `{signals['avg_flight_ms']:.0f}ms`")
            if signals.get("avg_dwell_ms", 0.0) > 0:
                st.markdown(f"- Avg Dwell: `{signals['avg_dwell_ms']:.0f}ms`")
            if signals.get("backspace_count", 0) > 0:
                st.markdown(f"- Backspaces: `{signals['backspace_count']}`")

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
    st.markdown("### ⚙️ Session Settings")
    show_load_badge = st.toggle("Show load state in chat", value=True)

    if st.button("Clear session memory"):
        st.session_state["chat_history"] = []
        st.session_state["clarification_window"] = []
        st.session_state["load_state_history"] = []
        st.rerun()


# ── Validate credentials & compile agent ─────────────────────────────────────

if not resolved_api_key:
    st.error(f"API Key for {provider} not found. Please paste it in the sidebar or add it to your `.env` file.")
    st.stop()

llm = get_llm_instance(provider, model_name, resolved_api_key)
compiled_graph = get_agent(llm)
extractor_llm = get_extractor_llm(provider, model_name, resolved_api_key)


# ── Render existing chat history ──────────────────────────────────────────────

for turn in st.session_state["chat_history"]:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if show_load_badge and turn["role"] == "assistant" and "load_state" in turn:
            badge = {"OVERLOADED": "🔴 simplified", "OPTIMAL": "🟢 normal", "UNDERLOADED": "🔵 enriched"}
            st.caption(badge.get(turn["load_state"], ""))


# ── Chat input ────────────────────────────────────────────────────────────────

if user_input := st.chat_input("Ask me anything..."):

    # 1. Get typing telemetry (JS primary, fallback secondary)
    telemetry = get_typing_telemetry(st.session_state)

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
                pause_seconds=telemetry["pause_seconds"],
                chat_history=st.session_state["chat_history"],
                memory_collection=memory_collection,
                recent_clarification_count=clarification_count,
                avg_dwell_ms=telemetry["avg_dwell_ms"],
                avg_flight_ms=telemetry["avg_flight_ms"],
                backspace_count=telemetry["backspace_count"],
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
        topics = extract_topics_from_response(extractor_llm, result["response"])
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
        "Sentio will detect when you're overwhelmed or bored and adapt automatically. "
        "Watch the **Load Monitor** in the sidebar."
    )
