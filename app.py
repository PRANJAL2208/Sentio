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
        "srs_quiz_active":       False,
        "srs_topic_id":          "",
        "turns_since_last_quiz": 4,
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


# ── Page Layout: Tabs ─────────────────────────────────────────────────────────

tab_tutor, tab_dashboard = st.tabs(["💬 Sentio Tutor", "📊 Learning Dashboard"])


# ── Render Tutor Interface ───────────────────────────────────────────────────

with tab_tutor:
    # Render existing chat history
    for turn in st.session_state["chat_history"]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if show_load_badge and turn["role"] == "assistant" and "load_state" in turn:
                badge = {"OVERLOADED": "🔴 simplified", "OPTIMAL": "🟢 normal", "UNDERLOADED": "🔵 enriched"}
                st.caption(badge.get(turn["load_state"], ""))
            if turn["role"] == "assistant" and turn.get("srs_evaluation"):
                grade_emojis = {"correct": "🎯 Correct!", "partial": "⚠️ Close!", "incorrect": "❌ Not quite."}
                st.info(f"**Spaced Repetition Practice**: {grade_emojis.get(turn['srs_evaluation'], '')}")

    if not st.session_state["chat_history"]:
        st.info(
            "👋 Start chatting about any topic you're learning. "
            "Sentio will detect when you're overwhelmed or bored and adapt automatically. "
            "Watch the **Load Monitor** in the sidebar."
        )


# ── Chat input ────────────────────────────────────────────────────────────────

if user_input := st.chat_input("Ask me anything..."):

    # 1. Get typing telemetry (JS primary, fallback secondary)
    telemetry = get_typing_telemetry(st.session_state)

    # 2. Count recent clarifications
    clarification_count = update_clarification_count(st.session_state, user_input)

    # 3. Show user message and query agent inside tutor tab
    with tab_tutor:
        with st.chat_message("user"):
            st.markdown(user_input)

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
                    srs_quiz_active=st.session_state["srs_quiz_active"],
                    srs_topic_id=st.session_state["srs_topic_id"],
                    turns_since_last_quiz=st.session_state["turns_since_last_quiz"],
                )

            st.markdown(result["response"])

            if show_load_badge:
                badge = {
                    "OVERLOADED":  "🔴 simplified response",
                    "OPTIMAL":     "🟢 normal response",
                    "UNDERLOADED": "🔵 enriched response",
                }
                st.caption(badge.get(result["load_state"], ""))

    # 4. Update session state
    st.session_state["srs_quiz_active"] = result["srs_quiz_active"]
    st.session_state["srs_topic_id"] = result["srs_topic_id"]
    st.session_state["turns_since_last_quiz"] = result["turns_since_last_quiz"]

    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    st.session_state["chat_history"].append({
        "role":       "assistant",
        "content":    result["response"],
        "load_state": result["load_state"],
        "srs_evaluation": result.get("srs_evaluation", ""),
    })
    st.session_state["load_state_history"].append({
        "state":      result["load_state"],
        "confidence": result["load_confidence"],
        "signals":    result["load_signals"],
    })

    # 5. Store topics in memory (background, non-blocking)
    if not result.get("srs_evaluation") and not result.get("srs_quiz_active"):
        try:
            topics = extract_topics_from_response(extractor_llm, result["response"])
            for topic in topics:
                store_topic(
                    collection=memory_collection,
                    topic_summary=topic["summary"],
                    session_id=st.session_state["session_id"],
                    links_to=topic["links_to"],
                )

            # If any forgotten topics were resurfaced, update their stability
            forgotten_now = get_forgotten_topics(memory_collection)
            for item in forgotten_now:
                update_topic_on_review(memory_collection, item["topic_id"])

        except Exception:
            pass  # memory is enhancement — never block the chat

    st.rerun()


# ── Render learning dashboard ───────────────────────────────────────────────

with tab_dashboard:
    st.markdown("## 📊 Learning Analytics Dashboard")
    st.caption("Track your biometrics, memory retention curves, and concept associations in real-time.")

    # 1. Telemetry trends
    if st.session_state["load_state_history"]:
        st.markdown("### ⌨️ Typing Telemetry Trends")
        
        steps = list(range(1, len(st.session_state["load_state_history"]) + 1))
        flight_history = []
        dwell_history = []
        backspaces_history = []
        pause_history = []

        for item in st.session_state["load_state_history"]:
            sig = item["signals"]
            flight_history.append(sig.get("avg_flight_ms", 0.0))
            dwell_history.append(sig.get("avg_dwell_ms", 0.0))
            backspaces_history.append(sig.get("backspace_count", 0))
            pause_history.append(sig.get("pause_seconds", 0.0))

        import pandas as pd
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Typing Latencies (ms)**")
            df_latencies = pd.DataFrame({
                "Flight Time (ms)": flight_history,
                "Dwell Time (ms)": dwell_history
            }, index=steps)
            st.line_chart(df_latencies)
        with col2:
            st.markdown("**User Hesitations & Editing**")
            df_editing = pd.DataFrame({
                "Backspace Count": backspaces_history,
                "Pause Time (s)": pause_history
            }, index=steps)
            st.line_chart(df_editing)
    else:
        st.info("No telemetry logs recorded yet. Start chatting to populate these trends!")

    # 2. Memory retention curves
    st.markdown("---")
    st.markdown("### 🧠 Memory Retention Curves")
    all_items = memory_collection.get(include=["documents", "metadatas"])
    
    if all_items["ids"]:
        from memory.store import retention_score
        topic_summaries = []
        retentions = []
        stabilities = []

        for doc, meta in zip(all_items["documents"], all_items["metadatas"]):
            r = retention_score(meta["last_seen_at"], meta["stability"])
            topic_summaries.append(doc[:40] + "..." if len(doc) > 40 else doc)
            retentions.append(r)
            stabilities.append(meta["stability"])

        import pandas as pd
        df_retention = pd.DataFrame({
            "Topic": topic_summaries,
            "Retention Score": retentions,
            "Stability (days)": stabilities
        }).set_index("Topic")

        st.bar_chart(df_retention["Retention Score"])
        st.dataframe(df_retention.style.format({"Retention Score": "{:.1%}", "Stability (days)": "{:.2f}"}))
    else:
        st.info("No concepts recorded in your long-term memory database yet.")

    # 3. Knowledge Graph concept map
    st.markdown("---")
    st.markdown("### 🕸️ Concept Association Map")
    
    if all_items["ids"]:
        dot_lines = [
            "digraph G {",
            '  node [shape=box, style="filled,rounded", color="#e0f2fe", fillcolor="#f0f9ff", fontname="Arial", fontsize=10];',
            '  edge [color="#cbd5e1", penwidth=1.5];',
            '  bgcolor="transparent";',
        ]

        edges = set()
        for doc, meta in zip(all_items["documents"], all_items["metadatas"]):
            label = doc[:45] + "..." if len(doc) > 45 else doc
            label = label.replace('"', '\\"').replace('\n', ' ')
            node_id = f'"{label}"'

            links_raw = meta.get("links_to", "[]")
            try:
                links = json.loads(links_raw)
            except Exception:
                links = []

            for parent in links:
                parent_label = parent.replace('"', '\\"').replace('\n', ' ')
                parent_id = f'"{parent_label}"'
                edges.add((parent_id, node_id))

        for p_id, c_id in edges:
            dot_lines.append(f"  {p_id} -> {c_id};")

        dot_lines.append("}")
        dot_string = "\n".join(dot_lines)

        if len(edges) > 0:
            st.graphviz_chart(dot_string)
        else:
            st.caption("No connections established yet. Keep teaching concepts to see relationships form!")
    else:
        st.caption("Graph maps will draw automatically once you cover related topics.")

    # 4. Exporter
    st.markdown("---")
    st.markdown("### 📥 Study Resource Exporter")
    st.markdown("Generate and download a personalized Markdown Study Guide and Active Recall Flashcard deck.")
    
    if all_items["ids"]:
        if st.button("Build Study Guide & Flashcards"):
            with st.spinner("Compiling concepts and generating flashcards..."):
                md_lines = [
                    "# Sentio Study Guide & Flashcard Deck",
                    f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "\n## 📚 Core Concept Index\n"
                ]
                
                concepts_text = ""
                for i, (doc, meta) in enumerate(zip(all_items["documents"], all_items["metadatas"]), 1):
                    pct = int(retention_score(meta["last_seen_at"], meta["stability"]) * 100)
                    md_lines.append(f"{i}. **{doc}**")
                    md_lines.append(f"   * Stability: `{meta['stability']:.2f}` days | Estimated Retention: `{pct}%` \n")
                    concepts_text += f"Concept {i}: {doc}\n"

                prompt = f"""You are a helpful study assistant. Create 1 active-recall flashcard (1 Question, 1 Answer) for each of the following concepts.
Keep the questions and answers clear, concise, and focused on self-testing.

Concepts:
{concepts_text}

Format the output strictly as a Markdown checklist like:
### 🎴 Flashcard Deck
- **Q**: [Question]
  **A**: [Answer]
"""
                try:
                    from langchain_core.messages import SystemMessage
                    res = extractor_llm.invoke([SystemMessage(content=prompt)])
                    flashcards_md = res.content.strip()
                    md_lines.append("\n" + flashcards_md)
                except Exception as e:
                    md_lines.append("\n### 🎴 Flashcards (Fallback)\n(Could not auto-generate flashcards: " + str(e) + ")")

                full_md = "\n".join(md_lines)
                
                st.markdown("#### Preview:")
                st.markdown(full_md[:1000] + "\n..." if len(full_md) > 1000 else full_md)
                
                st.download_button(
                    label="💾 Download Markdown File",
                    data=full_md,
                    file_name="Sentio_Study_Guide.md",
                    mime="text/markdown"
                )
    else:
        st.info("Start chatting to generate concepts for a study guide!")
