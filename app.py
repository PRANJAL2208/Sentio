"""
app.py — Sentio

Run with: streamlit run app.py
"""

import os
import uuid
import time
from datetime import datetime

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
from core.db import (
    init_database,
    register_user,
    get_user_group,
    log_session_start,
    log_session_end,
    log_quiz,
    log_workload,
    log_telemetry,
    get_all_users,
    get_all_sessions,
    get_all_quizzes,
    get_all_workloads,
    get_all_telemetry,
)
from core.auth import (
    get_google_auth_url,
    exchange_code_for_email,
    is_google_auth_configured,
)

load_dotenv()

# Initialize Central SQLite WAL database tables
init_database()


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sentio",
    page_icon="🧠",
    layout="centered",
)

# ── Space HUD Viewport Overlay ──
st.markdown("""
<div class="sentio-space-hud">
    <div class="nebula-bg"></div>
    <div class="grid-overlay"></div>
    <div class="hud-line hud-top-left"></div>
    <div class="hud-line hud-top-right"></div>
    <div class="hud-line hud-bottom-left"></div>
    <div class="hud-line hud-bottom-right"></div>
</div>
""", unsafe_allow_html=True)


# ── Multi-User Authentication Gate ────────────────────────────────────────────

if "user_email" not in st.session_state:
    st.session_state["user_email"] = None
if "group_assignment" not in st.session_state:
    st.session_state["group_assignment"] = None
if "study_mode" not in st.session_state:
    st.session_state["study_mode"] = "Study Router (Balanced)"

# Check if code is in query params (Google redirect callback)
query_params = st.query_params
if not st.session_state["user_email"] and "code" in query_params:
    auth_code = query_params["code"]
    email = exchange_code_for_email(auth_code)
    if email:
        st.session_state["user_email"] = email
        # Deterministically assign user group based on email hash
        group = "Group A" if hash(email) % 2 != 0 else "Group B"
        st.session_state["group_assignment"] = group
        register_user(email, group)
        st.query_params.clear()
        st.rerun()
    else:
        st.error("Google authentication failed. Please try again or use the email fallback.")

# Render Login screen if not authenticated
if not st.session_state["user_email"]:
    # Custom CSS style
    st.markdown("""
        <style>
        .login-card {
            background: #111827;
            border: 1px solid #374151;
            border-radius: 16px;
            padding: 35px 30px;
            max-width: 450px;
            margin: 40px auto;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 10px 10px -5px rgba(0, 0, 0, 0.4);
            text-align: center;
        }
        .login-logo {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 10px;
            color: #ffffff;
        }
        .login-desc {
            font-size: 0.95rem;
            color: #9ca3af;
            margin-bottom: 30px;
            line-height: 1.5;
        }
        .google-login-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background-color: #ffffff;
            color: #374151;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            padding: 12px 24px;
            font-size: 0.95rem;
            font-weight: 600;
            text-decoration: none;
            width: 100%;
            transition: background-color 0.2s, box-shadow 0.2s;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .google-login-btn:hover {
            background-color: #f9fafb;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .google-icon {
            width: 18px;
            height: 18px;
            margin-right: 10px;
        }
        .login-divider {
            display: flex;
            align-items: center;
            text-align: center;
            color: #6b7280;
            margin: 25px 0;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .login-divider::before, .login-divider::after {
            content: '';
            flex: 1;
            border-bottom: 1px solid #374151;
        }
        .login-divider:not(:empty)::before {
            margin-right: .5em;
        }
        .login-divider:not(:empty)::after {
            margin-left: .5em;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Render Login Container
    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    st.markdown('<div class="login-logo">🧠 Sentio</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-desc">Cognitive-Adaptive Intelligent Tutoring Visor. Analyzing typing biometrics to customize learning flows.</div>', unsafe_allow_html=True)
    
    # 1. Google OAuth
    if is_google_auth_configured():
        auth_url = get_google_auth_url()
        google_btn_html = f"""
        <a href="{auth_url}" target="_self" class="google-login-btn">
            <svg class="google-icon" viewBox="0 0 24 24" width="18" height="18" xmlns="http://www.w3.org/2000/svg">
                <g transform="matrix(1, 0, 0, 1, 0, 0)">
                    <path d="M21.35,11.1H12v2.7h5.38c-0.24,1.28 -0.96,2.37 -2.04,3.1v2.58h3.3c1.93,-1.78 3.04,-4.4 3.04,-7.48c0,-0.61 -0.06,-1.2 -0.16,-1.72Z" fill="#4285F4"/>
                    <path d="M12,20.58c2.43,0 4.47,-0.81 5.96,-2.2l-3.3,-2.58c-0.91,0.61 -2.08,0.98 -3.3,0.98c-2.34,0 -4.33,-1.58 -5.04,-3.7L2.9,15.18v2.66c1.48,2.94 4.52,4.74 8.1,4.74Z" fill="#34A853"/>
                    <path d="M6.96,13.08c-0.18,-0.54 -0.28,-1.11 -0.28,-1.7c0,-0.59 0.1,-1.16 0.28,-1.7V7.02H2.9C2.29,8.24 1.94,9.63 1.94,11.1c0,1.47 0.35,2.86 0.96,4.08l4.06,-3.1Z" fill="#FBBC05"/>
                    <path d="M12,4.82c1.32,0 2.51,0.45 3.44,1.35l2.58,-2.58c-1.55,-1.44 -3.59,-2.31 -6.02,-2.31c-3.58,0 -6.62,1.8 -8.1,4.74l4.06,3.1c0.71,-2.12 2.7,-3.7 5.04,-3.7Z" fill="#EA4335"/>
                </g>
            </svg>
            Sign in with Google
        </a>
        """
        st.markdown(google_btn_html, unsafe_allow_html=True)
        st.markdown('<div class="login-divider">Or continue with email</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="login-divider">Developer Login</div>', unsafe_allow_html=True)
        
    # 2. Email Fallback
    with st.container():
        email_input = st.text_input("Email Address", placeholder="e.g. pranjal@domain.com", label_visibility="collapsed").strip()
        submit_login = st.button("Launch Console", use_container_width=True, type="primary")
        if submit_login:
            if not email_input or "@" not in email_input:
                st.warning("Please enter a valid email address.")
            else:
                email = email_input.lower().strip()
                st.session_state["user_email"] = email
                group = "Group A" if hash(email) % 2 != 0 else "Group B"
                st.session_state["group_assignment"] = group
                register_user(email, group)
                st.rerun()
                
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


# Move researcher metadata to sidebar
st.sidebar.markdown(f"**User:** `{st.session_state['user_email']}`")
st.sidebar.markdown(f"**Group:** `{st.session_state['group_assignment']}`")

st.markdown("<h2 style='text-align: center; margin-top: -40px; margin-bottom: 20px; font-weight: 800; color: #5B21B6;'>🧠 Sentio</h2>", unsafe_allow_html=True)


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
        "current_topic_index":   0,
        "quiz_step":             "FREE_CHAT", # FREE_CHAT, IDLE, PRE_TEST, STUDY, POST_TEST, NASA_TLX
        "current_session_id":    None,
        "pre_test_answers":      {},
        "post_test_answers":     {},
        "study_start_time":      None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()


# ── Curriculum Configuration & Mode Routing ───────────────────────────────────

CURRICULUM_TOPICS = [
    {
        "name": "1. Artificial Intelligence vs. Agentic AI",
        "description": "Learn what makes Agentic AI unique, autonomous decision loops, and the difference between reactive and proactive AI agents.",
        "pre_quiz": [
            {"q": "Which of the following best describes 'Agentic AI'?", "options": ["AI that only responds to direct user queries.", "AI that makes decisions and takes actions autonomously.", "A simple rule-based decision tree."], "a": 1},
            {"q": "What is a key difference between reactive LLMs and proactive agentic systems?", "options": ["Reactive systems take initiative, while proactive agents wait.", "Proactive agents monitor environment states and trigger actions autonomously.", "There is no difference."], "a": 1},
            {"q": "What component serves as the central orchestration hub for AI tools?", "options": ["An API endpoint", "An MCP (Model Context Protocol) Server", "A local database cache"], "a": 1}
        ],
        "post_quiz": [
            {"q": "Which of the following best describes 'Agentic AI'?", "options": ["AI that only responds to direct user queries.", "AI that makes decisions and takes actions autonomously.", "A simple rule-based decision tree."], "a": 1},
            {"q": "What is a key difference between reactive LLMs and proactive agentic systems?", "options": ["Reactive systems take initiative, while proactive agents wait.", "Proactive agents monitor environment states and trigger actions autonomously.", "There is no difference."], "a": 1},
            {"q": "What component serves as the central orchestration hub for AI tools?", "options": ["An API endpoint", "An MCP (Model Context Protocol) Server", "A local database cache"], "a": 1}
        ]
    },
    {
        "name": "2. Transformers & Self-Attention",
        "description": "Understand how the Transformer architecture processes sequences using multi-head self-attention.",
        "pre_quiz": [
            {"q": "What problem does the attention mechanism solve in sequence modeling?", "options": ["Reducing training memory size.", "Capturing long-range word relationships without step-by-step recurrence.", "Forcing the model to run sequentially."], "a": 1},
            {"q": "In self-attention, what are the three vectors generated for each word?", "options": ["Value, Target, Weight", "Query, Key, Value", "Input, Hidden, Output"], "a": 1},
            {"q": "What is the purpose of having 'multiple heads' in self-attention?", "options": ["To run multiple models in parallel.", "To focus on different parts of the sentence simultaneously (e.g. grammar vs. pronouns).", "To speed up vocabulary translation."], "a": 1}
        ],
        "post_quiz": [
            {"q": "What problem does the attention mechanism solve in sequence modeling?", "options": ["Reducing training memory size.", "Capturing long-range word relationships without step-by-step recurrence.", "Forcing the model to run sequentially."], "a": 1},
            {"q": "In self-attention, what are the three vectors generated for each word?", "options": ["Value, Target, Weight", "Query, Key, Value", "Input, Hidden, Output"], "a": 1},
            {"q": "What is the purpose of having 'multiple heads' in self-attention?", "options": ["To focus on different parts of the sentence simultaneously.", "To run multiple models in parallel.", "To speed up vocabulary translation."], "a": 0}
        ]
    },
    {
        "name": "3. Spaced Repetition & Ebbinghaus Decay",
        "description": "Learn the math behind the Ebbinghaus forgetting curve and how reviews stabilize memory.",
        "pre_quiz": [
            {"q": "According to Hermann Ebbinghaus, how does memory retention decay over time?", "options": ["Linearly", "Exponentially", "Logarithmically"], "a": 1},
            {"q": "What does 'stability' represent in the Ebbinghaus decay formula?", "options": ["The size of the context window.", "The strength of a memory, which dictates how slowly it decays.", "The correctness score of the user's answer."], "a": 1},
            {"q": "What happens to the rate of forgetting after a successful review interval?", "options": ["It gets faster.", "It slows down (stability increases).", "It stays exactly the same."], "a": 1}
        ],
        "post_quiz": [
            {"q": "According to Hermann Ebbinghaus, how does memory retention decay over time?", "options": ["Linearly", "Exponentially", "Logarithmically"], "a": 1},
            {"q": "What does 'stability' represent in the Ebbinghaus decay formula?", "options": ["The strength of a memory, which dictates how slowly it decays.", "The size of the context window.", "The correctness score of the user's answer."], "a": 0},
            {"q": "What happens to the rate of forgetting after a successful review interval?", "options": ["It gets faster.", "It slows down (stability increases).", "It stays exactly the same."], "a": 1}
        ]
    },
    {
        "name": "4. Cognitive Load Theory",
        "description": "Learn Sweller's Cognitive Load Theory, active working memory, and mental schema.",
        "pre_quiz": [
            {"q": "What is a core bottleneck in human information processing?", "options": ["Extraneous reading speed.", "Limited working memory capacity (typically 4-7 chunks).", "Infinite long-term memory access times."], "a": 1},
            {"q": "Which type of cognitive load is caused by poorly designed instructions or interfaces?", "options": ["Intrinsic load", "Extraneous load", "Germane load"], "a": 1},
            {"q": "How does schema acquisition help reduce working memory load?", "options": ["By grouping multiple pieces of info into a single chunk.", "By bypassing long-term memory storage.", "By increasing physical typing speed."], "a": 0}
        ],
        "post_quiz": [
            {"q": "What is a core bottleneck in human information processing?", "options": ["Extraneous reading speed.", "Limited working memory capacity (typically 4-7 chunks).", "Infinite long-term memory access times."], "a": 1},
            {"q": "Which type of cognitive load is caused by poorly designed instructions or interfaces?", "options": ["Intrinsic load", "Extraneous load", "Germane load"], "a": 1},
            {"q": "How does schema acquisition help reduce working memory load?", "options": ["By grouping multiple pieces of info into a single chunk.", "By bypassing long-term memory storage.", "By increasing physical typing speed."], "a": 0}
        ]
    }
]

def get_current_mode() -> str:
    chosen = st.session_state.get("study_mode", "Study Router (Balanced)")
    if chosen == "Sentio Mode (Force Adaptive)":
        return "SENTIO"
    elif chosen == "Control Mode (Force Vanilla)":
        return "CONTROL"
    else:
        # Balanced Router based on Odd/Even Group and Topic index
        group = st.session_state.get("group_assignment", "Group A")
        idx = st.session_state.get("current_topic_index", 0)
        if group == "Group A":
            return "SENTIO" if idx % 2 == 0 else "CONTROL"
        else:
            return "CONTROL" if idx % 2 == 0 else "SENTIO"


# ── Inject JS timing tracker & Telemetry Bridge ────────────────────────────────

# Hidden text input to sync telemetry JSON from JS component
st.text_input("telemetry_data", label_visibility="collapsed", key="telemetry_input")

# Hide the telemetry text input via CSS and apply premium styling overrides
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: #030307 !important;
    color: #f4f4f5 !important;
}

[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* 🌌 Cinematic Background layers */
.sentio-space-hud {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    pointer-events: none;
}

.nebula-bg {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.15) 0%, transparent 45%),
                radial-gradient(circle at 90% 80%, rgba(16, 185, 129, 0.1) 0%, transparent 45%),
                radial-gradient(circle at 50% 50%, rgba(244, 63, 94, 0.06) 0%, transparent 50%),
                #030307;
    z-index: -2;
    pointer-events: none;
}

.grid-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background-image: linear-gradient(rgba(255, 255, 255, 0.015) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(255, 255, 255, 0.015) 1px, transparent 1px);
    background-size: 30px 30px;
    z-index: -1;
    pointer-events: none;
}

/* Cinematic Cyber Brackets */
.hud-line {
    position: fixed;
    background-color: rgba(99, 102, 241, 0.35);
    pointer-events: none;
    z-index: 1000;
}

.hud-top-left {
    top: 20px;
    left: 20px;
    width: 60px;
    height: 2px;
    box-shadow: 0 0 10px rgba(99, 102, 241, 0.6);
}
.hud-top-left::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 2px;
    height: 60px;
    background-color: rgba(99, 102, 241, 0.35);
    box-shadow: 0 0 10px rgba(99, 102, 241, 0.6);
}

.hud-top-right {
    top: 20px;
    right: 20px;
    width: 60px;
    height: 2px;
    box-shadow: 0 0 10px rgba(99, 102, 241, 0.6);
}
.hud-top-right::after {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 2px;
    height: 60px;
    background-color: rgba(99, 102, 241, 0.35);
    box-shadow: 0 0 10px rgba(99, 102, 241, 0.6);
}

.hud-bottom-left {
    bottom: 20px;
    left: 20px;
    width: 60px;
    height: 2px;
    box-shadow: 0 0 10px rgba(99, 102, 241, 0.6);
}
.hud-bottom-left::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 2px;
    height: 60px;
    background-color: rgba(99, 102, 241, 0.35);
    box-shadow: 0 0 10px rgba(99, 102, 241, 0.6);
}

.hud-bottom-right {
    bottom: 20px;
    right: 20px;
    width: 60px;
    height: 2px;
    box-shadow: 0 0 10px rgba(99, 102, 241, 0.6);
}
.hud-bottom-right::after {
    content: '';
    position: absolute;
    bottom: 0;
    right: 0;
    width: 2px;
    height: 60px;
    background-color: rgba(99, 102, 241, 0.35);
    box-shadow: 0 0 10px rgba(99, 102, 241, 0.6);
}

/* ⚙️ Sidebar [SYSTEM DECK] Panel */
[data-testid="stSidebar"] {
    background-color: rgba(10, 10, 18, 0.96) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.25) !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 0 25px rgba(99, 102, 241, 0.08) !important;
}

/* Custom visual system titles inside sidebar */
[data-testid="stSidebarUserContent"]::before {
    content: '[ SYSTEM DECK ]' !important;
    display: block !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    color: #818cf8 !important;
    letter-spacing: 0.15em !important;
    border-bottom: 1px solid rgba(99, 102, 241, 0.2) !important;
    padding-bottom: 10px !important;
    margin-bottom: 20px !important;
    text-shadow: 0 0 10px rgba(99, 102, 241, 0.5) !important;
}

/* Styling Inputs with cyber themes */
div[data-testid="stSelectbox"] > label, div[data-testid="stTextInput"] > label {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    color: #a5b4fc !important;
    letter-spacing: 0.05em !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"], div[data-testid="stTextInput"] input {
    background-color: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 6px !important;
    color: #ffffff !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"]:hover, div[data-testid="stTextInput"] input:hover {
    border-color: #6366f1 !important;
    box-shadow: 0 0 8px rgba(99, 102, 241, 0.2) !important;
}

div[data-testid="stTextInput"]:has(input[aria-label="telemetry_data"]) {
    display: none !important;
}

/* 💬 Holographic central Visor console header */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px !important;
    background-color: rgba(99, 102, 241, 0.05) !important;
    padding: 6px !important;
    border-radius: 12px !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif !important;
    height: 38px !important;
    border-radius: 8px !important;
    border: none !important;
    color: rgba(255, 255, 255, 0.5) !important;
    background-color: transparent !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
    padding: 0 20px !important;
    transition: all 0.2s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #ffffff !important;
    background-color: rgba(99, 102, 241, 0.1) !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #ffffff !important;
    background-color: rgba(99, 102, 241, 0.25) !important;
    border: 1px solid rgba(99, 102, 241, 0.4) !important;
    box-shadow: 0 0 10px rgba(99, 102, 241, 0.2) !important;
}

.stTabs [role="tabpanel"] {
    padding-top: 1.5rem !important;
}

/* Chat Input styling */
[data-testid="stChatInput"] {
    background-color: rgba(10, 10, 18, 0.85) !important;
    border: 1px solid rgba(99, 102, 241, 0.35) !important;
    border-radius: 8px !important;
    box-shadow: 0 0 15px rgba(99, 102, 241, 0.08) !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 25px rgba(99, 102, 241, 0.2) !important;
}

[data-testid="stChatInput"] textarea {
    color: #fafafa !important;
    background: transparent !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Glassmorphic custom styling classes */
.sentio-chat-container {
    display: flex;
    flex-direction: column;
    gap: 20px;
    margin-bottom: 2rem;
}

.sentio-row {
    display: flex;
    width: 100%;
}

.sentio-row-user {
    justify-content: flex-end;
}

.sentio-row-tutor {
    justify-content: flex-start;
}

.sentio-bubble {
    max-width: 80%;
    padding: 16px 20px;
    border-radius: 16px;
    font-size: 0.95rem;
    line-height: 1.6;
    color: #f4f4f5;
    position: relative;
    box-sizing: border-box;
}

.sentio-bubble-user {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    border-bottom-right-radius: 4px !important;
    box-shadow: 0 0 12px rgba(99, 102, 241, 0.05) !important;
}

.sentio-bubble-tutor {
    background: rgba(10, 10, 18, 0.6) !important;
    border-bottom-left-radius: 4px !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15) !important;
}

.sentio-glow-overloaded {
    border: 1px solid rgba(244, 63, 94, 0.4) !important;
    box-shadow: 0 0 20px rgba(244, 63, 94, 0.12) !important;
}

.sentio-glow-optimal {
    border: 1px solid rgba(16, 185, 129, 0.4) !important;
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.12) !important;
}

.sentio-glow-underloaded {
    border: 1px solid rgba(99, 102, 241, 0.4) !important;
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.12) !important;
}

.sentio-badge {
    display: inline-flex;
    align-items: center;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 6px;
    margin-bottom: 8px;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

.sentio-badge-overloaded {
    background: rgba(244, 63, 94, 0.15) !important;
    color: #fb7185 !important;
    border: 1px solid rgba(244, 63, 94, 0.25) !important;
}

.sentio-badge-optimal {
    background: rgba(16, 185, 129, 0.15) !important;
    color: #34d399 !important;
    border: 1px solid rgba(16, 185, 129, 0.25) !important;
}

.sentio-badge-underloaded {
    background: rgba(99, 102, 241, 0.15) !important;
    color: #818cf8 !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
}

.sentio-srs-card {
    background: rgba(245, 158, 11, 0.04) !important;
    border: 1px solid rgba(245, 158, 11, 0.2) !important;
    padding: 10px 14px;
    border-radius: 8px;
    margin-top: 10px;
    font-size: 0.85rem;
    color: #fcd34d;
}

/* Stats Dashboard widgets */
.sentio-stats-grid {
    display: flex;
    gap: 16px;
    width: 100%;
    margin-bottom: 30px;
    flex-wrap: wrap;
}

.sentio-stats-card {
    flex: 1;
    min-width: 180px;
    background: rgba(10, 10, 18, 0.7) !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    padding: 20px;
    border-radius: 0px !important;
    clip-path: polygon(0 0, 100% 0, 100% calc(100% - 15px), calc(100% - 15px) 100%, 0 100%) !important;
    box-shadow: 0 0 15px rgba(99, 102, 241, 0.05);
    transition: all 0.3s ease;
    position: relative;
}

.sentio-stats-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%),
                linear-gradient(90deg, rgba(255, 0, 0, 0.04), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.04));
    background-size: 100% 4px, 6px 100%;
    pointer-events: none;
    z-index: 2;
}

.sentio-stats-card:hover {
    border-color: rgba(99, 102, 241, 0.5) !important;
    background: rgba(99, 102, 241, 0.06) !important;
    transform: translateY(-3px) scale(1.01);
    box-shadow: 0 0 25px rgba(99, 102, 241, 0.15);
}

.sentio-stats-label {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.75rem;
    font-weight: 700;
    color: #818cf8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.sentio-stats-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #ffffff;
    margin: 8px 0;
    text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
}

.sentio-stats-desc {
    font-size: 0.75rem;
    color: rgba(255, 255, 255, 0.4);
}

.stDownloadButton button {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(79, 70, 229, 0.4)) !important;
    color: #ffffff !important;
    border: 1px solid rgba(99, 102, 241, 0.4) !important;
    border-radius: 0px !important;
    clip-path: polygon(0 0, 100% 0, 92% 100%, 0 100%) !important;
    padding: 12px 24px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 15px rgba(99, 102, 241, 0.1) !important;
}

.stDownloadButton button:hover {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.35), rgba(79, 70, 229, 0.55)) !important;
    border-color: #6366f1 !important;
    box-shadow: 0 0 25px rgba(99, 102, 241, 0.25) !important;
    transform: scale(1.03) !important;
}

/* Ensure all markdown text and list items inside the sidebar are highly legible (light grey/indigo) */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
    color: #e2e8f0 !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li strong {
    color: #ffffff !important;
}

/* Sidebar captions, small elements, and minor system labels */
[data-testid="stSidebar"] p[class*="caption"],
[data-testid="stSidebar"] span[class*="caption"],
[data-testid="stSidebar"] div[data-testid="stCaptionContainer"],
[data-testid="stSidebar"] small {
    color: #94a3b8 !important;
}

/* Sidebar dividers */
[data-testid="stSidebar"] hr {
    border-color: rgba(99, 102, 241, 0.25) !important;
}

/* Sidebar expanders styling (Raw signals, etc.) */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background-color: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    border-radius: 6px !important;
}

[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    color: #a5b4fc !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
}

/* Sidebar checkbox and toggle labels */
[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p {
    color: #cbd5e1 !important;
}

/* Force the sticky bottom chat input container block to match the deep dark space theme and not render white */
[data-testid="stBottom"] {
    background-color: #030307 !important;
    background: #030307 !important;
    border-top: 1px solid rgba(99, 102, 241, 0.15) !important;
}

[data-testid="stBottom"] > div {
    background-color: transparent !important;
    background: transparent !important;
}

[data-testid="stChatInput"] {
    background-color: rgba(10, 10, 18, 0.95) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 8px !important;
    box-shadow: 0 0 15px rgba(99, 102, 241, 0.08) !important;
}

[data-testid="stChatInput"] div {
    background-color: transparent !important;
    background: transparent !important;
}

[data-testid="stChatInput"] textarea {
    color: #fafafa !important;
    background-color: transparent !important;
    background: transparent !important;
}

textarea::placeholder, input::placeholder {
    color: rgba(255, 255, 255, 0.4) !important;
}

/* Force main content text color to light grey/white to prevent dark-on-dark invisible text in Light OS mode */
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p,
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] li,
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] span,
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] small,
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] strong,
[data-testid="stAppViewContainer"] label {
    color: #e2e8f0 !important;
}

[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] h5,
[data-testid="stAppViewContainer"] h6 {
    color: #ffffff !important;
}

/* Futuristic warning alert blocks (st.error/st.warning/st.info overrides) */
div[data-testid="stAlert"] {
    background-color: rgba(244, 63, 94, 0.08) !important;
    border: 1px solid rgba(244, 63, 94, 0.35) !important;
    border-radius: 8px !important;
    box-shadow: 0 0 15px rgba(244, 63, 94, 0.1) !important;
}

div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
    color: #fb7185 !important;
}

div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] strong {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)


import re

def markdown_to_html(text: str) -> str:
    # 1. Bold
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    # 2. Italic
    html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)
    # 3. Inline code
    html = re.sub(r'`(.*?)`', r'<code style="background: rgba(255,255,255,0.06); padding: 2px 5px; border-radius: 4px; font-family: monospace;">\1</code>', html)
    # 4. Fenced code blocks
    html = re.sub(r'```python(.*?)```', r'<pre style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); padding: 12px; border-radius: 8px; font-family: monospace; overflow-x: auto;"><code>\1</code></pre>', html, flags=re.DOTALL)
    html = re.sub(r'```(.*?)```', r'<pre style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); padding: 12px; border-radius: 8px; font-family: monospace; overflow-x: auto;"><code>\1</code></pre>', html, flags=re.DOTALL)
    # 5. Unordered lists
    lines = html.split('\n')
    in_list = False
    for i, line in enumerate(lines):
        match = re.match(r'^\s*[\-\*]\s+(.*)', line)
        if match:
            item = match.group(1)
            if not in_list:
                lines[i] = '<ul><li>' + item + '</li>'
                in_list = True
            else:
                lines[i] = '<li>' + item + '</li>'
        else:
            if in_list:
                lines[i-1] = lines[i-1] + '</ul>'
                in_list = False
    if in_list:
        lines[-1] = lines[-1] + '</ul>'
        
    html = '\n'.join(lines)
    
    # 6. Newlines to br
    html = html.replace('\n', '<br/>')
    return html


def render_chat_bubble(role: str, content: str, load_state: str = "OPTIMAL", srs_evaluation: str = "", show_load_badge: bool = True):
    html_content = markdown_to_html(content)
    
    if role == "user":
        bubble_html = (
            f'<div class="sentio-row sentio-row-user" style="margin-bottom: 20px; display: flex; justify-content: flex-end; width: 100%;">'
            f'  <div class="sentio-bubble sentio-bubble-user">'
            f'    {html_content}'
            f'  </div>'
            f'</div>'
        )
    else:
        glow_class = {
            "OVERLOADED": "sentio-glow-overloaded",
            "OPTIMAL": "sentio-glow-optimal",
            "UNDERLOADED": "sentio-glow-underloaded"
        }.get(load_state, "sentio-glow-optimal")
        
        badge_text = {
            "OVERLOADED": "Simplified",
            "OPTIMAL": "Normal",
            "UNDERLOADED": "Enriched"
        }.get(load_state, "Normal")
        
        badge_class = {
            "OVERLOADED": "sentio-badge-overloaded",
            "OPTIMAL": "sentio-badge-optimal",
            "UNDERLOADED": "sentio-badge-underloaded"
        }.get(load_state, "sentio-badge-optimal")
        
        badge_html = f'<div class="sentio-badge {badge_class}">{badge_text}</div>' if show_load_badge else ''
        
        eval_html = ''
        if srs_evaluation:
            grade_emojis = {"correct": "🎯 Correct!", "partial": "⚠️ Close!", "incorrect": "❌ Not quite."}
            eval_html = f'<div class="sentio-srs-card"><b>Spaced Repetition Practice:</b> {grade_emojis.get(srs_evaluation, "")}</div>'
            
        bubble_html = (
            f'<div class="sentio-row sentio-row-tutor" style="margin-bottom: 20px; display: flex; justify-content: flex-start; width: 100%;">'
            f'  <div class="sentio-bubble sentio-bubble-tutor {glow_class}">'
            f'    {badge_html}'
            f'    <div>{html_content}</div>'
            f'    {eval_html}'
            f'  </div>'
            f'</div>'
        )
        
    st.markdown(bubble_html, unsafe_allow_html=True)

inject_timing_tracker()


# ── Setup Memory (needs to be initialized early for sidebar rendering) ────────

@st.cache_resource
def get_memory(user_id: str = "default"):
    client = get_chroma_client()
    return get_or_create_collection(client, user_id=user_id)

def get_current_user_collection():
    email = st.session_state.get("user_email", "default")
    if not email:
        email = "default"
    import re
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', email)
    clean_name = clean_name[:60]
    if len(clean_name) < 3:
        clean_name = "default_" + clean_name
    return get_memory(clean_name)

memory_collection = get_current_user_collection()


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
    st.markdown("### 🛠️ Study Controller")
    st.session_state["study_mode"] = st.selectbox(
        "Study Mode Selector",
        ["Study Router (Balanced)", "Sentio Mode (Force Adaptive)", "Control Mode (Force Vanilla)"],
        index=["Study Router (Balanced)", "Sentio Mode (Force Adaptive)", "Control Mode (Force Vanilla)"].index(st.session_state.get("study_mode", "Study Router (Balanced)"))
    )
    
    st.divider()
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
        "API Key (Leave blank for fallback)",
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
        
    if not resolved_api_key:
        default_groq_key = ""
        try:
            default_groq_key = st.secrets.get("DEFAULT_GROQ_API_KEY", "")
        except Exception:
            pass
        default_groq_key = os.getenv("DEFAULT_GROQ_API_KEY") or os.getenv("GROQ_API_KEY") or default_groq_key
        if default_groq_key:
            provider = "Groq"
            model_name = "llama-3.3-70b-versatile"
            resolved_api_key = default_groq_key
            st.sidebar.info("🤖 Using system Groq fallback.")

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

admin_emails = ["admin@sentio.org", "tester@sentio.org", "pranjal2208@gmail.com"]
is_admin = st.session_state.get("user_email") in admin_emails or st.query_params.get("admin", "false") == "true"

if is_admin:
    tab_tutor, tab_dashboard, tab_admin = st.tabs(["💬 Sentio Tutor", "📊 Learning Dashboard", "🔐 Admin Console"])
else:
    tab_tutor, tab_dashboard = st.tabs(["💬 Sentio Tutor", "📊 Learning Dashboard"])


# ── Render Tutor Interface ───────────────────────────────────────────────────

with tab_tutor:
    current_idx = st.session_state["current_topic_index"]
    
    if current_idx >= len(CURRICULUM_TOPICS):
        st.balloons()
        st.markdown("## 🎉 Study Complete!")
        st.markdown(
            "Thank you for participating in this research study! "
            "You have completed all 4 topic sessions. Please download the research data log "
            "below and submit it to the study administrator."
        )
        
        # Read database bytes
        try:
            with open("sentio_study.db", "rb") as f:
                db_bytes = f.read()
            st.download_button(
                label="💾 Download sentio_study.db (Database File)",
                data=db_bytes,
                file_name=f"sentio_study_{st.session_state['user_email'].replace('@', '_').replace('.', '_')}.db",
                mime="application/x-sqlite3"
            )
        except Exception as e:
            st.error(f"Could not load database file for download: {e}")
            
        st.stop()
        
    topic = CURRICULUM_TOPICS[current_idx]
    mode = get_current_mode()
    step = st.session_state["quiz_step"]
    
    # Exit study session button in sidebar if active
    if step != "FREE_CHAT":
        st.sidebar.divider()
        if st.sidebar.button("💬 Exit to General Chat", help="Exit the research wizard and return to free chat"):
            st.session_state["quiz_step"] = "FREE_CHAT"
            st.session_state["chat_history"] = []
            st.rerun()
            
    if step == "FREE_CHAT":
        col_t, col_btn = st.columns([3, 1])
        with col_t:
            st.markdown("### 💬 General-Purpose Adaptive Chat")
            st.caption("Ask me anything about any topic! I will analyze your typing speed, dwell times, and flight times to dynamically adjust my vocabulary complexity and explain key concepts step-by-step.")
        with col_btn:
            if st.button("🧪 Launch Study Wizard", help="Start the structured 4-topic research evaluation"):
                st.session_state["quiz_step"] = "IDLE"
                st.session_state["chat_history"] = []
                st.rerun()
                
        st.divider()

        # Render existing chat history
        st.markdown('<div class="sentio-chat-container">', unsafe_allow_html=True)
        for turn in st.session_state["chat_history"]:
            load_badge = show_load_badge if mode == "SENTIO" else False
            render_chat_bubble(
                role=turn["role"],
                content=turn["content"],
                load_state=turn.get("load_state", "OPTIMAL") if mode == "SENTIO" else "OPTIMAL",
                srs_evaluation=turn.get("srs_evaluation", "") if mode == "SENTIO" else "",
                show_load_badge=load_badge
            )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if not st.session_state["chat_history"]:
            st.info(
                "👋 Welcome! I am your cognitive-adaptive AI tutor. "
                "Type or paste any topic you want to learn about—I will adjust explanation spacing and details in real-time."
            )
            
        # Chat input
        if user_input := st.chat_input("Ask me anything..."):
            telemetry = get_typing_telemetry(st.session_state)
            clarification_count = update_clarification_count(st.session_state, user_input)
            
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
            st.rerun()
            
    elif step == "IDLE":
        st.markdown(f"## Topic {current_idx + 1}: {topic['name']}")
        st.markdown(topic['description'])
        st.info(f"Session Mode: **{mode}** (Assigned automatically for balanced research control)")
        
        if st.button("Start Study Session", key="start_session_btn"):
            st.session_state["quiz_step"] = "PRE_TEST"
            st.session_state["pre_test_answers"] = {}
            st.session_state["current_session_id"] = f"{st.session_state['user_email']}_{current_idx}_{int(time.time())}"
            log_session_start(
                st.session_state["current_session_id"],
                st.session_state["user_email"],
                topic["name"],
                mode
            )
            st.rerun()
            
    elif step == "PRE_TEST":
        st.markdown(f"## 📋 Pre-Test: {topic['name']}")
        st.caption("Please answer the following questions based on your current knowledge. (Try your best, guessing is okay!)")
        
        pre_answers = {}
        with st.form("pre_test_form"):
            for i, q in enumerate(topic["pre_quiz"]):
                st.markdown(f"**Q{i+1}: {q['q']}**")
                pre_answers[i] = st.radio(
                    f"Select answer for Q{i+1}:", 
                    q["options"], 
                    key=f"pre_q_{i}", 
                    index=None
                )
            
            submit_pre = st.form_submit_button("Submit Pre-Test & Start Studying")
            if submit_pre:
                if any(ans is None for ans in pre_answers.values()):
                    st.warning("Please answer all questions before submitting.")
                else:
                    score = 0
                    for i, q in enumerate(topic["pre_quiz"]):
                        selected_index = q["options"].index(pre_answers[i])
                        if selected_index == q["a"]:
                            score += 1
                    
                    log_quiz(st.session_state["user_email"], topic["name"], "PRE", score, len(topic["pre_quiz"]))
                    st.session_state["quiz_step"] = "STUDY"
                    st.session_state["chat_history"] = []
                    st.session_state["study_start_time"] = time.time()
                    st.rerun()
                    
    elif step == "STUDY":
        elapsed = time.time() - st.session_state.get("study_start_time", time.time())
        remaining = max(0, 300 - int(elapsed))  # 5 minutes study duration
        
        st.markdown(f"### 💬 Topic Study: {topic['name']}")
        
        col_timer, col_mode = st.columns(2)
        with col_timer:
            mins, secs = divmod(remaining, 60)
            st.metric("⏳ Study Time Remaining", f"{mins:02d}:{secs:02d}")
        with col_mode:
            st.metric("🔬 Current Mode", mode)
            
        st.divider()
        
        # Render existing chat history
        st.markdown('<div class="sentio-chat-container">', unsafe_allow_html=True)
        for turn in st.session_state["chat_history"]:
            load_badge = show_load_badge if mode == "SENTIO" else False
            render_chat_bubble(
                role=turn["role"],
                content=turn["content"],
                load_state=turn.get("load_state", "OPTIMAL") if mode == "SENTIO" else "OPTIMAL",
                srs_evaluation=turn.get("srs_evaluation", "") if mode == "SENTIO" else "",
                show_load_badge=load_badge
            )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if not st.session_state["chat_history"]:
            st.info(
                f"👋 Welcome! I am your AI tutor for **{topic['name']}**. "
                "Ask me any questions you have about this topic. I will adapt to help you learn."
            )
            
        # Check if time is completed
        if remaining <= 0:
            st.success("🎉 Time's up! You have completed the 5-minute study window.")
            if st.button("Proceed to Post-Test Evaluation", key="post_test_proceed_btn"):
                log_session_end(st.session_state["current_session_id"])
                st.session_state["quiz_step"] = "POST_TEST"
                st.session_state["post_test_answers"] = {}
                st.rerun()
        else:
            # Render chat input
            if user_input := st.chat_input("Ask me anything..."):
                telemetry = get_typing_telemetry(st.session_state)
                clarification_count = update_clarification_count(st.session_state, user_input)
                
                # Log telemetry to database
                log_telemetry(st.session_state["current_session_id"], telemetry)
                
                with st.spinner("Thinking..."):
                    # If in CONTROL mode, telemetry is calculated and logged, but bypassed/ignored for routing
                    active_load_state = None
                    if mode == "CONTROL":
                        # Bypass and force OPTIMAL prompt settings
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
                            srs_quiz_active=False,  # Bypass Ebbinghaus quizzes in Control
                            srs_topic_id="",
                            turns_since_last_quiz=999,
                        )
                        result["load_state"] = "OPTIMAL" # Override response visual badges
                    else:
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
                        
                # Update session states
                if mode == "SENTIO":
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
                
                # Store memory (only in Sentio mode)
                if mode == "SENTIO" and not result.get("srs_evaluation") and not result.get("srs_quiz_active"):
                    try:
                        topics = extract_topics_from_response(extractor_llm, result["response"])
                        for t in topics:
                            store_topic(
                                collection=memory_collection,
                                topic_summary=t["summary"],
                                session_id=st.session_state["session_id"],
                                links_to=t["links_to"],
                            )
                        # Resurface reviews
                        forgotten_now = get_forgotten_topics(memory_collection)
                        for item in forgotten_now:
                            update_topic_on_review(memory_collection, item["topic_id"])
                    except Exception:
                        pass
                        
                st.rerun()

    elif step == "POST_TEST":
        st.markdown(f"## 📋 Post-Test Evaluation: {topic['name']}")
        st.caption("Please answer the following questions to verify what you learned during the session.")
        
        post_answers = {}
        with st.form("post_test_form"):
            for i, q in enumerate(topic["post_quiz"]):
                st.markdown(f"**Q{i+1}: {q['q']}**")
                post_answers[i] = st.radio(
                    f"Select answer for Q{i+1}:", 
                    q["options"], 
                    key=f"post_q_{i}", 
                    index=None
                )
            
            submit_post = st.form_submit_button("Submit Post-Test")
            if submit_post:
                if any(ans is None for ans in post_answers.values()):
                    st.warning("Please answer all questions before submitting.")
                else:
                    score = 0
                    for i, q in enumerate(topic["post_quiz"]):
                        selected_index = q["options"].index(post_answers[i])
                        if selected_index == q["a"]:
                            score += 1
                    
                    log_quiz(st.session_state["user_email"], topic["name"], "POST", score, len(topic["post_quiz"]))
                    st.session_state["quiz_step"] = "NASA_TLX"
                    st.rerun()
                    
    elif step == "NASA_TLX":
        st.markdown(f"## 📊 Cognitive Workload Assessment (NASA-TLX)")
        st.caption("Please rate your experience during the last study session on each of the scales below.")
        
        with st.form("nasa_tlx_form"):
            mental = st.slider("🧠 Mental Demand: How mentally demanding was the task?", 0, 100, 50)
            physical = st.slider("💪 Physical Demand: How physically demanding was it?", 0, 100, 10)
            temporal = st.slider("⏳ Temporal Demand: Did you feel rushed or hurried?", 0, 100, 30)
            performance = st.slider("🎯 Performance: How successful do you think you were?", 0, 100, 70)
            effort = st.slider("🔥 Effort: How hard did you have to work to learn?", 0, 100, 50)
            frustration = st.slider("😡 Frustration: How insecure, discouraged, or stressed were you?", 0, 100, 20)
            
            submit_tlx = st.form_submit_button("Complete Session & Next Topic")
            if submit_tlx:
                ratings = {
                    "mental_demand": mental,
                    "physical_demand": physical,
                    "temporal_demand": temporal,
                    "performance": performance,
                    "effort": effort,
                    "frustration": frustration,
                }
                log_workload(st.session_state["user_email"], topic["name"], mode, ratings)
                st.session_state["current_topic_index"] += 1
                st.session_state["quiz_step"] = "IDLE"
                st.session_state["chat_history"] = []
                st.rerun()


# ── Render learning dashboard ───────────────────────────────────────────────

with tab_dashboard:
    st.markdown("## 📊 Learning Analytics Dashboard")
    st.caption("Track your biometrics, memory retention curves, and concept associations in real-time.")

    # Render custom Stats Grid
    last_sig = {}
    if st.session_state["load_state_history"]:
        last_sig = st.session_state["load_state_history"][-1]["signals"]
        
    flight = f"{last_sig.get('avg_flight_ms', 0.0):.0f} ms" if last_sig.get('avg_flight_ms', 0.0) > 0 else "N/A"
    dwell = f"{last_sig.get('avg_dwell_ms', 0.0):.0f} ms" if last_sig.get('avg_dwell_ms', 0.0) > 0 else "N/A"
    backspaces = f"{last_sig.get('backspace_count', 0)}" if last_sig.get('backspace_count', 0) > 0 else "0"
    pause = f"{last_sig.get('pause_seconds', 0.0):.1f} s" if last_sig.get('pause_seconds', 0.0) > 0 else "0.0 s"
    
    stats_html = f"""
    <div class="sentio-stats-grid">
        <div class="sentio-stats-card">
            <div class="sentio-stats-label">✈️ Flight Time</div>
            <div class="sentio-stats-value">{flight}</div>
            <div class="sentio-stats-desc">Transition delay between keys</div>
        </div>
        <div class="sentio-stats-card">
            <div class="sentio-stats-label">🔏 Dwell Time</div>
            <div class="sentio-stats-value">{dwell}</div>
            <div class="sentio-stats-desc">Key hold duration</div>
        </div>
        <div class="sentio-stats-card">
            <div class="sentio-stats-label">⌫ Backspaces</div>
            <div class="sentio-stats-value">{backspaces}</div>
            <div class="sentio-stats-desc">Deletions / revisions</div>
        </div>
        <div class="sentio-stats-card">
            <div class="sentio-stats-label">⏳ Reading Pause</div>
            <div class="sentio-stats-value">{pause}</div>
            <div class="sentio-stats-desc">Time spent planning input</div>
        </div>
    </div>
    """
    st.markdown(stats_html, unsafe_allow_html=True)

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
            '  node [shape=box, style="filled,rounded", color="#6366f1", fillcolor="#18181b", fontcolor="#fafafa", fontname="Arial", fontsize=10];',
            '  edge [color="#4f46e5", penwidth=1.2];',
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


if is_admin:
    with tab_admin:
        st.markdown("## 🔐 Admin Research Console")
        st.caption("Central control panel for monitoring study sessions, excluding outliers, and compiling statistics.")
        
        # Password Gate
        admin_pass = st.text_input("Enter Admin Password", type="password", key="admin_password_field")
        if admin_pass != os.getenv("ADMIN_KEY", "sentio2026"):
            if admin_pass:
                st.error("Incorrect Admin Password.")
            st.info("Please enter the admin security password to unlock raw study data and analytical compile features.")
        else:
            st.success("Admin access granted. Unlocking telemetry datastores...")
            st.divider()
            
            # 1. Cloud Connection Status
            from core.db import is_supabase_enabled
            if is_supabase_enabled():
                st.success("🟢 Supabase Cloud Sync active. Fetching data from Supabase remote database.")
            else:
                st.info("⚪ Local SQLite only. Fetching local data from sentio_study.db.")
                
            # Fetch all records from datastores
            with st.spinner("Fetching data records..."):
                users = get_all_users()
                sessions = get_all_sessions()
                quizzes = get_all_quizzes()
                workloads = get_all_workloads()
                telemetry = get_all_telemetry()
                
            if not users:
                st.warning("No participants registered yet.")
            else:
                import pandas as pd
                import numpy as np
                import math
                
                df_users = pd.DataFrame(users)
                df_sessions = pd.DataFrame(sessions) if sessions else pd.DataFrame(columns=["session_id", "email", "topic_name", "study_mode", "start_time", "end_time"])
                df_quizzes = pd.DataFrame(quizzes) if quizzes else pd.DataFrame(columns=["email", "topic_name", "quiz_type", "score", "total_questions", "timestamp"])
                df_workload = pd.DataFrame(workloads) if workloads else pd.DataFrame(columns=["email", "topic_name", "study_mode", "mental_demand", "physical_demand", "temporal_demand", "performance", "effort", "frustration", "timestamp"])
                df_telemetry = pd.DataFrame(telemetry) if telemetry else pd.DataFrame(columns=["session_id", "backspace_count", "avg_dwell_ms", "avg_flight_ms", "pause_seconds", "timestamp"])
                
                # Format display
                st.markdown("### 👥 Participant Cohort Status")
                st.dataframe(df_users, use_container_width=True)
                
                # Multi-select to filter outliers
                emails_list = sorted(list(df_users["email"].unique()))
                excluded_emails = st.multiselect(
                    "Select User Emails to EXCLUDE from statistical analysis (outliers/testers):",
                    options=emails_list,
                    default=[]
                )
                
                included_emails = [e for e in emails_list if e not in excluded_emails]
                
                if not included_emails:
                    st.error("Please include at least one user email to compile statistics.")
                else:
                    # Filter dataframes
                    df_q_filtered = df_quizzes[df_quizzes["email"].isin(included_emails)].copy()
                    df_w_filtered = df_workload[df_workload["email"].isin(included_emails)].copy()
                    df_s_filtered = df_sessions[df_sessions["email"].isin(included_emails)].copy()
                    
                    # Match quiz scores to calculate learning gains
                    if df_q_filtered.empty or len(df_q_filtered[df_q_filtered["quiz_type"] == "PRE"]) == 0 or len(df_q_filtered[df_q_filtered["quiz_type"] == "POST"]) == 0:
                        st.warning("Insufficient pre-test or post-test quiz results to compile learning gains.")
                    else:
                        # Self join to get PRE and POST matching rows
                        pre_df = df_q_filtered[df_q_filtered["quiz_type"] == "PRE"]
                        post_df = df_q_filtered[df_q_filtered["quiz_type"] == "POST"]
                        
                        merged_q = pd.merge(
                            pre_df, post_df, 
                            on=["email", "topic_name"], 
                            suffixes=("_pre", "_post")
                        )
                        
                        if not df_s_filtered.empty and not merged_q.empty:
                            # Clean session names to match topic names in quizzes
                            merged_q = pd.merge(
                                merged_q, 
                                df_s_filtered[["email", "topic_name", "study_mode"]].drop_duplicates(), 
                                on=["email", "topic_name"]
                            )
                            merged_q["learning_gain"] = merged_q["score_post"] - merged_q["score_pre"]
                        else:
                            merged_q = pd.DataFrame()
                        
                        st.markdown("### 📊 Compiled Study Results")
                        
                        # Section 1: Learning gains
                        st.markdown("#### 1. Learning Gains (Post-Test minus Pre-Test)")
                        if merged_q.empty:
                            st.caption("No matched learning gain pairs found.")
                        else:
                            gains_grouped = merged_q.groupby("study_mode")["learning_gain"].agg(["mean", "std", "count"]).round(2)
                            st.dataframe(gains_grouped, use_container_width=True)
                            
                            # Compute Paired t-test
                            sentio_gains = merged_q[merged_q["study_mode"] == "SENTIO"]["learning_gain"].values
                            control_gains = merged_q[merged_q["study_mode"] == "CONTROL"]["learning_gain"].values
                            
                            n_pairs = min(len(sentio_gains), len(control_gains))
                            if n_pairs > 1:
                                s_vals = sentio_gains[:n_pairs]
                                c_vals = control_gains[:n_pairs]
                                diff = s_vals - c_vals
                                mean_diff = np.mean(diff)
                                std_diff = np.std(diff, ddof=1)
                                se_diff = std_diff / math.sqrt(n_pairs)
                                
                                if se_diff > 0:
                                    t_stat = mean_diff / se_diff
                                    df_deg = n_pairs - 1
                                    
                                    # high-precision standard cumulative t-distribution approximation
                                    d = abs(t_stat)
                                    a = 1.0 / (1.0 + 0.196854 * d + 0.115194 * d**2 + 0.000344 * d**3 + 0.019527 * d**4)
                                    p_val = min(1.0, max(0.0, 0.5 * (a**4) * 2))
                                    
                                    col_t, col_p = st.columns(2)
                                    with col_t:
                                        st.metric("t-statistic", f"{t_stat:.3f}", help=f"Degrees of freedom: {df_deg}")
                                    with col_p:
                                        sig_icon = "🟢 Significant (p < 0.05)" if p_val < 0.05 else "⚪ Not Significant"
                                        st.metric("p-value", f"{p_val:.5f}", delta=sig_icon, delta_color="normal" if p_val < 0.05 else "inverse")
                                else:
                                    st.info("Zero variance. T-test skipped.")
                            else:
                                st.info("Additional paired sessions needed to compute paired t-test statistics.")
                                
                        # Section 2: NASA-TLX workload
                        st.markdown("#### 2. NASA-TLX Subjective Workload Comparison")
                        if df_w_filtered.empty:
                            st.caption("No NASA-TLX responses logged yet.")
                        else:
                            w_grouped = df_w_filtered.groupby("study_mode")[["mental_demand", "temporal_demand", "performance", "effort", "frustration"]].mean().round(1)
                            st.dataframe(w_grouped, use_container_width=True)
                            
                            # Simple Welch's t-test for Frustration
                            s_frust = df_w_filtered[df_w_filtered["study_mode"] == "SENTIO"]["frustration"].values
                            c_frust = df_w_filtered[df_w_filtered["study_mode"] == "CONTROL"]["frustration"].values
                            if len(s_frust) > 1 and len(c_frust) > 1:
                                mean_s, mean_c = np.mean(s_frust), np.mean(c_frust)
                                var_s, var_c = np.var(s_frust, ddof=1), np.var(c_frust, ddof=1)
                                n_s, n_c = len(s_frust), len(c_frust)
                                
                                se = math.sqrt((var_s / n_s) + (var_c / n_c))
                                if se > 0:
                                    t_val = (mean_s - mean_c) / se
                                    welch_df = n_s + n_c - 2
                                    d = abs(t_val)
                                    a = 1.0 / (1.0 + 0.196854 * d + 0.115194 * d**2 + 0.000344 * d**3 + 0.019527 * d**4)
                                    p_val = min(1.0, max(0.0, 0.5 * (a**4) * 2))
                                    
                                    st.markdown(f"**Frustration Welch t-test**: $t({welch_df}) = {t_val:.3f}$, $p = {p_val:.5f}$ " + ("(Significant)" if p_val < 0.05 else ""))
                                    
                        # Section 3: Telemetry profiles
                        st.markdown("#### 3. Keystroke Dynamics Telemetry Comparison")
                        if df_telemetry.empty or df_sessions.empty:
                            st.caption("No typing telemetry logged yet.")
                        else:
                            merged_telemetry = pd.merge(
                                df_telemetry, 
                                df_sessions[["session_id", "study_mode"]], 
                                on="session_id"
                            )
                            if not merged_telemetry.empty:
                                telemetry_grouped = merged_telemetry.groupby("study_mode")[["backspace_count", "avg_dwell_ms", "avg_flight_ms", "pause_seconds"]].mean().round(2)
                                st.dataframe(telemetry_grouped, use_container_width=True)
                                
                                # Render small bar charts directly in panel
                                st.markdown("##### Backspace Rate Comparison")
                                st.bar_chart(telemetry_grouped["backspace_count"])
                                st.markdown("##### Flight Time (ms) Comparison")
                                st.bar_chart(telemetry_grouped["avg_flight_ms"])
