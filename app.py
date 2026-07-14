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
    # Render existing chat history in a custom styled chat container
    st.markdown('<div class="sentio-chat-container">', unsafe_allow_html=True)
    for turn in st.session_state["chat_history"]:
        render_chat_bubble(
            role=turn["role"],
            content=turn["content"],
            load_state=turn.get("load_state", "OPTIMAL"),
            srs_evaluation=turn.get("srs_evaluation", ""),
            show_load_badge=show_load_badge
        )
    st.markdown('</div>', unsafe_allow_html=True)

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

    # 3. Query agent inside tutor tab
    with tab_tutor:
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
