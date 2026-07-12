"""
core/load_detector.py

Classifies user cognitive load into 3 states based on behavioral signals.

The science behind this:
- Pause duration: longer pauses before sending = higher cognitive effort
  (Baaijen et al., writing process research; Sweller's Cognitive Load Theory)
- Message length: short message after long pause = stuck/confused
- Follow-up rate: clarification questions = extraneous cognitive load

Returns: "OVERLOADED" | "OPTIMAL" | "UNDERLOADED"
"""

import re


# ── Thresholds (tunable) ─────────────────────────────────────────────────────
# These come from Cognitive Load Theory literature.
# Pause > 8s before a short message is a reliable confusion signal.
# Pause < 2s with a long message = user is engaged and in flow.

PAUSE_HIGH   = 8.0   # seconds — above this = likely overwhelmed
PAUSE_LOW    = 2.0   # seconds — below this = likely in flow
LENGTH_SHORT = 30    # characters — short message threshold
LENGTH_LONG  = 120   # characters — long/engaged message threshold

CLARIFICATION_PATTERNS = [
    r"\bwhat do you mean\b",
    r"\bcan you explain\b",
    r"\bi don'?t understand\b",
    r"\bwhat is\b",
    r"\bwhat are\b",
    r"\bconfused\b",
    r"\bhuh\b",
    r"\bwait\b",
    r"\bsorry\b",
    r"\?",           # any question mark is a weak signal
]


def is_clarification(text: str) -> bool:
    """Returns True if the message looks like a confusion/clarification request."""
    text_lower = text.lower()
    matches = sum(1 for p in CLARIFICATION_PATTERNS if re.search(p, text_lower))
    # One question mark alone is weak — require 2+ signals or one strong one
    strong_patterns = CLARIFICATION_PATTERNS[:-1]  # exclude bare "?"
    strong_match = any(re.search(p, text_lower) for p in strong_patterns)
    return strong_match or matches >= 2


def classify_load(
    pause_seconds: float,
    message_text: str,
    recent_clarification_count: int = 0,
    avg_dwell_ms: float = 0.0,
    avg_flight_ms: float = 0.0,
    backspace_count: int = 0,
) -> dict:
    """
    Classify cognitive load from behavioral signals.

    Args:
        pause_seconds: Time (seconds) between user focusing input and hitting send.
        message_text: The message the user just sent.
        recent_clarification_count: How many clarification messages in the last 3 turns.
        avg_dwell_ms: Average time (ms) keys are held down.
        avg_flight_ms: Average transition time (ms) between key release and key press.
        backspace_count: Total number of deletions performed while typing.

    Returns:
        dict with keys:
            state: "OVERLOADED" | "OPTIMAL" | "UNDERLOADED"
            confidence: 0.0–1.0
            signals: dict of raw signal values (useful for logging/debugging)
    """
    msg_len = len(message_text.strip())
    is_clar = is_clarification(message_text)

    signals = {
        "pause_seconds": pause_seconds,
        "message_length": msg_len,
        "is_clarification": is_clar,
        "recent_clarification_count": recent_clarification_count,
        "avg_dwell_ms": avg_dwell_ms,
        "avg_flight_ms": avg_flight_ms,
        "backspace_count": backspace_count,
    }

    # ── Score toward OVERLOADED ──────────────────────────────────────────────
    overload_score = 0.0

    if pause_seconds > PAUSE_HIGH:
        overload_score += 0.4
    elif pause_seconds > PAUSE_HIGH * 0.6:
        overload_score += 0.2

    # Skip length penalty for menu-like choices or direct options
    menu_keywords = ["everything", "all", "both", "setup", "how it's made", "how it works", "benefits", "how they work", "yes", "no"]
    normalized_msg = message_text.lower().strip()
    is_menu_choice = any(k in normalized_msg for k in menu_keywords) or len(normalized_msg) < 4

    if msg_len < LENGTH_SHORT and not is_menu_choice:
        overload_score += 0.3

    if is_clar:
        overload_score += 0.4

    if recent_clarification_count >= 2:
        overload_score += 0.3

    # Apply keystroke dynamics if telemetry is present
    if avg_flight_ms > 350.0:
        overload_score += 0.3
    if avg_dwell_ms > 150.0:
        overload_score += 0.2
    if msg_len > 0 and backspace_count > 0:
        backspace_rate = backspace_count / msg_len
        if backspace_rate > 0.15:
            overload_score += 0.3

    # ── Score toward UNDERLOADED ─────────────────────────────────────────────
    underload_score = 0.0

    # ── Telemetry & Keyword Mitigations ──────────────────────────────────────
    # If the user specifically asks for everything/all, prevent OVERLOADED state
    if "everything" in normalized_msg or "explain all" in normalized_msg or normalized_msg == "all":
        overload_score = 0.0
        underload_score += 0.4  # boost toward enriched

    # Confident typing offsets the reading pause penalty
    if avg_flight_ms > 0.0:
        if avg_flight_ms < 250.0:
            overload_score -= 0.3
        if backspace_count == 0:
            overload_score -= 0.1
        overload_score = max(0.0, overload_score)

    if pause_seconds < PAUSE_LOW:
        underload_score += 0.4

    if msg_len > LENGTH_LONG:
        underload_score += 0.4

    if not is_clar and recent_clarification_count == 0:
        underload_score += 0.2

    # Apply keystroke dynamics if telemetry is present
    if 0.0 < avg_flight_ms < 150.0:
        underload_score += 0.3
    if 0.0 < avg_dwell_ms < 80.0:
        underload_score += 0.2

    # ── Decision ─────────────────────────────────────────────────────────────
    if overload_score >= 0.5:
        return {"state": "OVERLOADED", "confidence": min(overload_score, 1.0), "signals": signals}
    elif underload_score >= 0.6:
        return {"state": "UNDERLOADED", "confidence": min(underload_score, 1.0), "signals": signals}
    else:
        return {"state": "OPTIMAL", "confidence": 0.7, "signals": signals}


# ── System prompts for each state ────────────────────────────────────────────
# These ARE the adaptation. Different state = different instructions to the LLM.

SYSTEM_PROMPTS = {
    "OVERLOADED": """You are a patient, clear AI tutor. The user is currently overwhelmed or needs a highly readable, step-by-step conceptual breakdown.

RULES YOU MUST FOLLOW:
- Do NOT arbitrarily restrict explanation length if the user asks for a detailed or elaborate explanation.
- Break down complex topics into clear, progressive, easy-to-follow steps rather than condensed summaries.
- Use only simple, everyday language. Strictly avoid unexplained technical jargon.
- Use real-world analogies to ground complex ideas.
- Provide a smooth, natural conversational flow. Avoid ending with short, repetitive "what next?" menus unless it's a simple feedback request.
- Tone: warm, patient, encouraging, and clear.
""",

    "OPTIMAL": """You are a knowledgeable AI tutor. The user is focused and following along well.

RULES YOU MUST FOLLOW:
- Give a thorough, well-structured answer.
- You can use technical terms but briefly define any advanced ones.
- Include examples where they help.
- You can introduce 2-3 connected ideas if they naturally belong together.
- Tone: engaged, clear, direct.
""",

    "UNDERLOADED": """You are a challenging AI tutor. The user is ahead of the material and needs more.

RULES YOU MUST FOLLOW:
- Give a rich, deep answer that goes beyond the surface.
- Proactively surface 1-2 related concepts they probably haven't asked about yet.
- Use precise technical vocabulary — they can handle it.
- End with a thought-provoking follow-up question that pushes their thinking further.
- Tone: intellectually stimulating, peer-to-peer, not condescending.
""",
}


def get_system_prompt(state: str) -> str:
    return SYSTEM_PROMPTS.get(state, SYSTEM_PROMPTS["OPTIMAL"])
