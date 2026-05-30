"""
ui/timing.py

Streamlit custom component that captures typing pause duration.

How it works:
- JavaScript in the browser records when the user clicks/focuses the chat input
- When the user sends a message, it calculates seconds elapsed
- Sends that value back to Python via Streamlit's component communication

This is the bridge between browser behavior and the Python classifier.
"""

import streamlit.components.v1 as components


# HTML + JS component that captures focus → send timing
TIMING_COMPONENT_HTML = """
<script>
(function() {
    let focusTime = null;
    let pauseSeconds = 0;

    // Poll for the Streamlit chat input element (it renders async)
    function attachListeners() {
        const input = document.querySelector('textarea[data-testid="stChatInputTextArea"]')
                   || document.querySelector('textarea');

        if (!input) {
            setTimeout(attachListeners, 300);
            return;
        }

        // Record when user focuses (starts to think about typing)
        input.addEventListener('focus', function() {
            focusTime = Date.now();
        }, { once: false });

        // When user submits, calculate pause and send to Streamlit
        input.closest('form')?.addEventListener('submit', function() {
            if (focusTime) {
                pauseSeconds = (Date.now() - focusTime) / 1000.0;
                focusTime = null;

                // Send to Streamlit via postMessage
                window.parent.postMessage({
                    type: 'cogniflow_pause',
                    pause_seconds: pauseSeconds
                }, '*');
            }
        });

        // Also listen for Enter key (Streamlit chat submits on Enter)
        input.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                if (focusTime) {
                    pauseSeconds = (Date.now() - focusTime) / 1000.0;
                    focusTime = null;
                    window.parent.postMessage({
                        type: 'cogniflow_pause',
                        pause_seconds: pauseSeconds
                    }, '*');
                }
            }
        });
    }

    // Start polling after DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', attachListeners);
    } else {
        attachListeners();
    }
})();
</script>
"""


def inject_timing_tracker():
    """
    Inject the JS timing tracker into the Streamlit page.
    Call this once at the top of app.py.
    The pause value is stored in st.session_state by the receiver below.
    """
    components.html(TIMING_COMPONENT_HTML, height=0, scrolling=False)


# ── Fallback: estimate pause from Streamlit's own timing ─────────────────────
# The JS component is the primary method.
# If JS communication fails (e.g. sandboxed environments), use this fallback
# which estimates pause from the time between Streamlit reruns.

import time

def get_pause_seconds(session_state, fallback_key: str = "_last_interaction_time") -> float:
    """
    Get pause duration. Primary: JS-captured value in session_state.
    Fallback: time since last Streamlit rerun.

    Call this at the start of each message handler.
    """
    # Check if JS component sent a value
    js_pause = session_state.get("js_pause_seconds", None)
    if js_pause is not None and js_pause > 0:
        # Clear it so it doesn't persist to next turn
        session_state["js_pause_seconds"] = None
        return float(js_pause)

    # Fallback: time between reruns (less accurate but always works)
    now = time.time()
    last = session_state.get(fallback_key, now)
    pause = now - last
    session_state[fallback_key] = now

    # Cap at 120s (if they walked away and came back, that's not "confused")
    return min(pause, 120.0)


def update_clarification_count(session_state, user_message: str):
    """
    Track how many of the last 3 messages were clarification requests.
    Updates session_state["clarification_window"] (a list of bools).
    """
    from core.load_detector import is_clarification

    window = session_state.get("clarification_window", [])
    window.append(is_clarification(user_message))

    # Keep only last 3 turns
    if len(window) > 3:
        window = window[-3:]

    session_state["clarification_window"] = window
    return sum(window)  # count of True values = clarification count
