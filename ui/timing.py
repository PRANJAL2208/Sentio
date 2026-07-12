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


# HTML + JS component that captures keystroke dynamics
TIMING_COMPONENT_HTML = """
<script>
(function() {
    let focusTime = null;
    let lastKeyUpTime = null;
    let dwellTimes = [];
    let flightTimes = [];
    let backspaceCount = 0;
    let keyPressTimes = {};

    function getTelemetry() {
        const totalPause = focusTime ? (Date.now() - focusTime) / 1000.0 : 0;
        const avgDwell = dwellTimes.length > 0 ? (dwellTimes.reduce((a, b) => a + b, 0) / dwellTimes.length) : 0;
        const avgFlight = flightTimes.length > 0 ? (flightTimes.reduce((a, b) => a + b, 0) / flightTimes.length) : 0;
        return {
            pause_seconds: totalPause,
            avg_dwell_ms: avgDwell,
            avg_flight_ms: avgFlight,
            backspace_count: backspaceCount
        };
    }

    function syncTelemetry() {
        try {
            const telemetry = getTelemetry();
            const parentDoc = window.parent.document;
            const hiddenInput = parentDoc.querySelector('input[aria-label="telemetry_data"]');
            if (hiddenInput) {
                const valStr = JSON.stringify(telemetry);
                const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
                nativeSetter.call(hiddenInput, valStr);
                const event = new Event('input', { bubbles: true });
                hiddenInput.dispatchEvent(event);
            }
        } catch (e) {
            console.error("Telemetry sync error:", e);
        }
    }

    function attachListeners() {
        const input = document.querySelector('textarea[data-testid="stChatInputTextArea"]')
                   || document.querySelector('textarea');

        if (!input) {
            setTimeout(attachListeners, 300);
            return;
        }

        // Record when user focuses (starts to think about typing)
        input.addEventListener('focus', function() {
            if (!focusTime) {
                focusTime = Date.now();
                lastKeyUpTime = null;
                dwellTimes = [];
                flightTimes = [];
                backspaceCount = 0;
                keyPressTimes = {};
            }
        }, { once: false });

        input.addEventListener('keydown', function(e) {
            if (!focusTime) focusTime = Date.now();
            
            const key = e.key;
            if (key.length === 1 || key === 'Backspace') {
                if (key === 'Backspace') {
                    backspaceCount++;
                }
                
                // Track flight time
                if (lastKeyUpTime) {
                    const flight = Date.now() - lastKeyUpTime;
                    flightTimes.push(flight);
                }
                
                // Track dwell start
                if (!keyPressTimes[key]) {
                    keyPressTimes[key] = Date.now();
                }
            }
        });

        input.addEventListener('keyup', function(e) {
            const key = e.key;
            if (key.length === 1 || key === 'Backspace') {
                // Track dwell end
                if (keyPressTimes[key]) {
                    const dwell = Date.now() - keyPressTimes[key];
                    dwellTimes.push(dwell);
                    delete keyPressTimes[key];
                }
                lastKeyUpTime = Date.now();
                syncTelemetry();
            }
        });

        // Form submit integration
        input.closest('form')?.addEventListener('submit', function() {
            syncTelemetry();
            focusTime = null; // Reset for next turn
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
import json

def get_typing_telemetry(session_state, fallback_key: str = "_last_interaction_time") -> dict:
    """
    Retrieve typing telemetry. Primary: JS-captured JSON from st.session_state["telemetry_input"].
    Fallback: estimate pause time from rerun interval.
    """
    raw_telemetry = session_state.get("telemetry_input", "")
    
    if raw_telemetry:
        try:
            data = json.loads(raw_telemetry)
            # Reset it in session_state so it doesn't leak to the next turn
            session_state["telemetry_input"] = ""
            return {
                "pause_seconds": float(data.get("pause_seconds", 0.0)),
                "avg_dwell_ms": float(data.get("avg_dwell_ms", 0.0)),
                "avg_flight_ms": float(data.get("avg_flight_ms", 0.0)),
                "backspace_count": int(data.get("backspace_count", 0)),
            }
        except Exception:
            pass

    # Fallback timing
    now = time.time()
    last = session_state.get(fallback_key, now)
    pause = now - last
    session_state[fallback_key] = now
    
    return {
        "pause_seconds": min(pause, 120.0),
        "avg_dwell_ms": 0.0,
        "avg_flight_ms": 0.0,
        "backspace_count": 0,
    }


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
