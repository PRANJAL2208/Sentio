# Walkthrough — Keystroke Telemetry & Sentio Core Refactoring

We have successfully implemented fine-grained keystroke telemetry monitoring, integrated it into the cognitive load detector, rebranded the remaining memory directory, and resolved the decommissioned Groq model setting.

---

## 1. Summary of Changes

### Directory Rebranding
* **Folder Rename**: Renamed the vector database persistence folder `cogniflow_memory` to `sentio_memory` at the workspace root to finalize the rebranding effort.

### Keystroke Telemetry Collection
* **[ui/timing.py](file:///c:/Users/DELL/Desktop/projects/cogniflow/ui/timing.py)**:
  * Refactored the embedded JS component inside `TIMING_COMPONENT_HTML` to track keyboard events (`keydown` and `keyup`).
  * Calculates key hold duration (**Average Dwell Time**) and key transition latency (**Average Flight Time**).
  * Counts **Backspace deletions** while the user types in the Streamlit text area.
  * Synchronizes this structured data via a DOM input bridge to Streamlit parent frame's `telemetry_data` text input widget.
  * Replaced `get_pause_seconds` with `get_typing_telemetry` to load and parse JSON telemetry state with elegant fallback options.

### Cognitive Load Classification
* **[core/load_detector.py](file:///c:/Users/DELL/Desktop/projects/cogniflow/core/load_detector.py)**:
  * Modified `classify_load()` to evaluate the new telemetry signals.
  * **Overloaded State Rule Extensions**: Adds `+0.3` to the overload score if average flight time exceeds `350ms`, `+0.2` if average dwell time exceeds `150ms`, and `+0.3` if backspaces make up more than `15%` of the typed character length.
  * **Underloaded State Rule Extensions**: Adds `+0.3` to the underload score if average flight time is between `0ms` and `150ms` and `+0.2` if average dwell time is between `0ms` and `80ms`.

### Agent State & Front-End UI
* **[agent/graph.py](file:///c:/Users/DELL/Desktop/projects/cogniflow/agent/graph.py)**: Updated `AgentState` schema and `run_agent()` function to accept, pass, and return the new telemetry metrics.
* **[app.py](file:///c:/Users/DELL/Desktop/projects/cogniflow/app.py)**:
  * Integrated a hidden text input component targeting `telemetry_data` to serialize telemetry state from Javascript to Streamlit.
  * Injected custom CSS rules to hide the telemetry input container from view.
  * Parsed telemetry state during submissions and forwarded it to the LangGraph runner.
  * Expanded the "Raw signals" panel in the sidebar to dynamically display Average Flight, Average Dwell, and Backspaces when present.
  * Replaced decommissioned Groq model `llama-3.3-70b-specdec` with `llama-3.3-70b-versatile` in the sidebar select dropdown list.

### Cognitive Load Mitigations
* **[core/load_detector.py](file:///c:/Users/DELL/Desktop/projects/cogniflow/core/load_detector.py)**:
  * Added **Menu Option Filter**: Skip length penalties for common one-word options (e.g. "everything", "benefits", "setup") to avoid false-positive overload flags.
  * Added **Telemetry Mitigation**: If typing is fluent (e.g., flight time is under `250ms` and backspaces are `0`), mitigate the reading pause penalty by subtracting up to `0.4` from the overload score.
  * Added **Explicit Override**: Explicit requests for comprehensive information (e.g., typing "everything" or "all") force clear the overload flag and boost the underload score, ensuring the system returns a detailed answer instead of trapping the user in simplified loops.
  * Added **Refined OVERLOADED System Prompt**: Redesigned the instructions to explain complex topics step-by-step using simple vocabulary rather than strictly capping length, allowing detailed responses to flow naturally when explicitly requested by the user.

---

## 2. Verification Results

### Automated Tests
* We updated [tests.py](file:///c:/Users/DELL/Desktop/projects/cogniflow/tests.py) to assert telemetry mitigations and explicit override behaviors.
* Running the test suite shows all **24 assertions pass successfully**:
  ```
  Sentio — Component Tests
  ============================================================
  [1] core/load_detector.py
    ✓  Long pause + short confused message → OVERLOADED
    ✓  Medium pause + clear question → OPTIMAL
    ✓  Fast + long engaged message → UNDERLOADED
    ✓  High transition latency (flight time) → OVERLOADED
    ✓  High backspace rate (>15%) → OVERLOADED
    ✓  Fast flight and dwell times → UNDERLOADED
    ✓  Confident telemetry offsets high reading pause → OPTIMAL
    ✓  Explicit 'everything' request overrides overload state
    ...
    Results: 24 passed, 0 failed
    All tests passed. Ready to run: streamlit run app.py
  ```
