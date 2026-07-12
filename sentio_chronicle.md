# The Sentio Project Chronicle
*A Living Record of Architecture, Science, and Development History*

**Sentio** (formerly **CogniFlow**) is an adaptive AI tutoring system designed to align human cognitive limits with AI instruction. By passively monitoring user typing behavior and timing parameters, the system infers cognitive load in real-time, adjusts LLM response complexity, and uses spaced repetition (Ebbinghaus Forgetting Curve) to proactively re-ground the user on fading concepts.

---

## 1. Core Architecture Overview

Sentio is built modularly with a separation of concern between frontend timing, load classification, agent routing, and spaced retrieval memory:

```
Sentio/
├── app.py                  # Entry point: Streamlit UI, provider configuration, and state orchestrator
├── core/
│   └── load_detector.py    # Classifies user cognitive state into OVERLOADED, OPTIMAL, or UNDERLOADED
├── agent/
│   └── graph.py            # LangGraph routing state machine using generic BaseChatModel
├── memory/
│   └── store.py            # ChromaDB interface + Ebbinghaus Spaced Repetition logic
├── ui/
│   └── timing.py           # JavaScript telemetry bridge monitoring focus and submit transitions
└── tests.py                # Component test suite (executes timing, memory, and load classification)
```

---

## 2. Scientific Foundations

Sentio integrates three established psychological and computer science methodologies:

### A. Cognitive Load Theory (CLT)
* **Foundation**: Established by John Sweller (1988), CLT asserts that humans possess a highly constrained working memory capacity. 
* **Sentio Application**: The system monitors features like writing hesitation pauses and text complexity. If cognitive capacity is overloaded, Sentio dynamically simplifies grammar, filters jargon, provides analogies, and restricts information flow to a single core idea at a time.

### B. Spaced Repetition (Ebbinghaus Forgetting Curve)
* **Foundation**: Hermann Ebbinghaus discovered that memory retention decays exponentially over time without reinforcement:

$$R = e^{-\frac{t}{S}}$$

* **Sentio Application**: Every concept explained by the tutor is logged in ChromaDB. If the user's estimated retention ($R$) drops below a configured threshold ($70\%$), a memory re-grounding instruction is prepended to the system prompt, causing the LLM to seamlessly weave in a review of the forgotten topic.

### C. Keystroke Dynamics (KSD) & Affective Computing
Instead of requiring invasive physiological sensors (like EEG or facial tracking cameras), typing dynamics are used as a non-intrusive window into human state. Key literature backing this includes:
1. **Stress and Cognitive Strain**: *Vizer, Zhou, and Sears (2009)* demonstrated that cognitive workload changes user keystroke rhythms and causes highly irregular transition times.
2. **Emotion and Frustration**: *Epp, Lippold, and Mandryk (2011)* proved that negative affect (like frustration or lack of confidence) causes erratic cadence, spiked backspace usage, and irregular dwell times, allowing emotions to be classified with 80%+ accuracy.
3. **Engagement and Flow**: *Bixler and D'Mello (2013)* demonstrated that lexical transitions and pause distributions indicate whether a user is in a state of engagement, flow, or boredom.

#### Telemetry Threshold Justification
The mathematical thresholds used in Sentio's load classifier are drawn directly from empirical findings in typing rhythm studies:

| Telemetry Metric | Default Baseline (Flow State) | Overload Threshold | Underload Threshold | Scientific Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Flight Time (Transition Latency)** | `150ms` - `250ms` | `> 350ms` | `< 150ms` | When working memory is loaded, linguistic planning creates hesitations between key presses. Rapid transitions (<150ms) reflect low-friction flow. |
| **Dwell Time (Key Hold Duration)** | `80ms` - `120ms` | `> 150ms` | `< 80ms` | High hold times indicate physical drag, stress-induced muscle tension, or cognitive fatigue. Fast taps (<80ms) indicate confident execution. |
| **Backspace/Deletion Rate** | `5%` - `8%` | `> 15%` | — | High backspace rates (>15% of character count) indicate severe semantic/syntactic editing loops, signifying confusion or struggle. |

---

## 3. Chronicle of Changes
 
### Phase 1: Rebranding & API Generalization (Completed)
* **Rebranding**: Rebranded the codebase from **CogniFlow** to **Sentio** across print statements, document titles, database persist directories, and Javascript postMessage event names.
* **OpenAI to Multi-Provider Migration**: 
  * The codebase was in a broken state, half-migrated between OpenAI and Gemini.
  * Generalised [`agent/graph.py`](file:///c:/Users/DELL/Desktop/projects/cogniflow/agent/graph.py) and [`memory/store.py`](file:///c:/Users/DELL/Desktop/projects/cogniflow/memory/store.py) to accept a generic LangChain `BaseChatModel` object. Both response generation and topic extraction now run dynamically on the same model definition.
  * Added dynamic dropdown selectors in the Streamlit Sidebar of [`app.py`](file:///c:/Users/DELL/Desktop/projects/cogniflow/app.py) allowing users to select their model provider (**Gemini**, **OpenAI**, **Anthropic**, **Groq**), select default models, input custom model strings, and enter custom API keys.
* **Groq Integration**: Integrated the user's custom Groq API key directly into the project's [`.env`](file:///c:/Users/DELL/Desktop/projects/cogniflow/.env) file under `GROQ_API_KEY`.
* **Testing & Verification**: Verified that all **19 unit tests pass successfully** under UTF-8 console configurations. Run manual checks using an automated browser subagent to verify the Streamlit frontend compiles and operates without console errors.

### Phase 2: Keystroke Telemetry & Usability Loop Mitigation (Completed)
* **Keystroke Dynamics Logging**: 
  * Refactored [`ui/timing.py`](file:///c:/Users/DELL/Desktop/projects/cogniflow/ui/timing.py) to log keyboard event listeners (`keydown` and `keyup`).
  * Calculates key transition flight time (Average Flight), key hold duration (Average Dwell), and backspace deletion rates.
  * *Project Benefit*: Replaced raw reading pause duration (which includes healthy reading time) with direct, objective motor signals that separate cognitive load from simple text intake.
* **React input Serialization Bridge**:
  * Implemented a hidden text widget and a Javascript prototype-descriptor value setter bypass.
  * *Project Benefit*: Bypasses React's internal value state tracking to force instantaneous, silent synchronization of telemetry metrics from the browser sandbox to Streamlit when messages are sent.
* **Usability Loop Mitigations**:
  * Refactored [`core/load_detector.py`](file:///c:/Users/DELL/Desktop/projects/cogniflow/core/load_detector.py) to skip message-length penalties for common one-word answers or menu options (e.g. `everything`, `yes`, `setup`).
  * Implemented telemetry offsets where confident typing rhythms directly mitigate reading-pause penalties.
  * Added explicit overrides where direct requests for comprehensive information (e.g. "everything") force-promote the conversation out of the `OVERLOADED` state.
  * *Project Benefit*: Resolved the critical "Brevity Loop" UX bug. The agent no longer enters infinite loops of shortened explanations and choices when a user asks to learn "everything".
* **Refined OVERLOADED System Prompt**:
  * Rewrote the system instructions to focus on simple vocabulary, conceptual scaffolding, real-world analogies, and natural conversational flow rather than arbitrary sentence-length limits.
  * *Project Benefit*: Allows detailed, elaborate explanations to be delivered naturally when requested, ensuring simplicity of concepts is never conflated with lack of depth.
* **Groq Model Correction**:
  * Replaced decommissioned Groq model `llama-3.3-70b-specdec` with the currently active `llama-3.3-70b-versatile`.
  * *Project Benefit*: Restores API calling stability for users on free tiers.

---

## 4. Development Roadmap

We are tracking the following future modifications to be implemented and maintained in this chronicle:

```
[Phase 3: Active SRS Practice] ──> [Phase 4: Scaffolded Routing]
```

* **Phase 3: Active SRS Practice**
  * Transition from passive memory grounding to active concept reviews. 
  * Generate dynamic retrieval practice (e.g., small diagnostic quiz questions) when the user is in a focused optimal flow state.
* **Phase 4: Scaffolded Routing**
  * Expand LangGraph conditional nodes to support structurally different teaching modes: Socratic questioning for the advanced (Underloaded), Direct instruction for the Optimal, and Scaffolding/Worked Examples for the Overloaded.
