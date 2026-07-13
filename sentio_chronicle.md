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

* **Sentio Application**: Every concept explained by the tutor is logged in ChromaDB. If the user's estimated retention ($R$) drops below a configured threshold ($70\%$), a memory recall prompt is generated.

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
  * Generalised [`agent/graph.py`](file:///c:/Users/DELL/Desktop/projects/cogniflow/agent/graph.py) and [`memory/store.py`](file:///c:/Users/DELL/Desktop/projects/cogniflow/memory/store.py) to accept a generic LangChain `BaseChatModel` object. Both response generation and topic extraction now run dynamically on the same model definition.
  * Added dynamic dropdown selectors in the Streamlit Sidebar of [`app.py`](file:///c:/Users/DELL/Desktop/projects/cogniflow/app.py) allowing users to select their model provider (**Gemini**, **OpenAI**, **Anthropic**, **Groq**), select default models, input custom model strings, and enter custom API keys.
* **Groq Integration**: Integrated the user's custom Groq API key directly into the project's [`.env`](file:///c:/Users/DELL/Desktop/projects/cogniflow/.env) file under `GROQ_API_KEY`.
* **Testing & Verification**: Verified that all **19 unit tests pass successfully** under UTF-8 console configurations.

### Phase 2: Keystroke Telemetry & Usability Loop Mitigation (Completed)
* **Keystroke Dynamics Logging**: 
  * Refactored [`ui/timing.py`](file:///c:/Users/DELL/Desktop/projects/cogniflow/ui/timing.py) to log keyboard event listeners (`keydown` and `keyup`).
  * Calculates key transition flight time (Average Flight), key hold duration (Average Dwell), and backspace deletion rates.
* **React input Serialization Bridge**:
  * Implemented a hidden text widget and a Javascript prototype-descriptor value setter bypass in [`app.py`](file:///c:/Users/DELL/Desktop/projects/cogniflow/app.py) to bridge client telemetry into Streamlit.
* **Usability Loop Mitigations**:
  * Refactored [`core/load_detector.py`](file:///c:/Users/DELL/Desktop/projects/cogniflow/core/load_detector.py) to skip message-length penalties for common one-word answers or menu options (e.g. `everything`, `yes`, `setup`).
  * Implemented telemetry offsets where confident typing rhythms directly mitigate reading-pause penalties.
  * Added explicit overrides where direct requests for comprehensive information (e.g. "everything") force-promote the conversation out of the `OVERLOADED` state.
* **Refined OVERLOADED System Prompt**:
  * Rewrote the system instructions to focus on simple vocabulary, conceptual scaffolding, real-world analogies, and natural conversational flow rather than arbitrary sentence-length limits.
* **Groq Model Correction**:
  * Replaced decommissioned Groq model `llama-3.3-70b-specdec` with the currently active `llama-3.3-70b-versatile`.

### Phase 3 & 4: Scaffolded Routing & Active SRS Practice (Completed)
* **Closed-Loop Active Recall**:
  * Implemented `evaluate_user_answer()` and `update_topic_stability()` in [`memory/store.py`](file:///c:/Users/DELL/Desktop/projects/cogniflow/memory/store.py) to grade active recall quiz answers using the LLM.
  * Correct answers double memory stability (`x2.0`), partial responses increase it slightly (`x1.2`), and incorrect responses reset stability to baseline (`1.0`) to queue the topic for immediate re-learning.
* **Scaffolded Routing Nodes**:
  * Expanded `AgentState` in [`agent/graph.py`](file:///c:/Users/DELL/Desktop/projects/cogniflow/agent/graph.py) to manage active quiz sessions.
  * Designed structurally distinct teaching nodes in LangGraph:
    * `generate_overloaded`: Scaffolds explanations with simple vocabulary and worked examples, ending with a simple check-in question.
    * `generate_underloaded`: Uses a Socratic approach to ask leading questions and prompt the user to discover solutions on their own.
    * `generate_optimal`: Standard direct instruction, with automatic interception to generate active recall quiz questions when forgotten concepts are found. It uses a **Spacing Cooldown** (minimum of 4 turns between quizzes) and checks for active **Clarification requests** to prevent disrupting natural follow-up conversations.
    * `evaluate_quiz`: Evaluates answers, writes updates to ChromaDB, prints feedback, and returns control to standard tutoring.
* **UI Integration**:
  * Integrated a visual feedback card in [`app.py`](file:///c:/Users/DELL/Desktop/projects/cogniflow/app.py) showing SRS results.
  * Bypassed topic extraction on quiz interactions to prevent administrative text from polluting ChromaDB.
* **Expanded Tests**:
  * Added unit test assertions checking recall grading accuracy, Ebbinghaus stability calculations, spacing cooldown, clarification suppression, and graph routing transitions. All **38 tests pass successfully**.

---

## 5. Development Diary (Design Comparisons & Analytics Expansion)

### A. Graph Visualization Choice: Graphviz vs. Streamlit-Agraph vs. Pyvis
When planning the visual Concept Map visualization, we compared three main technical paths:
1. **Streamlit-Agraph (React wrapper)**:
   * *Pros*: Supports interactive draggable nodes, zoom, and custom CSS styling.
   * *Cons*: Requires separate package installations (`pip install streamlit-agraph`), which has dependency resolution errors on some lockfiles and requires custom server-side state compilation.
2. **Pyvis (HTML compilation inside Iframe)**:
   * *Pros*: High visual polish, physics-based animations, search boxes, and zoom.
   * *Cons*: Requires writing custom local HTML files to the filesystem and wrapping them inside an Iframe, which causes cross-origin warning banners inside Streamlit.
3. **Graphviz (Native Streamlit component `st.graphviz_chart`)**:
   * *Pros*: Zero external package requirements (built natively into Streamlit), extremely reliable across Windows/Linux without compilation errors, clean mathematical layouts, and outputs crisp vector SVGs.
   * *Selection*: We selected **Graphviz** for its robustness, native engine integration, and layout rendering safety.

### B. Analytical UI Layout: Tabs vs. Sidebar Expanders
To incorporate analytics without cluttering the chat tutor experience:
* *Sidebar Expanders*: Placing graphs and metrics in the sidebar squeezes charts into a narrow width (`< 300px`), making text overlaps in Graphviz and detailed telemetry timelines hard to read.
* *Tabbed Layout (`st.tabs`)*: Tab division keeps the screen wide and readable. The user can chat distraction-free in the first tab and toggle to the second tab to view charts and download study guides in full resolution.
* *Selection*: We implemented a top-level tab division: **`💬 Sentio Tutor`** and **`📊 Learning Dashboard`**.
