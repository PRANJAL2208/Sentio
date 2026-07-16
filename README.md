# Sentio — Cognitive-Adaptive Tutoring Platform

Sentio is a privacy-preserving, cognitive-adaptive Intelligent Tutoring System (ITS) designed for multi-user academic research. It passively infers a learner's cognitive load in real-time through **Keystroke Dynamics (KSD)** telemetry—tracking typing dwell times, flight times, backspaces, and planning pauses. 

Based on this telemetry, Sentio dynamically adjusts explanation complexity (scaffolding conceptual detail) and modulates Spaced Repetition (SRS) stability factors using a load-sensitive Ebbinghaus forgetting model to maximize learning efficiency.

---

## Key Features

* **Keystroke Biometric Telemetry**: Real-time browser logging of keystroke flight/dwell intervals and deletion counts to classify mental strain (Overloaded, Optimal, Underloaded).
* **Pedagogical Scaffolding State Machine**: Routes conversation dynamically through distinct LangGraph nodes based on cognitive load.
* **Google OAuth & Fallback Authentication**: Secure login portal with automatic balanced counterbalance assignment (Sentio Mode vs. Control Mode).
* **Dual-Write Data Sync (SQLite & Supabase)**: Records pre/post tests, NASA-TLX workloads, and typing logs simultaneously to thread-safe local SQLite (WAL mode) and remote Supabase PostgreSQL.
* **🔐 Interactive Admin Control Console**: Web-native compiler tab unlocked via password that lets researchers inspect participant lists, exclude outlier emails, and calculate paired/Welch t-tests with charts directly in the browser.

---

## Project Structure

```
Sentio/
├── app.py                  # Streamlit web app and Admin Console — RUN THIS
├── tests.py                # Comprehensive 48-assertion unit/integration test suite
├── analyze_results.py      # Statistical analysis script & simulated cohort generator
├── requirements.txt        # Package dependencies
├── core/
│   ├── auth.py             # Google OAuth and email session validation gate
│   └── db.py               # SQLite WAL & Supabase REST PostgREST synchronization
├── agent/
│   └── graph.py            # LangGraph routing state machine for cognitive tutoring
├── memory/
│   └── store.py            # ChromaDB long-term memory & Ebbinghaus decay formulas
└── ui/
    └── timing.py           # Typing telemetry collection bridge
```

---

## Installation & Setup

1. **Clone the repository and install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**:
   Create a `.env` file in the project root:
   ```env
   # LLM API Configuration (Falls back to Groq if others are empty)
   GROQ_API_KEY=gsk_your_groq_key_here
   
   # Optional: Google OAuth Configuration
   GOOGLE_CLIENT_ID=your_google_id_here
   GOOGLE_CLIENT_SECRET=your_google_secret_here
   
   # Optional: Persistent Supabase Cloud DB Connection (Otherwise defaults to local SQLite)
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your-supabase-key-here
   
   # Security Access Keys
   ADMIN_KEY=sentio2026
   ```

---

## Running the Platform

To run the web console locally:
```bash
streamlit run app.py
```

### Participant Study Flow
1. Open the login page, enter email (e.g., `tester@domain.com`).
2. Launch the Console. The curriculum engine deterministically assigns the user to a research group based on email hash.
3. Select a Topic, complete the **Pre-Test**, go through the **5-minute study visor**, complete the **Post-Test**, and fill out the **NASA-TLX survey**.

### Researcher Admin Flow
1. Navigate to the **🔐 Admin Console** tab in the app.
2. Enter your `ADMIN_KEY` password (default: `sentio2026`).
3. View participant logs, exclude testing emails from the cohort, and instantly view computed statistical tables and plots (paired t-tests and Welch's t-tests).

---

## Running Automated Tests

To verify that the database tables, API routing fallbacks, and memory decay formulas are functioning correctly, run the 48-assertion test suite:
```bash
python tests.py
```

---

## Statistical Analysis CLI

To simulate a participant cohort or analyze an active database file locally:
* **To generate mock cohort data (10 users, 40 sessions, 200 telemetry entries) for evaluation**:
  ```bash
  python analyze_results.py --mock
  ```
* **To compile statistics (T-Statistics, P-Values, and NASA-TLX curves) from the active database**:
  ```bash
  python analyze_results.py
  ```

---

*Built by Pranjal — 2026*