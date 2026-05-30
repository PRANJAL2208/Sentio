# Sentio — an AI with EQ

> An AI that reads how you feel.
An AI that detects your cognitive load in real-time from typing behavior. It watches how you type, your pauses, your phrasing, your confusion signals and silently adapts every response to match your cognitive and emotional state in real time.

Overwhelmed? It simplifies. Focused? It goes deep. Bored? It challenges you.

---

## How It Works

**Cognitive Load Detection** — Measures typing pause duration, message length, and clarification patterns to classify your mental state as Overloaded, Optimal, or Underloaded.

**Adaptive Response Engine** — A LangGraph agent routes your message to one of three response styles, each with a different system prompt tuned for your current state.

**Ebbinghaus Memory** — ChromaDB stores concepts you've learned. A forgetting curve formula (`R = e^(-t/S)`) tracks what you've likely forgotten and quietly reintroduces it.

---

## Tech Stack

- LangGraph — agent orchestration and conditional routing
- ChromaDB — local vector memory store
- Streamlit — UI and real-time JS timing bridge
- OpenAI / Groq / Gemini — pluggable LLM backend

---

## Project Structure

```
cogniflow/
├── app.py                  # Streamlit entry point — run this
├── requirements.txt        # All dependencies
├── core/
│   └── load_detector.py    # Cognitive load classifier (pause + length + follow-ups)
├── agent/
│   └── graph.py            # LangGraph agent with conditional routing
├── memory/
│   └── store.py            # ChromaDB session memory + Ebbinghaus forgetting curve
└── ui/
    └── timing.py           # JS component that captures typing pause duration
```

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Environment Variables

Create a `.env` file:
```
OPENAI_API_KEY=your_key_here
```
Or swap for Anthropic — see agent/graph.py comments.


Run:
```bash
streamlit run app.py
```

---

## Science Behind It

Built on Cognitive Load Theory (Sweller, 1988) and the Ebbinghaus Forgetting Curve. The classifier uses behavioral signals — not self-reported mood — to infer cognitive state passively and in real time.

---

*Built by Pranjal — 2026*