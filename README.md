# CogniFlow

An AI chat agent that detects your cognitive load in real-time from typing behavior
and adapts how it responds — simpler when you're overwhelmed, richer when you're in flow.

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
