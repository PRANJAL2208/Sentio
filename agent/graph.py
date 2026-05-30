"""
agent/graph.py

LangGraph agent with 4 nodes:

  [classify_load] → (conditional edge) → [overloaded | optimal | underloaded]
                                                        ↓
                                               [generate_response]

Each response node has a different system prompt baked in.
The memory context (forgotten topics) is prepended at generation time.
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from core.load_detector import classify_load, get_system_prompt
from memory.store import build_memory_context


# ── Agent state ───────────────────────────────────────────────────────────────
# Everything the graph passes between nodes.

class AgentState(TypedDict):
    # Inputs
    user_message: str
    pause_seconds: float
    recent_clarification_count: int
    chat_history: list[dict]       # [{"role": "user"|"assistant", "content": "..."}]
    memory_collection: object      # ChromaDB collection (passed in, not serialized)

    # Computed by nodes
    load_state: str                # "OVERLOADED" | "OPTIMAL" | "UNDERLOADED"
    load_confidence: float
    load_signals: dict
    system_prompt: str
    memory_context: str
    response: str


# ── Node 1: Classify cognitive load ──────────────────────────────────────────

def classify_load_node(state: AgentState) -> AgentState:
    """
    Reads behavioral signals → sets load_state, load_confidence, load_signals.
    No LLM call. Pure Python logic.
    """
    result = classify_load(
        pause_seconds=state["pause_seconds"],
        message_text=state["user_message"],
        recent_clarification_count=state["recent_clarification_count"],
    )

    system_prompt = get_system_prompt(result["state"])
    memory_context = build_memory_context(state["memory_collection"])

    return {
        **state,
        "load_state": result["state"],
        "load_confidence": result["confidence"],
        "load_signals": result["signals"],
        "system_prompt": system_prompt,
        "memory_context": memory_context,
    }


# ── Conditional edge: route based on load state ───────────────────────────────

def route_by_load(state: AgentState) -> Literal["generate_overloaded", "generate_optimal", "generate_underloaded"]:
    """
    LangGraph calls this to decide which node to go to next.
    Returns the name of the next node as a string.
    """
    routing = {
        "OVERLOADED":   "generate_overloaded",
        "OPTIMAL":      "generate_optimal",
        "UNDERLOADED":  "generate_underloaded",
    }
    return routing.get(state["load_state"], "generate_optimal")


# ── Node factory: generate response ──────────────────────────────────────────
# We create 3 identical nodes with different labels so LangGraph can route to them.
# The actual behavior difference comes entirely from the system_prompt in state,
# which was set in classify_load_node.

def _make_generate_node(llm: ChatOpenAI):
    """Returns a generation function that closes over the LLM client."""

    def generate(state: AgentState) -> AgentState:
        # Build full system prompt: memory context + load-adapted instructions
        full_system = ""
        if state["memory_context"]:
            full_system += state["memory_context"] + "\n\n"
        full_system += state["system_prompt"]

        # Build message list for the LLM
        messages = [SystemMessage(content=full_system)]

        # Add recent chat history (last 6 turns to stay within context)
        for turn in state["chat_history"][-6:]:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                from langchain_core.messages import AIMessage
                messages.append(AIMessage(content=turn["content"]))

        # Add current user message
        messages.append(HumanMessage(content=state["user_message"]))

        response = llm.invoke(messages)

        return {
            **state,
            "response": response.content,
        }

    return generate


# ── Build the graph ───────────────────────────────────────────────────────────

def build_graph(openai_api_key: str, model: str = "gpt-4o-mini") -> object:
    """
    Assembles and compiles the LangGraph state machine.

    Swap model="gpt-4o-mini" for:
    - "gpt-4o" for better quality
    - Claude: replace ChatOpenAI with ChatAnthropic from langchain_anthropic
    """
    llm = ChatOpenAI(
        model=model,
        api_key=openai_api_key,
        temperature=0.7,
    )

    generate_node = _make_generate_node(llm)

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify_load",        classify_load_node)
    graph.add_node("generate_overloaded",  generate_node)
    graph.add_node("generate_optimal",     generate_node)
    graph.add_node("generate_underloaded", generate_node)

    # Entry point
    graph.set_entry_point("classify_load")

    # Conditional routing from classify → one of 3 generate nodes
    graph.add_conditional_edges(
        "classify_load",
        route_by_load,
        {
            "generate_overloaded":  "generate_overloaded",
            "generate_optimal":     "generate_optimal",
            "generate_underloaded": "generate_underloaded",
        },
    )

    # All generate nodes → END
    graph.add_edge("generate_overloaded",  END)
    graph.add_edge("generate_optimal",     END)
    graph.add_edge("generate_underloaded", END)

    return graph.compile()


# ── Convenience runner ────────────────────────────────────────────────────────

def run_agent(
    compiled_graph,
    user_message: str,
    pause_seconds: float,
    chat_history: list[dict],
    memory_collection,
    recent_clarification_count: int = 0,
) -> dict:
    """
    Single function to call from app.py.

    Returns dict with:
        response (str)
        load_state (str)
        load_confidence (float)
        load_signals (dict)
    """
    initial_state: AgentState = {
        "user_message": user_message,
        "pause_seconds": pause_seconds,
        "recent_clarification_count": recent_clarification_count,
        "chat_history": chat_history,
        "memory_collection": memory_collection,
        "load_state": "",
        "load_confidence": 0.0,
        "load_signals": {},
        "system_prompt": "",
        "memory_context": "",
        "response": "",
    }

    final_state = compiled_graph.invoke(initial_state)

    return {
        "response":        final_state["response"],
        "load_state":      final_state["load_state"],
        "load_confidence": final_state["load_confidence"],
        "load_signals":    final_state["load_signals"],
    }
