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
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from core.load_detector import classify_load, get_system_prompt
from memory.store import (
    build_memory_context,
    get_forgotten_topics,
    evaluate_user_answer,
    update_topic_stability
)


# ── Agent state ───────────────────────────────────────────────────────────────
# Everything the graph passes between nodes.

class AgentState(TypedDict):
    # Inputs
    user_message: str
    pause_seconds: float
    recent_clarification_count: int
    avg_dwell_ms: float
    avg_flight_ms: float
    backspace_count: int
    chat_history: list[dict]       # [{"role": "user"|"assistant", "content": "..."}]
    memory_collection: object      # ChromaDB collection (passed in, not serialized)

    # Active SRS Quiz fields
    srs_quiz_active: bool
    srs_topic_id: str
    srs_evaluation: str

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
        avg_dwell_ms=state.get("avg_dwell_ms", 0.0),
        avg_flight_ms=state.get("avg_flight_ms", 0.0),
        backspace_count=state.get("backspace_count", 0),
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

def route_by_load(state: AgentState) -> Literal["generate_overloaded", "generate_optimal", "generate_underloaded", "evaluate_quiz"]:
    """
    LangGraph calls this to decide which node to go to next.
    """
    if state.get("srs_quiz_active"):
        return "evaluate_quiz"

    routing = {
        "OVERLOADED":   "generate_overloaded",
        "OPTIMAL":      "generate_optimal",
        "UNDERLOADED":  "generate_underloaded",
    }
    return routing.get(state["load_state"], "generate_optimal")


# ── Node factory: generate response ──────────────────────────────────────────
# We create 3 identical nodes with different labels so LangGraph can route to them.
# The actual behavior difference comes entirely from the system_prompt in state,
# ── Specialized Graph Nodes ──────────────────────────────────────────────────

SCAFFOLD_PROMPT = """You are a patient, clear AI tutor. The user is currently overloaded.
Your explanation must follow these guidelines:
- Present the concept using a concrete worked example.
- Break down the logic into progressive, easy-to-follow numbered steps.
- Explain each step using simple everyday analogies. Strictly avoid unexplained jargon.
- End by asking the user a single, very simple interactive check-in question to confirm they followed the example (e.g. "Does this analogy make sense, or would you like to see another example?").
"""

SOCRATIC_PROMPT = """You are a challenging Socratic AI tutor. The user is highly capable and underloaded.
Your explanation must follow these guidelines:
- Do NOT directly explain the answer or give a flat explanation.
- Instead, prompt the user with 1-2 structured guiding questions or a conceptual thought experiment that leads them to discover the answer themselves.
- Use precise, advanced terminology where appropriate.
- Encourage them to articulate their mental model.
"""

def _make_scaffold_node(llm: BaseChatModel):
    def node(state: AgentState) -> AgentState:
        full_system = SCAFFOLD_PROMPT
        if state["memory_context"]:
            full_system = state["memory_context"] + "\n\n" + full_system
        messages = [SystemMessage(content=full_system)]
        for turn in state["chat_history"][-6:]:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                from langchain_core.messages import AIMessage
                messages.append(AIMessage(content=turn["content"]))
        messages.append(HumanMessage(content=state["user_message"]))
        response = llm.invoke(messages)
        return {**state, "response": response.content}
    return node

def _make_socratic_node(llm: BaseChatModel):
    def node(state: AgentState) -> AgentState:
        full_system = SOCRATIC_PROMPT
        messages = [SystemMessage(content=full_system)]
        for turn in state["chat_history"][-6:]:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                from langchain_core.messages import AIMessage
                messages.append(AIMessage(content=turn["content"]))
        messages.append(HumanMessage(content=state["user_message"]))
        response = llm.invoke(messages)
        return {**state, "response": response.content}
    return node

def _make_optimal_node(llm: BaseChatModel):
    def node(state: AgentState) -> AgentState:
        # Check for forgotten topics (retention < 0.7)
        forgotten = get_forgotten_topics(state["memory_collection"], retention_threshold=0.7, max_results=1)
        if forgotten:
            topic = forgotten[0]
            quiz_prompt = f"""You are a helpful AI tutor. The user is in flow (Optimal state). 
Before answering their current query, you must test their recall of a previously covered concept: "{topic['summary']}".

Ask the user a single friendly diagnostic recall question about this topic (e.g., "To quickly review, do you remember how/what...").
Let them know that after they answer this quick review, you will address their question: "{state['user_message']}".
Keep the quiz question simple, clear, and engaging.
"""
            messages = [SystemMessage(content=quiz_prompt)]
            response = llm.invoke(messages)
            return {
                **state,
                "response": response.content,
                "srs_quiz_active": True,
                "srs_topic_id": topic["topic_id"],
                "srs_evaluation": "",
            }
        
        # Standard direct instruction node
        full_system = state["system_prompt"]
        messages = [SystemMessage(content=full_system)]
        for turn in state["chat_history"][-6:]:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                from langchain_core.messages import AIMessage
                messages.append(AIMessage(content=turn["content"]))
        messages.append(HumanMessage(content=state["user_message"]))
        response = llm.invoke(messages)
        return {
            **state,
            "response": response.content,
            "srs_quiz_active": False,
            "srs_topic_id": "",
            "srs_evaluation": "",
        }
    return node

def _make_evaluate_quiz_node(llm: BaseChatModel):
    def node(state: AgentState) -> AgentState:
        topic_id = state["srs_topic_id"]
        result = state["memory_collection"].get(ids=[topic_id], include=["documents"])
        if not result["ids"]:
            return {
                **state,
                "response": "Got it! Let's continue. What would you like to discuss next?",
                "srs_quiz_active": False,
                "srs_topic_id": "",
                "srs_evaluation": "",
            }
        
        topic_summary = result["documents"][0]
        user_answer = state["user_message"]
        
        grade = evaluate_user_answer(llm, topic_summary, user_answer)
        update_topic_stability(state["memory_collection"], topic_id, grade)
        
        feedback_prompts = {
            "correct": f"Correct! Spot-on recall of the concept: '{topic_summary}'. I've updated your memory retention curve. What would you like to learn next?",
            "partial": f"Close! You got parts of it right. The full concept is: '{topic_summary}'. I've updated your curve. What would you like to learn next?",
            "incorrect": f"Not quite. The concept was: '{topic_summary}'. I've reset the review clock so we can revisit it soon. What would you like to learn next?"
        }
        response_text = feedback_prompts.get(grade, feedback_prompts["incorrect"])
        
        return {
            **state,
            "response": response_text,
            "srs_quiz_active": False,
            "srs_topic_id": "",
            "srs_evaluation": grade,
        }
    return node


# ── Build the graph ───────────────────────────────────────────────────────────

def build_graph(llm: BaseChatModel) -> object:
    """
    Assembles and compiles the LangGraph state machine.
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify_load",        classify_load_node)
    graph.add_node("generate_overloaded",  _make_scaffold_node(llm))
    graph.add_node("generate_optimal",     _make_optimal_node(llm))
    graph.add_node("generate_underloaded", _make_socratic_node(llm))
    graph.add_node("evaluate_quiz",        _make_evaluate_quiz_node(llm))

    # Entry point
    graph.set_entry_point("classify_load")

    # Conditional routing
    graph.add_conditional_edges(
        "classify_load",
        route_by_load,
        {
            "generate_overloaded":  "generate_overloaded",
            "generate_optimal":     "generate_optimal",
            "generate_underloaded": "generate_underloaded",
            "evaluate_quiz":        "evaluate_quiz",
        },
    )

    # All generate/evaluate nodes → END
    graph.add_edge("generate_overloaded",  END)
    graph.add_edge("generate_optimal",     END)
    graph.add_edge("generate_underloaded", END)
    graph.add_edge("evaluate_quiz",        END)

    return graph.compile()


# ── Convenience runner ────────────────────────────────────────────────────────

def run_agent(
    compiled_graph,
    user_message: str,
    pause_seconds: float,
    chat_history: list[dict],
    memory_collection,
    recent_clarification_count: int = 0,
    avg_dwell_ms: float = 0.0,
    avg_flight_ms: float = 0.0,
    backspace_count: int = 0,
    srs_quiz_active: bool = False,
    srs_topic_id: str = "",
) -> dict:
    """
    Single function to call from app.py.

    Returns dict with response, load details, and SRS quiz state.
    """
    initial_state: AgentState = {
        "user_message": user_message,
        "pause_seconds": pause_seconds,
        "recent_clarification_count": recent_clarification_count,
        "avg_dwell_ms": avg_dwell_ms,
        "avg_flight_ms": avg_flight_ms,
        "backspace_count": backspace_count,
        "chat_history": chat_history,
        "memory_collection": memory_collection,
        "srs_quiz_active": srs_quiz_active,
        "srs_topic_id": srs_topic_id,
        "srs_evaluation": "",
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
        "srs_quiz_active": final_state.get("srs_quiz_active", False),
        "srs_topic_id":    final_state.get("srs_topic_id", ""),
        "srs_evaluation":  final_state.get("srs_evaluation", ""),
    }
