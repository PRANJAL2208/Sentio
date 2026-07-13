"""
memory/store.py

Two things:
1. ChromaDB — stores what topics were covered in each session.
2. Ebbinghaus Forgetting Curve — decides which past topics to resurface.

The forgetting curve formula: R = e^(-t/S)
  R = retention (1.0 = remember everything, 0.0 = forgotten completely)
  t = days elapsed since the topic was last seen
  S = stability (how well it was learned; starts at 1.0, increases with re-exposure)

If R < 0.5, the user has forgotten more than half — resurface the topic.
"""

import math
import time
import uuid
import json
from datetime import datetime

import chromadb
from chromadb.config import Settings


# ── ChromaDB setup ────────────────────────────────────────────────────────────

def get_chroma_client(persist_dir: str = "./sentio_memory") -> chromadb.ClientAPI:
    """Returns a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=persist_dir)


def get_or_create_collection(client: chromadb.ClientAPI, user_id: str = "default"):
    """Each user gets their own collection."""
    return client.get_or_create_collection(
        name=f"user_{user_id}",
        metadata={"hnsw:space": "cosine"},
    )


# ── Storing what was learned ──────────────────────────────────────────────────

def store_topic(
    collection,
    topic_summary: str,
    session_id: str,
    stability: float = 1.0,
    links_to: list[str] = None,
) -> str:
    """
    Store a topic that was explained in this session, along with optional conceptual links.

    Args:
        collection: ChromaDB collection
        topic_summary: 1-2 sentence summary of what was explained
        session_id: identifier for this chat session
        stability: how deeply it was covered (1.0 default, increase for revisited topics)
        links_to: list of related parent topic strings

    Returns:
        topic_id (str)
    """
    topic_id = str(uuid.uuid4())
    now_ts = time.time()
    links_str = json.dumps(links_to or [])

    collection.add(
        documents=[topic_summary],
        ids=[topic_id],
        metadatas=[{
            "session_id": session_id,
            "learned_at": now_ts,                    # Unix timestamp
            "last_seen_at": now_ts,
            "stability": stability,
            "review_count": 0,
            "learned_date": datetime.now().isoformat(),
            "links_to": links_str,
        }],
    )
    return topic_id


def update_topic_on_review(collection, topic_id: str):
    """
    Call this when a topic is resurfaced/reviewed.
    Increases stability (makes it harder to forget again) and resets the clock.
    """
    result = collection.get(ids=[topic_id], include=["metadatas"])
    if not result["ids"]:
        return

    meta = result["metadatas"][0]
    new_stability = meta["stability"] * 2.0   # Each review doubles retention duration
    new_review_count = meta["review_count"] + 1

    collection.update(
        ids=[topic_id],
        metadatas=[{
            **meta,
            "last_seen_at": time.time(),
            "stability": new_stability,
            "review_count": new_review_count,
        }],
    )


# ── Ebbinghaus forgetting curve ───────────────────────────────────────────────

def retention_score(learned_at_ts: float, stability: float) -> float:
    """
    Calculate how much a user probably remembers right now.

    R = e^(-t/S)
    t = days elapsed since last seen
    S = stability (higher = slower forgetting)

    Returns float between 0.0 and 1.0.
    """
    days_elapsed = (time.time() - learned_at_ts) / 86400.0  # convert seconds → days
    days_elapsed = max(days_elapsed, 0.0)                    # never negative
    return math.exp(-days_elapsed / stability)


def get_forgotten_topics(
    collection,
    retention_threshold: float = 0.5,
    max_results: int = 3,
) -> list[dict]:
    """
    Find topics the user has probably forgotten (retention < threshold).

    Returns list of dicts: [{topic_id, summary, retention, days_ago}, ...]
    Sorted by most forgotten first (lowest retention).
    """
    all_items = collection.get(include=["documents", "metadatas"])

    if not all_items["ids"]:
        return []

    forgotten = []
    for doc_id, document, meta in zip(
        all_items["ids"],
        all_items["documents"],
        all_items["metadatas"],
    ):
        r = retention_score(
            learned_at_ts=meta["last_seen_at"],
            stability=meta["stability"],
        )
        if r < retention_threshold:
            days_ago = (time.time() - meta["last_seen_at"]) / 86400.0
            forgotten.append({
                "topic_id": doc_id,
                "summary": document,
                "retention": round(r, 3),
                "days_ago": round(days_ago, 1),
                "review_count": meta["review_count"],
            })

    # Sort by most forgotten first
    forgotten.sort(key=lambda x: x["retention"])
    return forgotten[:max_results]


def build_memory_context(collection) -> str:
    """
    Build a context string to prepend to the system prompt when forgotten topics exist.
    This tells the LLM what to briefly re-ground the user on before the new answer.
    """
    forgotten = get_forgotten_topics(collection)

    if not forgotten:
        return ""

    lines = ["[MEMORY NOTE — quietly re-ground the user on these before answering:]"]
    for item in forgotten:
        pct = int(item["retention"] * 100)
        lines.append(
            f"- '{item['summary']}' "
            f"(covered {item['days_ago']} days ago, ~{pct}% estimated retention)"
        )
    lines.append(
        "\nNaturally weave a brief reminder of the above into your response. "
        "Don't announce that you're doing this. Just do it smoothly.\n"
    )
    return "\n".join(lines)


# ── Extracting topics from LLM responses ─────────────────────────────────────

TOPIC_EXTRACTION_PROMPT = """
You are a topic and relationship extractor. Given an AI tutor's response, extract the 1-3 main concepts
that were explained. For each concept, provide:
1. "summary": A 1-sentence summary of the concept explained.
2. "links_to": A list of 1-3 parent topics or related terms (like keywords or categories) that this concept connects to.

Return ONLY a JSON object with a single key "concepts" mapping to a list of these objects.
Example:
{
  "concepts": [
    {
      "summary": "Transformer attention computes similarity between query and key vectors",
      "links_to": ["Transformers", "Attention Mechanism"]
    },
    {
      "summary": "Softmax normalizes attention scores into probabilities",
      "links_to": ["Attention Mechanism", "Softmax"]
    }
  ]
}
If no clear concept was explained, return {"concepts": []}.
Only return the JSON object, nothing else.
"""


def extract_topics_from_response(llm, response_text: str) -> list[dict]:
    """
    Use a fast LLM call to extract topics and their relationship links from the response.
    Returns: [{"summary": str, "links_to": list[str]}]
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    try:
        messages = [
            SystemMessage(content=TOPIC_EXTRACTION_PROMPT),
            HumanMessage(content=response_text[:2000]),
        ]
        result = llm.invoke(messages)
        raw = result.content.strip()

        # Clean markdown code block wraps if LLM returns it wrapped in ```json
        if raw.startswith("```"):
            lines = raw.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()

        data = json.loads(raw)
        if isinstance(data, dict) and "concepts" in data:
            return data["concepts"]
        elif isinstance(data, list):
            # Fallback for old output format
            out = []
            for item in data:
                if isinstance(item, str):
                    out.append({"summary": item, "links_to": []})
                elif isinstance(item, dict):
                    out.append({
                        "summary": item.get("summary", ""),
                        "links_to": item.get("links_to", [])
                    })
            return out
        return []
    except Exception:
        return []  # fail silently — memory is enhancement, not core path


# ── Active SRS practice evaluations ──────────────────────────────────────────

def evaluate_user_answer(llm, topic_summary: str, user_answer: str) -> str:
    """
    Evaluate the user's recall answer using the LLM.
    Returns: "correct" | "partial" | "incorrect"
    """
    from langchain_core.messages import SystemMessage
    prompt = """You are an objective grading assistant for a tutoring system.
Your job is to evaluate if a user's answer shows they remember the core concept of a previously learned topic.

Topic Summary: "{topic}"
User's Answer: "{answer}"

Evaluate the answer and output EXACTLY one word:
- "correct" if the user correctly recalls the main idea or the essence of the concept.
- "partial" if the user has some parts right but misses key details.
- "incorrect" if the user is completely wrong, confused, or says they don't know/forgot.

Do not output any punctuation, code blocks, or extra text. Output only one of: correct, partial, incorrect.
"""
    messages = [
        SystemMessage(content=prompt.format(topic=topic_summary, answer=user_answer))
    ]
    try:
        result = llm.invoke(messages)
        grade = result.content.strip().lower()
        for choice in ["incorrect", "partial", "correct"]:
            if choice in grade:
                return choice
        return "incorrect"
    except Exception:
        return "incorrect"


def update_topic_stability(collection, topic_id: str, grade: str) -> float:
    """
    Updates the topic's stability based on a retrieval practice grade.
    Returns the new stability value.
    """
    result = collection.get(ids=[topic_id], include=["metadatas"])
    if not result["ids"]:
        return 1.0

    meta = result["metadatas"][0]
    current_stability = float(meta.get("stability", 1.0))
    
    if grade == "correct":
        new_stability = current_stability * 2.0
    elif grade == "partial":
        new_stability = current_stability * 1.2
    else:
        new_stability = 1.0

    new_review_count = int(meta.get("review_count", 0)) + 1

    collection.update(
        ids=[topic_id],
        metadatas=[{
            **meta,
            "last_seen_at": time.time(),
            "stability": new_stability,
            "review_count": new_review_count,
        }],
    )
    return new_stability
