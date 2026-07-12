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
) -> str:
    """
    Store a topic that was explained in this session.

    Args:
        collection: ChromaDB collection
        topic_summary: 1-2 sentence summary of what was explained
        session_id: identifier for this chat session
        stability: how deeply it was covered (1.0 default, increase for revisited topics)

    Returns:
        topic_id (str)
    """
    topic_id = str(uuid.uuid4())
    now_ts = time.time()

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
You are a topic extractor. Given an AI tutor's response, extract the 1-3 main concepts
that were explained. Return ONLY a JSON array of short strings (1 sentence each).
Example: ["Transformer attention computes similarity between query and key vectors",
          "Softmax normalizes attention scores into probabilities"]
If no clear concept was explained, return [].
Only return the JSON array, nothing else.
"""


def extract_topics_from_response(llm, response_text: str) -> list[str]:
    """
    Use a fast LLM call to extract topics from the assistant's response.
    These get stored in ChromaDB for the forgetting curve.
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

        topics = json.loads(raw)
        return topics if isinstance(topics, list) else []
    except Exception:
        return []  # fail silently — memory is enhancement, not core path
