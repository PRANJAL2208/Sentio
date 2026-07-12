"""
tests.py

Run before touching app.py to confirm each component works in isolation.
Usage: python tests.py

No pytest needed. Just run it.
"""

import sys
import math
import time

print("=" * 60)
print("Sentio — Component Tests")
print("=" * 60)

passed = 0
failed = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  ✓  {name}")
        passed += 1
    else:
        print(f"  ✗  {name}  ← FAILED {detail}")
        failed += 1


# ── Test 1: Cognitive Load Detector ──────────────────────────────────────────
print("\n[1] core/load_detector.py")

from core.load_detector import classify_load, is_clarification, get_system_prompt

# Overloaded: long pause, short message, clarification
r = classify_load(pause_seconds=12.0, message_text="huh?", recent_clarification_count=2)
test("Long pause + short confused message → OVERLOADED", r["state"] == "OVERLOADED", r)

# Optimal: medium pause, medium message
r = classify_load(pause_seconds=4.0, message_text="Can you explain how backpropagation works in neural networks?")
test("Medium pause + clear question → OPTIMAL", r["state"] == "OPTIMAL", r)

# Underloaded: fast response, long message, no confusion
r = classify_load(pause_seconds=1.0, message_text="I understand gradient descent and backprop. What about second-order optimization methods like L-BFGS or natural gradient? I'm curious about the curvature landscape and how Hessian approximation helps.")
test("Fast + long engaged message → UNDERLOADED", r["state"] == "UNDERLOADED", r)

# Keystroke dynamics tests
r = classify_load(pause_seconds=4.0, message_text="Standard question text.", avg_flight_ms=400.0)
test("High transition latency (flight time) → OVERLOADED", r["state"] == "OVERLOADED", r)

r = classify_load(pause_seconds=4.0, message_text="This is a sentence.", backspace_count=5)
test("High backspace rate (>15%) → OVERLOADED", r["state"] == "OVERLOADED", r)

r = classify_load(pause_seconds=3.0, message_text="Short text but typed fast.", avg_flight_ms=100.0, avg_dwell_ms=60.0)
test("Fast flight and dwell times → UNDERLOADED", r["state"] == "UNDERLOADED", r)

r = classify_load(pause_seconds=10.0, message_text="Short query.", avg_flight_ms=150.0, backspace_count=0)
test("Confident telemetry offsets high reading pause → OPTIMAL", r["state"] == "OPTIMAL", r)

r = classify_load(pause_seconds=12.0, message_text="everything", recent_clarification_count=1)
test("Explicit 'everything' request overrides overload state", r["state"] != "OVERLOADED", r)

# Clarification detection
test("'what do you mean' → is_clarification", is_clarification("wait, what do you mean by that?"))
test("'okay great!' → not clarification", not is_clarification("okay great, that makes sense!"))

# System prompts exist for all 3 states
for state in ["OVERLOADED", "OPTIMAL", "UNDERLOADED"]:
    prompt = get_system_prompt(state)
    test(f"System prompt for {state} is non-empty", len(prompt) > 50)


# ── Test 2: Ebbinghaus Forgetting Curve ──────────────────────────────────────
print("\n[2] memory/store.py — Ebbinghaus formula")

from memory.store import retention_score

# Just learned (0 days) → ~100% retention
r = retention_score(learned_at_ts=time.time(), stability=1.0)
test("Just learned → retention near 1.0", r > 0.95, f"got {r:.3f}")

# 1 day ago, stability=1 → retention = e^(-1) ≈ 0.368
one_day_ago = time.time() - 86400
r = retention_score(learned_at_ts=one_day_ago, stability=1.0)
expected = math.exp(-1)
test(f"1 day ago → retention ≈ {expected:.3f}", abs(r - expected) < 0.01, f"got {r:.3f}")

# 3 days ago, stability=1 → retention = e^(-3) ≈ 0.050
three_days_ago = time.time() - (3 * 86400)
r = retention_score(learned_at_ts=three_days_ago, stability=1.0)
test("3 days ago → retention < 0.1 (mostly forgotten)", r < 0.1, f"got {r:.3f}")

# Higher stability = slower forgetting
r_low_stab  = retention_score(learned_at_ts=one_day_ago, stability=1.0)
r_high_stab = retention_score(learned_at_ts=one_day_ago, stability=3.0)
test("Higher stability → higher retention", r_high_stab > r_low_stab,
     f"stab=1→{r_low_stab:.3f}, stab=3→{r_high_stab:.3f}")


# ── Test 3: ChromaDB store/retrieve ──────────────────────────────────────────
print("\n[3] memory/store.py — ChromaDB")

try:
    import chromadb
    from memory.store import get_chroma_client, get_or_create_collection, store_topic, get_forgotten_topics

    client = get_chroma_client(persist_dir="/tmp/sentio_test_db")
    collection = get_or_create_collection(client, user_id="test_user")
    test("ChromaDB client created", client is not None)

    # Store a topic learned 3 days ago (should be forgotten)
    topic_id = store_topic(
        collection=collection,
        topic_summary="Transformers use self-attention to weigh token relationships",
        session_id="test_session",
        stability=1.0,
    )
    # Manually backdate the timestamp to 3 days ago
    meta = collection.get(ids=[topic_id], include=["metadatas"])["metadatas"][0]
    collection.update(
        ids=[topic_id],
        metadatas=[{**meta, "last_seen_at": time.time() - (3 * 86400)}]
    )
    test("Topic stored in ChromaDB", topic_id is not None)

    forgotten = get_forgotten_topics(collection, retention_threshold=0.5)
    test("Backdated topic appears as forgotten", len(forgotten) > 0,
         f"got {len(forgotten)} forgotten topics")

    # Clean up test collection
    client.delete_collection(f"user_test_user")
    test("Test collection cleaned up", True)

except ImportError:
    print("  ⚠  ChromaDB not installed — run: pip install chromadb")
    print("     (skipping ChromaDB tests)")
except Exception as e:
    test("ChromaDB tests", False, str(e))


# ── Test 4: Timing utilities ──────────────────────────────────────────────────
print("\n[4] ui/timing.py")

from ui.timing import update_clarification_count

class FakeSessionState(dict):
    pass

ss = FakeSessionState()
count = update_clarification_count(ss, "what do you mean by that?")
test("Clarification message → count=1", count == 1, f"got {count}")

count = update_clarification_count(ss, "okay that makes sense")
test("Normal message after clarification → count stays at 1", count == 1, f"got {count}")

count = update_clarification_count(ss, "huh? I'm confused")
test("Second clarification → count=2", count == 2, f"got {count}")


# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print(f"  Results: {passed} passed, {failed} failed")
if failed == 0:
    print("  All tests passed. Ready to run: streamlit run app.py")
else:
    print("  Fix failing tests before running the app.")
print("=" * 60)

sys.exit(0 if failed == 0 else 1)
