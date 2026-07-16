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


# ── Test 5: Active SRS & Scaffolded Routing ──────────────────────────────────
print("\n[5] Active SRS & Scaffolded Routing")

from memory.store import evaluate_user_answer, update_topic_stability
from agent.graph import route_by_load

class FakeLLM:
    def __init__(self, response_content: str):
        self.response_content = response_content

    class ContentHolder:
        def __init__(self, content: str):
            self.content = content

    def invoke(self, messages):
        return self.ContentHolder(self.response_content)

# 1. Test user recall evaluation
llm_correct = FakeLLM("correct")
grade_correct = evaluate_user_answer(llm_correct, "Self-attention weighs token similarity", "It calculates token similarity")
test("evaluate_user_answer parses 'correct' grade", grade_correct == "correct", f"got {grade_correct}")

llm_incorrect = FakeLLM("incorrect")
grade_incorrect = evaluate_user_answer(llm_incorrect, "Self-attention weighs token similarity", "I forgot")
test("evaluate_user_answer parses 'incorrect' grade", grade_incorrect == "incorrect", f"got {grade_incorrect}")

# 2. Test stability updates
try:
    from memory.store import get_chroma_client, get_or_create_collection, store_topic
    client = get_chroma_client(persist_dir="/tmp/sentio_test_db_srs")
    collection = get_or_create_collection(client, user_id="test_user_srs")

    topic_id = store_topic(collection, "Attention weighs queries and keys", "test_session", stability=2.0)

    # Double on correct
    new_stab = update_topic_stability(collection, topic_id, "correct")
    test("Stability doubled on correct grade", new_stab == 4.0, f"got {new_stab}")

    # Reset on incorrect
    new_stab = update_topic_stability(collection, topic_id, "incorrect")
    test("Stability reset to 1.0 on incorrect grade", new_stab == 1.0, f"got {new_stab}")

    client.delete_collection("user_test_user_srs")
except Exception as e:
    test("SRS stability database updates", False, str(e))


# 3. Test optimal node suppression logic
from agent.graph import _make_optimal_node

class FakeCollection:
    def get(self, *args, **kwargs):
        import time
        return {
            "ids": ["dummy_id"],
            "documents": ["Self-attention is a concept"],
            "metadatas": [{"last_seen_at": time.time() - 999999, "stability": 1.0, "review_count": 0, "links_to": "[]"}]
        }

opt_node = _make_optimal_node(FakeLLM("quiz question"))

state_cooldown = {
    "memory_collection": FakeCollection(),
    "turns_since_last_quiz": 2,
    "recent_clarification_count": 0,
    "user_message": "tell me more",
    "chat_history": [],
    "system_prompt": "You are a tutor",
}
res_cooldown = opt_node(state_cooldown)
test("Optimal node suppresses quiz during cooldown (<4 turns)", res_cooldown["srs_quiz_active"] == False, f"got {res_cooldown['srs_quiz_active']}")

state_clarifying = {
    "memory_collection": FakeCollection(),
    "turns_since_last_quiz": 4,
    "recent_clarification_count": 1,
    "user_message": "tell me more",
    "chat_history": [],
    "system_prompt": "You are a tutor",
}
res_clarifying = opt_node(state_clarifying)
test("Optimal node suppresses quiz during clarification phase", res_clarifying["srs_quiz_active"] == False, f"got {res_clarifying['srs_quiz_active']}")

state_trigger = {
    "memory_collection": FakeCollection(),
    "turns_since_last_quiz": 4,
    "recent_clarification_count": 0,
    "user_message": "tell me more",
    "chat_history": [],
    "system_prompt": "You are a tutor",
}
res_trigger = opt_node(state_trigger)
test("Optimal node triggers quiz after cooldown", res_trigger["srs_quiz_active"] == True, f"got {res_trigger['srs_quiz_active']}")


# 4. Test graph routing
state_quiz_active = {"srs_quiz_active": True, "load_state": "OPTIMAL"}
next_node = route_by_load(state_quiz_active)
test("route_by_load routes to 'evaluate_quiz' if active", next_node == "evaluate_quiz", f"got {next_node}")

state_quiz_inactive = {"srs_quiz_active": False, "load_state": "OVERLOADED"}
next_node = route_by_load(state_quiz_inactive)
test("route_by_load routes to load state if inactive", next_node == "generate_overloaded", f"got {next_node}")


# ── Test 6: Relationships & Concept Maps ──────────────────────────────────────
print("\n[6] Relationships & Concept Maps")

from memory.store import extract_topics_from_response

# 1. Test relationship extraction prompt format
llm_relation = FakeLLM('{"concepts": [{"summary": "Transformers use self-attention", "links_to": ["Transformers"]}]}')
concepts = extract_topics_from_response(llm_relation, "dummy tutor text explaining transformers")
test("extract_topics_from_response extracts structured dict", len(concepts) == 1 and isinstance(concepts[0], dict), f"got {concepts}")
test("extract_topics_from_response preserves summary", concepts[0].get("summary") == "Transformers use self-attention", f"got {concepts}")
test("extract_topics_from_response preserves links_to", concepts[0].get("links_to") == ["Transformers"], f"got {concepts}")

# 2. Test storing relationships in ChromaDB
try:
    client = get_chroma_client(persist_dir="/tmp/sentio_test_db_relations")
    collection = get_or_create_collection(client, user_id="test_user_relations")

    topic_id = store_topic(
        collection, 
        topic_summary="Transformers use self-attention", 
        session_id="test_session", 
        stability=1.0, 
        links_to=["Transformers", "Attention"]
    )

    stored = collection.get(ids=[topic_id], include=["metadatas"])
    meta = stored["metadatas"][0]
    test("stored topic preserves links_to as serialized JSON string", "links_to" in meta, f"got {meta}")
    
    import json
    links = json.loads(meta["links_to"])
    test("links_to list matches original values", links == ["Transformers", "Attention"], f"got {links}")

    client.delete_collection("user_test_user_relations")
except Exception as e:
    test("Relationships database updates", False, str(e))


# ── Test 7: core/db.py & core/auth.py (Multi-User & Counterbalancing) ─────────
print("\n[7] core/db.py & core/auth.py")

from core.db import (
    init_database,
    register_user,
    get_user_group,
    log_session_start,
    log_session_end,
    log_quiz,
    log_workload,
    log_telemetry,
    get_connection,
)
from core.auth import is_google_auth_configured

try:
    # 1. Database initialization check
    init_database()
    
    # Clean previous test entries to ensure clean run
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM telemetry_records WHERE session_id = 'test_session_99'")
    c.execute("DELETE FROM workload_records WHERE email = 'tester@sentio.org'")
    c.execute("DELETE FROM quiz_records WHERE email = 'tester@sentio.org'")
    c.execute("DELETE FROM sessions WHERE session_id = 'test_session_99'")
    c.execute("DELETE FROM users WHERE email = 'tester@sentio.org'")
    conn.commit()
    conn.close()
    
    test("WAL database initialized successfully", True)

    # 2. User registration check
    register_user("tester@sentio.org", "Group B")
    grp = get_user_group("tester@sentio.org")
    test("User registration & group retrieval successful", grp == "Group B", f"got {grp}")

    # 3. Dynamic session logging check
    log_session_start("test_session_99", "tester@sentio.org", "Transformers", "SENTIO")
    
    # 4. Telemetry logging check
    log_telemetry("test_session_99", {
        "backspace_count": 8,
        "avg_dwell_ms": 110.0,
        "avg_flight_ms": 220.0,
        "pause_seconds": 3.2
    })
    
    # 5. Workload (NASA-TLX) logging check
    log_workload("tester@sentio.org", "Transformers", "SENTIO", {
        "mental_demand": 45,
        "physical_demand": 5,
        "temporal_demand": 25,
        "performance": 85,
        "effort": 55,
        "frustration": 10
    })
    
    # 6. Quiz score logging check
    log_quiz("tester@sentio.org", "Transformers", "PRE", 2, 3)
    log_quiz("tester@sentio.org", "Transformers", "POST", 3, 3)
    log_session_end("test_session_99")
    test("Database logs (session, telemetry, workload, quizzes) completed", True)

    # Verify rows exist in DB
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT score FROM quiz_records WHERE email = 'tester@sentio.org' AND quiz_type = 'POST'")
    quiz_val = c.fetchone()[0]
    test("Database quiz record read verification successful", quiz_val == 3, f"got {quiz_val}")
    # 7. Unified Cloud/Local Datastore Fetchers check (New in Phase 6)
    from core.db import get_all_users, get_all_sessions, get_all_quizzes, get_all_workloads, get_all_telemetry
    
    all_users = get_all_users()
    all_sessions = get_all_sessions()
    all_quizzes = get_all_quizzes()
    all_workloads = get_all_workloads()
    all_telemetry = get_all_telemetry()
    
    test("get_all_users returns registered list", any(u["email"] == "tester@sentio.org" for u in all_users))
    test("get_all_sessions returns active sessions", any(s["session_id"] == "test_session_99" for s in all_sessions))
    test("get_all_quizzes returns PRE/POST scores", any(q["email"] == "tester@sentio.org" and q["quiz_type"] == "POST" for q in all_quizzes))
    test("get_all_workloads returns NASA-TLX ratings", any(w["email"] == "tester@sentio.org" and w["frustration"] == 10 for w in all_workloads))
    test("get_all_telemetry returns keystroke telemetry list", any(t["session_id"] == "test_session_99" for t in all_telemetry))

    conn.close()

except Exception as e:
    test("Multi-user study backend verification", False, str(e))

# 7. Check auth configuration endpoint
is_auth_set = is_google_auth_configured()
test("OAuth configuration validator returned boolean", isinstance(is_auth_set, bool))


# ── Test 8: Consent Gate & Permuted Block Randomization ────────────────────────
print("\n[8] Consent Gate & Permuted Block Randomization")

try:
    from core.db import has_user_consented, log_user_consent, get_or_create_user_assignment, save_familiarity_ratings
    
    # Clean queue & consent entries to ensure clean run
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM consent_records WHERE email LIKE '%@test-cohort.org'")
    c.execute("DELETE FROM familiarity_records WHERE email LIKE '%@test-cohort.org'")
    c.execute("UPDATE assignment_queue SET assigned_email = NULL, assigned_time = NULL WHERE assigned_email LIKE '%@test-cohort.org'")
    conn.commit()
    conn.close()

    # Consent Check
    test_email_1 = "p1@test-cohort.org"
    test("Initial consent status is False", has_user_consented(test_email_1) == False)
    log_user_consent(test_email_1)
    test("Consent status is True after logging", has_user_consented(test_email_1) == True)
    
    # Familiarity ratings & lowest-rated topic selection test
    ratings_p1 = {
        "grokking": 5,
        "blindsight": 1,
        "arrows_theorem": 2,
        "olbers_paradox": 3,
        "pyrrhonism": 4,
        "charvaka": 5,
        "piraha_language": 2,
        "antikythera_mechanism": 5
    }
    save_familiarity_ratings(test_email_1, ratings_p1)
    
    # Simulate lowest-rated selection logic:
    # Sort by rating ascending, on tie sort by topic_id (alphabetical)
    sorted_ratings = sorted(ratings_p1.items(), key=lambda x: (x[1], x[0]))
    selected_topics = [item[0] for item in sorted_ratings[:2]]
    test("Lowest rated selected topic 1 is blindsight (rating 1)", selected_topics[0] == "blindsight")
    test("Lowest rated selected topic 2 is arrows_theorem (rating 2, tie-breaker)", selected_topics[1] == "arrows_theorem")
    
    # Block Randomization Assignment Queue check
    mode_1, order_1 = get_or_create_user_assignment("p1@test-cohort.org")
    mode_2, order_2 = get_or_create_user_assignment("p2@test-cohort.org")
    mode_3, order_3 = get_or_create_user_assignment("p3@test-cohort.org")
    mode_4, order_4 = get_or_create_user_assignment("p4@test-cohort.org")
    
    modes = [mode_1, mode_2, mode_3, mode_4]
    test("Block randomization contains exactly 2 SENTIO_FIRST assignments", modes.count("SENTIO_FIRST") == 2, f"got {modes}")
    test("Block randomization contains exactly 2 CONTROL_FIRST assignments", modes.count("CONTROL_FIRST") == 2, f"got {modes}")
    
    orders = [order_1, order_2, order_3, order_4]
    test("Block randomization contains exactly 2 NORMAL_ORDER assignments", orders.count("NORMAL_ORDER") == 2, f"got {orders}")
    test("Block randomization contains exactly 2 SWAPPED_ORDER assignments", orders.count("SWAPPED_ORDER") == 2, f"got {orders}")
    
    # Topic pool static quiz check
    from core.topics import TOPIC_POOL
    test("TOPIC_POOL contains exactly 8 topics", len(TOPIC_POOL) == 8, f"got {len(TOPIC_POOL)}")
    test("Each topic in pool has static pre_quiz with 3 questions", all(len(t["pre_quiz"]) == 3 for t in TOPIC_POOL.values()))
    test("Each topic in pool has static post_quiz with 3 questions", all(len(t["post_quiz"]) == 3 for t in TOPIC_POOL.values()))

except Exception as e:
    test("Consent & Permuted block randomization test verification", False, str(e))


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
