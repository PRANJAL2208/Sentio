"""
core/db.py

SQLite Database Engine configured in Write-Ahead Logging (WAL) Mode.
This manages highly concurrent logging of user study metadata, quiz scores, 
NASA-TLX surveys, and typing dynamics without write locks under multi-user traffic.
"""

import sqlite3
import os
import requests
import streamlit as st
from datetime import datetime

DB_FILE = "sentio_study.db"

# Retrieve Supabase cloud configs
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

try:
    if not SUPABASE_URL:
        SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
    if not SUPABASE_KEY:
        SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")
except Exception:
    pass

def is_supabase_enabled() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)

def get_supabase_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

def supabase_post(table: str, data: dict):
    """
    Synchronously writes a log entry to remote Supabase DB via REST API.
    Does not raise exceptions so study participants are never blocked on network issues.
    """
    if not is_supabase_enabled():
        return
    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{table}"
    try:
        payload = {}
        for k, v in data.items():
            if isinstance(v, datetime):
                payload[k] = v.isoformat()
            else:
                payload[k] = v
        requests.post(url, headers=get_supabase_headers(), json=payload, timeout=5.0)
    except Exception:
        pass

def supabase_patch(table: str, filters: dict, data: dict):
    """
    Sends a PATCH update request to Supabase REST API.
    """
    if not is_supabase_enabled():
        return
    query_str = "&".join([f"{k}=eq.{v}" for k, v in filters.items()])
    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{table}?{query_str}"
    try:
        payload = {}
        for k, v in data.items():
            if isinstance(v, datetime):
                payload[k] = v.isoformat()
            else:
                payload[k] = v
        requests.patch(url, headers=get_supabase_headers(), json=payload, timeout=5.0)
    except Exception:
        pass

def supabase_get(table: str) -> list:
    """
    Fetch all records from the specified Supabase table.
    """
    if not is_supabase_enabled():
        return []
    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{table}?select=*"
    try:
        res = requests.get(url, headers=get_supabase_headers(), timeout=8.0)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []

def get_connection():
    """
    Establish a thread-safe connection to the SQLite database.
    Enables WAL mode and sets appropriate timeouts for high concurrency.
    """
    conn = sqlite3.connect(DB_FILE, timeout=30.0)
    # Enable WAL mode for concurrent readers and writers
    conn.execute("PRAGMA journal_mode=WAL;")
    # Ensure foreign keys are enforced
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def init_database():
    """
    Initialize SQL tables and indices for multi-user study logging.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # 1. Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        email TEXT PRIMARY KEY,
        group_assignment TEXT NOT NULL,
        signup_time TIMESTAMP NOT NULL
    );
    """)

    # 2. Study Sessions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        email TEXT NOT NULL,
        topic_name TEXT NOT NULL,
        study_mode TEXT NOT NULL,
        start_time TIMESTAMP NOT NULL,
        end_time TIMESTAMP,
        FOREIGN KEY (email) REFERENCES users (email) ON DELETE CASCADE
    );
    """)

    # 3. Quiz Records Table (Pre-test / Post-test scores)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS quiz_records (
        record_id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT NOT NULL,
        topic_name TEXT NOT NULL,
        quiz_type TEXT NOT NULL, -- 'PRE' or 'POST'
        score INTEGER NOT NULL,
        total_questions INTEGER NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        FOREIGN KEY (email) REFERENCES users (email) ON DELETE CASCADE
    );
    """)

    # 4. Workload Records Table (NASA-TLX sliders)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS workload_records (
        record_id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT NOT NULL,
        topic_name TEXT NOT NULL,
        study_mode TEXT NOT NULL,
        mental_demand INTEGER NOT NULL,
        physical_demand INTEGER NOT NULL,
        temporal_demand INTEGER NOT NULL,
        performance INTEGER NOT NULL,
        effort INTEGER NOT NULL,
        frustration INTEGER NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        FOREIGN KEY (email) REFERENCES users (email) ON DELETE CASCADE
    );
    """)

    # 5. Telemetry Records Table (Keystroke dynamics summary per message)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS telemetry_records (
        record_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        backspace_count INTEGER NOT NULL,
        avg_dwell_ms REAL NOT NULL,
        avg_flight_ms REAL NOT NULL,
        pause_seconds REAL NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
    );
    """)

    conn.commit()
    conn.close()

# ── Logging Functions ────────────────────────────────────────────────────────

def register_user(email: str, group: str):
    """
    Log in a user. Registers them with their random group assignment if new.
    """
    conn = get_connection()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO users (email, group_assignment, signup_time) VALUES (?, ?, ?)",
            (email.lower().strip(), group, datetime.now())
        )
        conn.commit()
    finally:
        conn.close()
    
    # Mirror to Supabase cloud
    supabase_post("users", {
        "email": email.lower().strip(),
        "group_assignment": group,
        "signup_time": datetime.now()
    })

def get_user_group(email: str) -> str:
    """
    Retrieve the group assignment for a registered user email.
    If Supabase is enabled, we check both SQLite and fallback to Supabase check.
    """
    # 1. Check local db first
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT group_assignment FROM users WHERE email = ?", (email.lower().strip(),))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return row[0]
        
    # 2. Check Supabase cloud database if not found locally
    if is_supabase_enabled():
        try:
            url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/users?email=eq.{email.lower().strip()}&select=group_assignment"
            res = requests.get(url, headers=get_supabase_headers(), timeout=5.0)
            if res.status_code == 200 and res.json():
                return res.json()[0].get("group_assignment")
        except Exception:
            pass
    return None

def log_session_start(session_id: str, email: str, topic_name: str, mode: str):
    """
    Create a new study session.
    """
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO sessions (session_id, email, topic_name, study_mode, start_time) VALUES (?, ?, ?, ?, ?)",
            (session_id, email.lower().strip(), topic_name, mode, datetime.now())
        )
        conn.commit()
    finally:
        conn.close()
        
    supabase_post("sessions", {
        "session_id": session_id,
        "email": email.lower().strip(),
        "topic_name": topic_name,
        "study_mode": mode,
        "start_time": datetime.now()
    })

def log_session_end(session_id: str):
    """
    Mark the study session as completed.
    """
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE sessions SET end_time = ? WHERE session_id = ?",
            (datetime.now(), session_id)
        )
        conn.commit()
    finally:
        conn.close()
        
    supabase_patch("sessions", {"session_id": session_id}, {"end_time": datetime.now()})

def log_quiz(email: str, topic_name: str, quiz_type: str, score: int, total: int):
    """
    Write a Pre-test or Post-test score to database.
    """
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO quiz_records (email, topic_name, quiz_type, score, total_questions, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (email.lower().strip(), topic_name, quiz_type.upper().strip(), score, total, datetime.now())
        )
        conn.commit()
    finally:
        conn.close()
        
    supabase_post("quiz_records", {
        "email": email.lower().strip(),
        "topic_name": topic_name,
        "quiz_type": quiz_type.upper().strip(),
        "score": score,
        "total_questions": total,
        "timestamp": datetime.now()
    })

def log_workload(email: str, topic_name: str, mode: str, ratings: dict):
    """
    Write a 6-item NASA-TLX workload survey response to database.
    """
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO workload_records (
                email, topic_name, study_mode, 
                mental_demand, physical_demand, temporal_demand, 
                performance, effort, frustration, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                email.lower().strip(), topic_name, mode,
                ratings.get("mental_demand", 0),
                ratings.get("physical_demand", 0),
                ratings.get("temporal_demand", 0),
                ratings.get("performance", 0),
                ratings.get("effort", 0),
                ratings.get("frustration", 0),
                datetime.now()
            )
        )
        conn.commit()
    finally:
        conn.close()
        
    supabase_post("workload_records", {
        "email": email.lower().strip(),
        "topic_name": topic_name,
        "study_mode": mode,
        "mental_demand": ratings.get("mental_demand", 0),
        "physical_demand": ratings.get("physical_demand", 0),
        "temporal_demand": ratings.get("temporal_demand", 0),
        "performance": ratings.get("performance", 0),
        "effort": ratings.get("effort", 0),
        "frustration": ratings.get("frustration", 0),
        "timestamp": datetime.now()
    })

def log_telemetry(session_id: str, telemetry: dict):
    """
    Log typing timing dynamic variables for a chat turn.
    """
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO telemetry_records (
                session_id, backspace_count, avg_dwell_ms, avg_flight_ms, pause_seconds, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                telemetry.get("backspace_count", 0),
                telemetry.get("avg_dwell_ms", 0.0),
                telemetry.get("avg_flight_ms", 0.0),
                telemetry.get("pause_seconds", 0.0),
                datetime.now()
            )
        )
        conn.commit()
    finally:
        conn.close()
        
    supabase_post("telemetry_records", {
        "session_id": session_id,
        "backspace_count": telemetry.get("backspace_count", 0),
        "avg_dwell_ms": telemetry.get("avg_dwell_ms", 0.0),
        "avg_flight_ms": telemetry.get("avg_flight_ms", 0.0),
        "pause_seconds": telemetry.get("pause_seconds", 0.0),
        "timestamp": datetime.now()
    })

def get_all_users() -> list:
    """
    Returns all registered user records as a list of dicts.
    """
    if is_supabase_enabled():
        return supabase_get("users")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT email, group_assignment, signup_time FROM users")
    rows = cursor.fetchall()
    conn.close()
    return [{"email": r[0], "group_assignment": r[1], "signup_time": r[2]} for r in rows]

def get_all_sessions() -> list:
    """
    Returns all study sessions as a list of dicts.
    """
    if is_supabase_enabled():
        return supabase_get("sessions")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT session_id, email, topic_name, study_mode, start_time, end_time FROM sessions")
    rows = cursor.fetchall()
    conn.close()
    return [{"session_id": r[0], "email": r[1], "topic_name": r[2], "study_mode": r[3], "start_time": r[4], "end_time": r[5]} for r in rows]

def get_all_quizzes() -> list:
    """
    Returns all pre-test and post-test quiz results as a list of dicts.
    """
    if is_supabase_enabled():
        return supabase_get("quiz_records")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT email, topic_name, quiz_type, score, total_questions, timestamp FROM quiz_records")
    rows = cursor.fetchall()
    conn.close()
    return [{"email": r[0], "topic_name": r[1], "quiz_type": r[2], "score": r[3], "total_questions": r[4], "timestamp": r[5]} for r in rows]

def get_all_workloads() -> list:
    """
    Returns all NASA-TLX workload records as a list of dicts.
    """
    if is_supabase_enabled():
        return supabase_get("workload_records")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT email, topic_name, study_mode, mental_demand, physical_demand, temporal_demand, performance, effort, frustration, timestamp FROM workload_records")
    rows = cursor.fetchall()
    conn.close()
    return [{
        "email": r[0], "topic_name": r[1], "study_mode": r[2],
        "mental_demand": r[3], "physical_demand": r[4], "temporal_demand": r[5],
        "performance": r[6], "effort": r[7], "frustration": r[8], "timestamp": r[9]
    } for r in rows]

def get_all_telemetry() -> list:
    """
    Returns all keystroke timing telemetry entries as a list of dicts.
    """
    if is_supabase_enabled():
        return supabase_get("telemetry_records")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT session_id, backspace_count, avg_dwell_ms, avg_flight_ms, pause_seconds, timestamp FROM telemetry_records")
    rows = cursor.fetchall()
    conn.close()
    return [{
        "session_id": r[0], "backspace_count": r[1], "avg_dwell_ms": r[2],
        "avg_flight_ms": r[3], "pause_seconds": r[4], "timestamp": r[5]
    } for r in rows]
