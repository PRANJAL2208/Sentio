"""
core/db.py

SQLite Database Engine configured in Write-Ahead Logging (WAL) Mode.
This manages highly concurrent logging of user study metadata, quiz scores, 
NASA-TLX surveys, and typing dynamics without write locks under multi-user traffic.
"""

import sqlite3
import os
from datetime import datetime

DB_FILE = "sentio_study.db"

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

def get_user_group(email: str) -> str:
    """
    Retrieve the group assignment for a registered user email.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT group_assignment FROM users WHERE email = ?", (email.lower().strip(),))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

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

def log_workload(email: str, topic_name: str, mode: str, ratings: dict):
    """
    Write a 6-item NASA-TLX survey response to database.
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
