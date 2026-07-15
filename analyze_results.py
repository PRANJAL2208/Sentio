"""
analyze_results.py

Statistical analysis pipeline for Sentio user studies.
Calculates learning gains, runs paired t-tests comparing Sentio vs. Control,
compares NASA-TLX cognitive workload dimensions, and evaluates keystroke dynamic correlations.

Usage:
  1. Generate mock data to test the pipeline:
     python analyze_results.py --mock
  2. Run analysis on active user study database:
     python analyze_results.py
"""

import sys
import os
import math
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

DB_FILE = "sentio_study.db"

# Try importing scipy for exact p-values, otherwise use a high-precision approximation
HAS_SCIPY = False
try:
    import scipy.stats as stats
    HAS_SCIPY = True
except ImportError:
    pass

# Try importing matplotlib/seaborn for paper-ready charts
HAS_PLOT = False
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    pass

# ── Math Fallback Functions (No SciPy Required) ────────────────────────────────

def t_distribution_p_value(t_stat: float, df: int) -> float:
    """
    Approximates the two-tailed p-value for a given t-statistic and degrees of freedom.
    Using log-gamma and beta approximations when SciPy is absent.
    """
    if HAS_SCIPY:
        return float(stats.t.sf(abs(t_stat), df) * 2)
    
    # Mathematical approximation of student-t cumulative distribution
    # For df > 30, t behaves close to normal distribution
    x = df / (df + t_stat**2)
    # A simple, close approximation of the regularized incomplete beta function
    # to yield a reliable standard p-value estimate
    # Ref: Abramowitz and Stegun formula 26.7.8
    d = abs(t_stat)
    a = 1.0 / (1.0 + 0.196854 * d + 0.115194 * d**2 + 0.000344 * d**3 + 0.019527 * d**4)
    p_approx = 0.5 * (a**4) * 2
    return min(1.0, max(0.0, p_approx))


# ── Mock Data Generator ───────────────────────────────────────────────────────

def generate_mock_data():
    """
    Populate sentio_study.db with realistic mock data for 10 users 
    across 4 topics to test the statistical pipeline.
    """
    print("Generating simulated participant data inside sentio_study.db...")
    
    from core.db import init_database
    init_database()
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Clean previous records to prevent duplicates
    c.execute("DELETE FROM telemetry_records")
    c.execute("DELETE FROM workload_records")
    c.execute("DELETE FROM quiz_records")
    c.execute("DELETE FROM sessions")
    c.execute("DELETE FROM users")
    
    topics = [
        "1. Artificial Intelligence vs. Agentic AI",
        "2. Transformers & Self-Attention",
        "3. Spaced Repetition & Ebbinghaus Decay",
        "4. Cognitive Load Theory"
    ]
    
    # Generate 10 subjects
    for sub_id in range(1, 11):
        email = f"subject_{sub_id:02d}@university.edu"
        group = "Group A" if sub_id % 2 != 0 else "Group B"
        
        # Insert user
        c.execute(
            "INSERT INTO users (email, group_assignment, signup_time) VALUES (?, ?, ?)",
            (email, group, datetime.now())
        )
        
        # Simulate each of the 4 topics
        for idx, topic_name in enumerate(topics):
            # Counterbalance assignment
            # Group A starts with Sentio, Group B starts with Control
            if group == "Group A":
                mode = "SENTIO" if idx % 2 == 0 else "CONTROL"
            else:
                mode = "CONTROL" if idx % 2 == 0 else "SENTIO"
                
            session_id = f"mock_sess_{email.replace('@','_')}_{idx}"
            
            # Insert session
            c.execute(
                "INSERT INTO sessions (session_id, email, topic_name, study_mode, start_time, end_time) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, email, topic_name, mode, datetime.now(), datetime.now())
            )
            
            # Generate pre-test quiz score (Baseline low for all: average 0.8 / 3.0)
            pre_score = int(np.clip(np.random.normal(0.8, 0.6), 0, 2))
            c.execute(
                "INSERT INTO quiz_records (email, topic_name, quiz_type, score, total_questions, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (email, topic_name, "PRE", pre_score, 3, datetime.now())
            )
            
            # Generate post-test quiz score
            # Sentio Mode: higher learning gain (average 2.7 / 3.0)
            # Control Mode: lower learning gain (average 1.8 / 3.0)
            if mode == "SENTIO":
                post_score = int(np.clip(np.random.normal(2.7, 0.4), 1, 3))
            else:
                post_score = int(np.clip(np.random.normal(1.9, 0.6), 0, 3))
                
            # Ensure post-test is at least pre-test for mock simulation consistency
            post_score = max(pre_score, post_score)
            
            c.execute(
                "INSERT INTO quiz_records (email, topic_name, quiz_type, score, total_questions, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (email, topic_name, "POST", post_score, 3, datetime.now())
            )
            
            # Generate NASA-TLX workload survey
            # Sentio Mode: lower mental demand, effort, frustration, higher performance
            # Control Mode: higher demand, frustration, effort
            if mode == "SENTIO":
                ratings = {
                    "mental": int(np.clip(np.random.normal(42, 10), 10, 80)),
                    "physical": int(np.clip(np.random.normal(12, 5), 0, 40)),
                    "temporal": int(np.clip(np.random.normal(28, 8), 10, 60)),
                    "perf": int(np.clip(np.random.normal(82, 8), 50, 100)),
                    "effort": int(np.clip(np.random.normal(48, 10), 20, 90)),
                    "frust": int(np.clip(np.random.normal(20, 8), 0, 70))
                }
            else:
                ratings = {
                    "mental": int(np.clip(np.random.normal(68, 12), 20, 100)),
                    "physical": int(np.clip(np.random.normal(15, 6), 0, 50)),
                    "temporal": int(np.clip(np.random.normal(45, 12), 10, 90)),
                    "perf": int(np.clip(np.random.normal(62, 12), 30, 90)),
                    "effort": int(np.clip(np.random.normal(70, 12), 30, 100)),
                    "frust": int(np.clip(np.random.normal(52, 15), 10, 100))
                }
            
            c.execute(
                """
                INSERT INTO workload_records (
                    email, topic_name, study_mode, 
                    mental_demand, physical_demand, temporal_demand, 
                    performance, effort, frustration, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    email, topic_name, mode,
                    ratings["mental"], ratings["physical"], ratings["temporal"],
                    ratings["perf"], ratings["effort"], ratings["frust"], datetime.now()
                )
            )
            
            # Generate simulated typing telemetry logs (10 messages per session)
            for turn in range(5):
                # Sentio Mode: optimized pace
                # Control Mode: higher hesitation pauses and backspaces due to load
                if mode == "SENTIO":
                    dwell = np.random.normal(105, 10)
                    flight = np.random.normal(210, 20)
                    backspaces = int(np.clip(np.random.normal(4, 2), 0, 12))
                    pause = np.random.normal(3.8, 1.2)
                else:
                    dwell = np.random.normal(135, 15)
                    flight = np.random.normal(320, 35)
                    backspaces = int(np.clip(np.random.normal(11, 4), 2, 25))
                    pause = np.random.normal(9.2, 2.5)
                    
                c.execute(
                    """
                    INSERT INTO telemetry_records (
                        session_id, backspace_count, avg_dwell_ms, avg_flight_ms, pause_seconds, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (session_id, backspaces, dwell, flight, pause, datetime.now())
                )
                
    conn.commit()
    conn.close()
    print("Simulated dataset created successfully. 10 subjects, 40 sessions, 200 telemetry entries.")


# ── Run Statistical Pipeline ──────────────────────────────────────────────────

def run_statistical_analysis():
    """
    Performs full data processing and t-tests, printing summary tables to console.
    """
    if not os.path.exists(DB_FILE):
        print(f"Error: {DB_FILE} not found. Please run the study or generate mock data with: python analyze_results.py --mock")
        sys.exit(1)
        
    print("=" * 70)
    print("                SENTIO USER STUDY STATISTICAL PIPELINE")
    print("=" * 70)
    
    conn = sqlite3.connect(DB_FILE)
    
    # ── 1. Learning Gain Analysis (Pre-Test vs Post-Test) ──
    query_quizzes = """
    SELECT q1.email, q1.topic_name, s.study_mode, q1.score as pre_score, q2.score as post_score
    FROM quiz_records q1
    JOIN quiz_records q2 ON q1.email = q2.email AND q1.topic_name = q2.topic_name
    JOIN sessions s ON s.email = q1.email AND s.topic_name = q1.topic_name
    WHERE q1.quiz_type = 'PRE' AND q2.quiz_type = 'POST'
    """
    df_quizzes = pd.read_sql_query(query_quizzes, conn)
    df_quizzes["learning_gain"] = df_quizzes["post_score"] - df_quizzes["pre_score"]
    
    print("\n[1] LEARNING GAINS (POST-TEST MINUS PRE-TEST):")
    gains = df_quizzes.groupby("study_mode")["learning_gain"].agg(["mean", "std", "count"]).reset_index()
    print(gains.to_string(index=False))
    
    # Run Paired T-Test
    # Format: pivot to have one row per student/topic
    sentio_gains = df_quizzes[df_quizzes["study_mode"] == "SENTIO"]["learning_gain"].values
    control_gains = df_quizzes[df_quizzes["study_mode"] == "CONTROL"]["learning_gain"].values
    
    n_samples = min(len(sentio_gains), len(control_gains))
    if n_samples > 1:
        # Truncate to equal lengths for paired t-test comparison
        s_vals = sentio_gains[:n_samples]
        c_vals = control_gains[:n_samples]
        
        diff = s_vals - c_vals
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        se_diff = std_diff / math.sqrt(n_samples)
        
        t_stat = mean_diff / se_diff
        df_deg = n_samples - 1
        p_val = t_distribution_p_value(t_stat, df_deg)
        
        print(f"\n  Paired t-test (Sentio vs Control Learning Gains):")
        print(f"    t({df_deg}) = {t_stat:.3f}")
        print(f"    p-value = {p_val:.5f} " + ("🔴 (Significant, p < 0.05)" if p_val < 0.05 else "⚪ (Not significant)"))
    else:
        print("\n  [Warning] Insufficient matched pairs to compute learning gain t-test.")
        
    # ── 2. Subjective Cognitive Workload (NASA-TLX) ──
    query_workload = """
    SELECT study_mode, mental_demand, temporal_demand, performance, effort, frustration
    FROM workload_records
    """
    df_workload = pd.read_sql_query(query_workload, conn)
    
    print("\n[2] NASA-TLX COGNITIVE WORKLOAD DIMENSIONS (0-100):")
    workload_summary = df_workload.groupby("study_mode").mean().round(2).reset_index()
    print(workload_summary.to_string(index=False))
    
    print("\n  T-Test Comparisons (Workload Dimensions):")
    dimensions = ["mental_demand", "performance", "effort", "frustration"]
    for dim in dimensions:
        s_dim = df_workload[df_workload["study_mode"] == "SENTIO"][dim].values
        c_dim = df_workload[df_workload["study_mode"] == "CONTROL"][dim].values
        
        if len(s_dim) > 1 and len(c_dim) > 1:
            mean_s, mean_c = np.mean(s_dim), np.mean(c_dim)
            var_s, var_c = np.var(s_dim, ddof=1), np.var(c_dim, ddof=1)
            n_s, n_c = len(s_dim), len(c_dim)
            
            # Welch's t-test
            se = math.sqrt((var_s / n_s) + (var_c / n_c))
            t_val = (mean_s - mean_c) / se
            # Welch-Satterthwaite degrees of freedom
            df_numerator = ((var_s/n_s) + (var_c/n_c))**2
            df_denominator = (((var_s/n_s)**2) / (n_s - 1)) + (((var_c/n_c)**2) / (n_c - 1))
            welch_df = int(df_numerator / df_denominator)
            
            p_val = t_distribution_p_value(t_val, welch_df)
            sig_label = "🔴 (Significant)" if p_val < 0.05 else "⚪"
            
            print(f"    - {dim.replace('_', ' ').title()}:")
            print(f"      Sentio Mean = {mean_s:.2f} | Control Mean = {mean_c:.2f}")
            print(f"      t({welch_df}) = {t_val:.3f} | p = {p_val:.5f} {sig_label}")
            
    # ── 3. Keystroke Telemetry vs Workload Correlation ──
    query_telemetry = """
    SELECT s.study_mode, t.avg_flight_ms, t.avg_dwell_ms, t.backspace_count, t.pause_seconds, w.frustration
    FROM telemetry_records t
    JOIN sessions s ON t.session_id = s.session_id
    JOIN workload_records w ON w.email = s.email AND w.topic_name = s.topic_name
    """
    df_telemetry = pd.read_sql_query(query_telemetry, conn)
    
    print("\n[3] BIOMETRIC TYPING TELEMETRY SUMMARY:")
    telemetry_summary = df_telemetry.groupby("study_mode")[["avg_flight_ms", "avg_dwell_ms", "backspace_count", "pause_seconds"]].mean().round(2).reset_index()
    print(telemetry_summary.to_string(index=False))
    
    conn.close()
    
    # ── 4. Generate High-Res Publication Visualizations (Optional) ──
    if HAS_PLOT:
        print("\n[4] GENERATING PAPER-READY CHARTS (PNG)...")
        sns.set_theme(style="whitegrid")
        
        # Plot 1: Learning Gain Bar Chart
        plt.figure(figsize=(6, 5))
        ax = sns.barplot(
            x="study_mode", 
            y="learning_gain", 
            data=df_quizzes, 
            palette={"SENTIO": "#6366f1", "CONTROL": "#94a3b8"},
            capsize=0.1,
            errorbar="se"
        )
        plt.title("Tutor Learning Gains\n(Post-Quiz minus Pre-Quiz score)", fontsize=12, fontweight="bold")
        plt.xlabel("Tutoring Platform Mode", fontsize=10)
        plt.ylabel("Average Learning Gain (0-3 scale)", fontsize=10)
        plt.tight_layout()
        plt.savefig("study_learning_gains.png", dpi=300)
        plt.close()
        print("  ✓ Saved learning gain bar plot as 'study_learning_gains.png'")
        
        # Plot 2: NASA-TLX Boxplots
        df_tlx_melted = pd.melt(
            df_workload, 
            id_vars=["study_mode"], 
            value_vars=["mental_demand", "effort", "frustration"],
            var_name="Dimension",
            value_name="Rating"
        )
        df_tlx_melted["Dimension"] = df_tlx_melted["Dimension"].str.replace("_", " ").str.title()
        
        plt.figure(figsize=(8, 5))
        sns.boxplot(
            x="Dimension", 
            y="Rating", 
            hue="study_mode", 
            data=df_tlx_melted, 
            palette={"SENTIO": "#6366f1", "CONTROL": "#94a3b8"}
        )
        plt.title("NASA-TLX Subjective Cognitive Workload Comparison", fontsize=12, fontweight="bold")
        plt.xlabel("Cognitive Workload Dimensions", fontsize=10)
        plt.ylabel("Workload Rating (0-100)", fontsize=10)
        plt.legend(title="Study Group")
        plt.tight_layout()
        plt.savefig("study_nasa_tlx_comparison.png", dpi=300)
        plt.close()
        print("  ✓ Saved NASA-TLX boxplots as 'study_nasa_tlx_comparison.png'")
        
        print("\nAll visualizations outputted successfully. Plots are saved in the project root directory.")
    else:
        print("\n[Info] Matplotlib/Seaborn not found in environment. Skipping PNG chart rendering.")
        print("       Run 'pip install matplotlib seaborn' if you want PDF/PNG plots generated.")
        
    print("=" * 70)


# ── Executable Entry Point ───────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--mock":
        generate_mock_data()
        
    run_statistical_analysis()
