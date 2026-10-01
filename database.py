"""
Database Module for Vision-Based Distracted Driver Detection System.
Handles SQLite storage, alert logging, historical query filtering, statistics, and CSV export.
"""

import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database')
DB_PATH = os.path.join(DB_DIR, 'driver_alerts.db')


def get_db_connection() -> sqlite3.Connection:
    """Returns a connection to the SQLite database with Row factory enabled."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initializes the database schema if not already created."""
    os.makedirs(DB_DIR, exist_ok=True)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                duration REAL NOT NULL,
                details TEXT,
                status TEXT NOT NULL DEFAULT 'Triggered'
            )
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts (timestamp)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts (severity)
        ''')
        conn.commit()


def insert_alert(alert_type: str, severity: str, duration: float, details: str = "", status: str = "Triggered") -> int:
    """
    Inserts a newly confirmed alert event into the database.
    
    Args:
        alert_type: e.g. 'Eye Closure', 'Looking Left', 'Single-Hand Driving', 'Yawning'
        severity: 'INFO', 'WARNING', or 'CRITICAL'
        duration: Duration in seconds for which threshold was met
        details: Extra context or landmark details
        status: 'Triggered' or 'Resolved'
    
    Returns:
        The row ID of the inserted alert.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO alerts (timestamp, alert_type, severity, duration, details, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (now_str, alert_type, severity, round(float(duration), 2), details, status))
        conn.commit()
        return cursor.lastrowid or 0


def get_alerts(limit: int = 100, severity: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves the recent alerts from the database."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if severity:
            cursor.execute('''
                SELECT id, timestamp, alert_type, severity, duration, details, status
                FROM alerts
                WHERE severity = ?
                ORDER BY id DESC
                LIMIT ?
            ''', (severity.upper(), limit))
        else:
            cursor.execute('''
                SELECT id, timestamp, alert_type, severity, duration, details, status
                FROM alerts
                ORDER BY id DESC
                LIMIT ?
            ''', (limit,))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def clear_alerts() -> bool:
    """Clears all alert history from the database."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM alerts')
        conn.commit()
        return True


def get_statistics() -> Dict[str, Any]:
    """
    Computes real-time statistics across all recorded alert events.
    Returns totals, severity breakdowns, alert category counts, and max durations.
    """
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Total counts
        cursor.execute('SELECT COUNT(*) FROM alerts')
        total_alerts = cursor.fetchone()[0]
        
        # Severity breakdown
        cursor.execute("SELECT COUNT(*) FROM alerts WHERE severity = 'CRITICAL'")
        critical_alerts = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM alerts WHERE severity = 'WARNING'")
        warning_alerts = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM alerts WHERE severity = 'INFO'")
        info_alerts = cursor.fetchone()[0]
        
        # Max durations
        cursor.execute("SELECT MAX(duration) FROM alerts WHERE alert_type LIKE '%Eye%' OR alert_type LIKE '%Drowsiness%'")
        longest_eye = cursor.fetchone()[0] or 0.0
        
        cursor.execute("SELECT MAX(duration) FROM alerts WHERE alert_type LIKE '%Looking%' OR alert_type LIKE '%Head%'")
        longest_head = cursor.fetchone()[0] or 0.0
        
        # Counts per alert type for Chart.js
        cursor.execute('''
            SELECT alert_type, COUNT(*) as count
            FROM alerts
            GROUP BY alert_type
            ORDER BY count DESC
        ''')
        type_counts = {row['alert_type']: row['count'] for row in cursor.fetchall()}
        
        return {
            'total_alerts': total_alerts,
            'critical_alerts': critical_alerts,
            'warning_alerts': warning_alerts,
            'info_alerts': info_alerts,
            'longest_eye_closure': round(float(longest_eye), 2),
            'longest_head_turn': round(float(longest_head), 2),
            'type_distribution': type_counts
        }


def export_csv_string() -> str:
    """Exports all alerts into a formatted CSV string."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, timestamp, alert_type, severity, duration, details, status FROM alerts ORDER BY id ASC')
        rows = cursor.fetchall()
        
        lines = ["ID,Timestamp,Alert Type,Severity,Duration (s),Details,Status"]
        for row in rows:
            clean_details = str(row['details']).replace('"', '""')
            lines.append(f'{row["id"]},"{row["timestamp"]}","{row["alert_type"]}","{row["severity"]}",{row["duration"]},"{clean_details}","{row["status"]}"')
        return "\n".join(lines)


# Ensure DB exists on import
init_db()
