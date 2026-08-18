import os
from pathlib import Path
from typing import Optional

import certifi
import mysql.connector
import pandas as pd

try:
    import streamlit as st
except Exception:
    st = None

BASE_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = BASE_DIR / "schema.sql"
BATCH_SCHEMA_PATH = BASE_DIR / "batch_schema.sql"


def _secret(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is not None:
        return value
    if st is not None:
        try:
            value = st.secrets.get(name)
            if value is not None:
                return str(value)
        except Exception:
            pass
    return default


def is_database_configured() -> bool:
    required = ["MYSQL_HOST", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE"]
    return all(_secret(key) for key in required)


def _config(include_database: bool = True) -> dict:
    user = _secret("MYSQL_USER")
    if user == "root":
        user = "3CnPSAnxy3nbikJ.root"

    config = {
        "host": _secret("MYSQL_HOST", "localhost"),
        "port": int(_secret("MYSQL_PORT", "3306")),
        "user": user,
        "password": _secret("MYSQL_PASSWORD"),
        "connection_timeout": int(_secret("MYSQL_CONNECT_TIMEOUT", "10")),
        "use_pure": True,
    }
    if include_database:
        config["database"] = _secret("MYSQL_DATABASE")

    ca_path = _secret("MYSQL_SSL_CA")
    if not ca_path or not Path(ca_path).is_file():
        ca_path = certifi.where()
    config.update({
        "ssl_ca": ca_path,
        "ssl_verify_cert": True,
        "ssl_verify_identity": True,
        "tls_versions": ["TLSv1.2", "TLSv1.3"],
    })
    return config


def get_connection():
    if not is_database_configured():
        raise RuntimeError("MySQL is not configured. Set MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD and MYSQL_DATABASE.")
    return mysql.connector.connect(**_config())


def _run_schema_file(cursor, path: Path) -> None:
    for statement in path.read_text(encoding="utf-8").split(";"):
        statement = statement.strip()
        if statement:
            cursor.execute(statement)


def init_database() -> bool:
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        _run_schema_file(cursor, SCHEMA_PATH)
        if BATCH_SCHEMA_PATH.exists():
            _run_schema_file(cursor, BATCH_SCHEMA_PATH)
        conn.commit()
        return True
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def insert_analysis(message: str, intent: str, intent_confidence: float, sentiment: str, sentiment_confidence: float, urgency: str, routing_status: str, batch_id: Optional[str] = None) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """INSERT INTO customer_analyses
            (message, intent, intent_confidence, sentiment, sentiment_confidence, urgency, routing_status, batch_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (message, intent, intent_confidence, sentiment, sentiment_confidence, urgency, routing_status, batch_id),
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()


def create_batch(batch_id: str, source_name: str, total_rows: int) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO analysis_batches (batch_id, source_name, total_rows, processed_rows, status) VALUES (%s, %s, %s, 0, 'Processing')",
            (batch_id, source_name[:255], total_rows),
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()


def bulk_insert_analyses(rows: list[dict], batch_id: str, chunk_size: int = 1000) -> int:
    if not rows:
        return 0
    conn = get_connection()
    cursor = conn.cursor()
    sql = """INSERT INTO customer_analyses
        (message, intent, intent_confidence, sentiment, sentiment_confidence, urgency, routing_status, batch_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""
    inserted = 0
    try:
        values = [(
            row["message"], row["intent"], row["intent_confidence"], row["sentiment"],
            row["sentiment_confidence"], row["urgency"], row["routing_status"], batch_id
        ) for row in rows]
        for start in range(0, len(values), chunk_size):
            chunk = values[start:start + chunk_size]
            cursor.executemany(sql, chunk)
            inserted += len(chunk)
        conn.commit()
        return inserted
    finally:
        cursor.close()
        conn.close()


def update_batch(batch_id: str, processed_rows: int, status: str) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE analysis_batches SET processed_rows=%s, status=%s WHERE batch_id=%s", (processed_rows, status, batch_id))
        conn.commit()
    finally:
        cursor.close()
        conn.close()


def fetch_batch_history(limit: int = 20) -> pd.DataFrame:
    limit = max(1, min(int(limit), 100))
    return _read_query(f"""SELECT batch_id, created_at, source_name, total_rows, processed_rows, status
        FROM analysis_batches ORDER BY created_at DESC LIMIT {limit}""")


def fetch_dashboard_data() -> dict[str, object]:
    """Return full-history KPIs/groupings using SQL aggregation, not Python row loading."""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""SELECT COUNT(*) AS total,
            SUM(urgency='High') AS high,
            SUM(sentiment='Negative') AS negative,
            SUM(routing_status='Auto-Routable') AS auto_routable,
            COALESCE(AVG(intent_confidence), 0) AS avg_confidence
            FROM customer_analyses""")
        kpi = cursor.fetchone() or {}

        queries = {
            "intent": "SELECT intent, COUNT(*) AS count FROM customer_analyses GROUP BY intent ORDER BY count DESC",
            "sentiment": "SELECT sentiment, COUNT(*) AS count FROM customer_analyses GROUP BY sentiment ORDER BY count DESC",
            "urgency": "SELECT urgency, COUNT(*) AS count FROM customer_analyses GROUP BY urgency ORDER BY count DESC",
            "routing": "SELECT routing_status, COUNT(*) AS count FROM customer_analyses GROUP BY routing_status ORDER BY count DESC",
        }
        groups = {}
        for key, query in queries.items():
            cursor.execute(query)
            groups[key] = pd.DataFrame(cursor.fetchall())
        total = int(kpi.get("total") or 0)
        return {
            "total": total,
            "high": int(kpi.get("high") or 0),
            "negative": int(kpi.get("negative") or 0),
            "auto_routable": int(kpi.get("auto_routable") or 0),
            "avg_confidence": float(kpi.get("avg_confidence") or 0),
            **groups,
        }
    finally:
        cursor.close()
        conn.close()


def _read_query(query: str, params: Optional[tuple] = None) -> pd.DataFrame:
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(query, params or ())
        return pd.DataFrame(cursor.fetchall())
    finally:
        cursor.close()
        conn.close()


def fetch_analytics() -> pd.DataFrame:
    return _read_query("""SELECT analysis_id, created_at, message, intent, intent_confidence, sentiment, sentiment_confidence, urgency, routing_status, batch_id FROM customer_analyses ORDER BY created_at DESC""")


def fetch_recent_analyses(limit: int = 5000) -> pd.DataFrame:
    limit = max(1, min(int(limit), 5000))
    return _read_query(f"""SELECT analysis_id, created_at, message, intent, intent_confidence, sentiment, urgency, routing_status, batch_id FROM customer_analyses ORDER BY created_at DESC LIMIT {limit}""")
