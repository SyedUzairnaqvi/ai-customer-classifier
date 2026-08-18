import os
from typing import Optional

import pandas as pd
import mysql.connector

try:
    import streamlit as st
except Exception:
    st = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(BASE_DIR, "schema.sql")


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
    config = {"host": _secret("MYSQL_HOST", "localhost"), "port": int(_secret("MYSQL_PORT", "3306")), "user": _secret("MYSQL_USER"), "password": _secret("MYSQL_PASSWORD"), "connection_timeout": 8}
    if include_database:
        config["database"] = _secret("MYSQL_DATABASE")
    return config


def get_connection():
    if not is_database_configured():
        raise RuntimeError("MySQL is not configured. Set MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD and MYSQL_DATABASE.")
    return mysql.connector.connect(**_config())


def init_database() -> bool:
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
            statements = [s.strip() for s in file.read().split(";") if s.strip()]
        for statement in statements:
            cursor.execute(statement)
        conn.commit()
        return True
    finally:
        if cursor: cursor.close()
        if conn: conn.close()


def insert_analysis(message: str, intent: str, intent_confidence: float, sentiment: str, sentiment_confidence: float, urgency: str, routing_status: str) -> None:
    conn = get_connection(); cursor = conn.cursor()
    try:
        cursor.execute("""INSERT INTO customer_analyses (message, intent, intent_confidence, sentiment, sentiment_confidence, urgency, routing_status) VALUES (%s, %s, %s, %s, %s, %s, %s)""", (message, intent, intent_confidence, sentiment, sentiment_confidence, urgency, routing_status))
        conn.commit()
    finally:
        cursor.close(); conn.close()


def _read_query(query: str, params: Optional[tuple] = None) -> pd.DataFrame:
    conn = get_connection(); cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(query, params or ())
        return pd.DataFrame(cursor.fetchall())
    finally:
        cursor.close(); conn.close()


def fetch_analytics() -> pd.DataFrame:
    return _read_query("SELECT analysis_id, created_at, message, intent, intent_confidence, sentiment, sentiment_confidence, urgency, routing_status FROM customer_analyses ORDER BY created_at DESC")


def fetch_recent_analyses(limit: int = 25) -> pd.DataFrame:
    limit = max(1, min(int(limit), 500))
    return _read_query(f"SELECT analysis_id, created_at, message, intent, intent_confidence, sentiment, urgency, routing_status FROM customer_analyses ORDER BY created_at DESC LIMIT {limit}")
