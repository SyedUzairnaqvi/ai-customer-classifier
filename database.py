import os
from typing import Optional

import pandas as pd
import mysql.connector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(BASE_DIR, "schema.sql")


def is_database_configured() -> bool:
    required = ["MYSQL_HOST", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE"]
    return all(os.getenv(key) is not None for key in required)


def _config(include_database: bool = True) -> dict:
    config = {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER", ""),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "connection_timeout": 8,
    }
    if include_database:
        config["database"] = os.getenv("MYSQL_DATABASE", "")
    return config


def get_connection():
    if not is_database_configured():
        raise RuntimeError(
            "MySQL is not configured. Set MYSQL_HOST, MYSQL_PORT, "
            "MYSQL_USER, MYSQL_PASSWORD and MYSQL_DATABASE."
        )
    return mysql.connector.connect(**_config())


def init_database() -> bool:
    """Create the analytics table and Power BI views if they do not exist."""
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
            statements = [
                statement.strip()
                for statement in file.read().split(";")
                if statement.strip()
            ]
        for statement in statements:
            cursor.execute(statement)
        conn.commit()
        return True
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def insert_analysis(
    message: str,
    intent: str,
    intent_confidence: float,
    sentiment: str,
    sentiment_confidence: float,
    urgency: str,
    routing_status: str,
) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO customer_analyses
            (message, intent, intent_confidence, sentiment,
             sentiment_confidence, urgency, routing_status)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                message,
                intent,
                intent_confidence,
                sentiment,
                sentiment_confidence,
                urgency,
                routing_status,
            ),
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()


def _read_query(query: str, params: Optional[tuple] = None) -> pd.DataFrame:
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(query, params or ())
        rows = cursor.fetchall()
        return pd.DataFrame(rows)
    finally:
        cursor.close()
        conn.close()


def fetch_analytics() -> pd.DataFrame:
    return _read_query(
        """
        SELECT analysis_id, created_at, message, intent, intent_confidence,
               sentiment, sentiment_confidence, urgency, routing_status
        FROM customer_analyses
        ORDER BY created_at DESC
        """
    )


def fetch_recent_analyses(limit: int = 25) -> pd.DataFrame:
    limit = max(1, min(int(limit), 500))
    return _read_query(
        f"""
        SELECT analysis_id, created_at, message, intent, intent_confidence,
               sentiment, urgency, routing_status
        FROM customer_analyses
        ORDER BY created_at DESC
        LIMIT {limit}
        """
    )
