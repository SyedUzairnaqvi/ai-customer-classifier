from __future__ import annotations

from typing import Callable

import pandas as pd


def normalize_messages(df: pd.DataFrame, message_column: str) -> pd.DataFrame:
    out = df.copy()
    out[message_column] = out[message_column].fillna("").astype(str).str.strip()
    return out.loc[out[message_column].str.len() > 0].reset_index(drop=True)


def classify_batch_fast(
    texts: list[str],
    intent_model,
    sentiment_model,
    urgency_fn: Callable[[str, str, str], str],
    batch_size: int = 128,
) -> list[dict]:
    """Classify one bounded chunk using vectorized intent and batched transformer inference."""
    if not texts:
        return []

    probabilities = intent_model.predict_proba(texts)
    classes = intent_model.classes_
    sentiment_results = sentiment_model(
        texts,
        batch_size=batch_size,
        truncation=True,
        max_length=512,
    )

    label_map = {
        "label_0": "Negative", "label_1": "Neutral", "label_2": "Positive",
        "negative": "Negative", "neutral": "Neutral", "positive": "Positive",
    }
    rows = []
    for i, text in enumerate(texts):
        probs = probabilities[i]
        best_idx = int(probs.argmax())
        intent = str(classes[best_idx])
        intent_confidence = float(probs[best_idx])
        raw = sentiment_results[i]
        sentiment = label_map.get(str(raw["label"]).lower(), str(raw["label"]).title())
        sentiment_confidence = float(raw["score"])
        urgency = urgency_fn(text, intent, sentiment)
        routing_status = "Needs Review" if intent_confidence < 0.55 else "Auto-Routable"
        rows.append({
            "message": text,
            "intent": intent,
            "intent_confidence": intent_confidence,
            "sentiment": sentiment,
            "sentiment_confidence": sentiment_confidence,
            "urgency": urgency,
            "routing_status": routing_status,
        })
    return rows
