from __future__ import annotations

from typing import Callable, Iterable

import pandas as pd


def normalize_messages(df: pd.DataFrame, message_column: str) -> pd.DataFrame:
    out = df.copy()
    out[message_column] = out[message_column].fillna("").astype(str).str.strip()
    out = out[out[message_column].str.len() > 0].copy()
    return out.reset_index(drop=True)


def batched(items: list[str], batch_size: int) -> Iterable[tuple[int, list[str]]]:
    for start in range(0, len(items), batch_size):
        yield start, items[start:start + batch_size]


def classify_batch(
    texts: list[str],
    intent_model,
    sentiment_model,
    urgency_fn: Callable[[str, str, str], str],
    batch_size: int = 32,
    progress_fn: Callable[[int], None] | None = None,
) -> list[dict]:
    """Run lightweight intent inference + batched RoBERTa inference efficiently."""
    if not texts:
        return []

    rows: list[dict] = []
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

    total = len(texts)
    for i, text in enumerate(texts):
        probs = probabilities[i]
        best_idx = int(probs.argmax())
        intent = str(classes[best_idx])
        intent_confidence = float(probs[best_idx])
        sentiment_raw = sentiment_results[i]
        sentiment = label_map.get(str(sentiment_raw["label"]).lower(), str(sentiment_raw["label"]).title())
        sentiment_confidence = float(sentiment_raw["score"])
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
        if progress_fn and ((i + 1) % max(1, batch_size) == 0 or i + 1 == total):
            progress_fn(i + 1)
    return rows
