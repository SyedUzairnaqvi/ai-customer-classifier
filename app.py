import base64
import json
from pathlib import Path

import joblib
import streamlit as st
from gtts import gTTS
from transformers import pipeline

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "model.pkl"
METRICS_PATH = BASE / "metrics.json"

st.set_page_config(page_title="AI Customer Insight Analyzer", page_icon="🤖", layout="centered")

# -----------------------------
# Model loading (cached once)
# -----------------------------
@st.cache_resource(show_spinner="Loading intent model...")
def load_intent_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner="Loading sentiment model...")
def load_sentiment_model():
    # 3-class pretrained sentiment model: Negative / Neutral / Positive.
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    )


intent_model = load_intent_model()
sentiment_model = load_sentiment_model()

# -----------------------------
# Helpers
# -----------------------------
def get_sentiment(text: str):
    result = sentiment_model(text, truncation=True)[0]
    label_map = {
        "label_0": "Negative",
        "label_1": "Neutral",
        "label_2": "Positive",
        "negative": "Negative",
        "neutral": "Neutral",
        "positive": "Positive",
    }
    label = label_map.get(str(result["label"]).lower(), str(result["label"]).title())
    return label, float(result["score"])


def get_urgency(text: str, intent: str, sentiment: str):
    text_lower = text.lower()
    high_terms = [
        "urgent", "urgently", "immediately", "right now",
        "stolen", "fraud", "fraudulent", "unauthorized", "scammed",
        "can't access", "cannot access", "locked out",
    ]
    medium_terms = [
        "asap", "soon", "as soon as possible", "quickly", "today",
        "time sensitive", "time-sensitive",
        "failed", "not working", "blocked", "missing", "wrong",
        "charged twice", "declined", "pending", "issue", "problem",
    ]

    if any(term in text_lower for term in high_terms):
        return "High"
    if any(term in text_lower for term in medium_terms):
        return "Medium"
    if sentiment == "Negative" and intent in {"billing", "transfer", "card"}:
        return "Medium"
    return "Low"


def speak(text: str):
    # gTTS requires network access; keep it optional so core analysis works
    # even when voice generation is unavailable.
    audio_path = BASE / "voice.mp3"
    tts = gTTS(text=text, lang="en")
    tts.save(audio_path)
    audio_bytes = audio_path.read_bytes()
    b64 = base64.b64encode(audio_bytes).decode()
    st.markdown(
        f'<audio controls autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>',
        unsafe_allow_html=True,
    )


# -----------------------------
# UI
# -----------------------------
st.markdown("# 🤖 AI Customer Insight Analyzer")
st.caption("Hybrid NLP system for customer intent, sentiment, and urgency analysis")

with st.sidebar:
    st.subheader("Model")
    st.write("**Intent:** TF-IDF + Logistic Regression")
    st.write("**Sentiment:** RoBERTa (3-class pretrained)")
    st.write("**Urgency:** Rule-based business logic")

    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
        st.divider()
        st.subheader("Offline evaluation")
        st.metric("Test accuracy", f"{metrics['accuracy'] * 100:.2f}%")
        st.metric("Macro F1", f"{metrics['macro_f1'] * 100:.2f}%")
        st.caption(
            f"{metrics['train_rows']:,} train / {metrics['test_rows']:,} test records "
            "from BANKING77"
        )

voice_enabled = st.toggle("🔊 Enable AI Voice", value=False)

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("🗑️ Clear Conversation", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("💬 Type a customer message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing intent, sentiment and urgency..."):
            probabilities = intent_model.predict_proba([user_input])[0]
            classes = intent_model.classes_
            best_idx = probabilities.argmax()
            intent = str(classes[best_idx])
            intent_confidence = float(probabilities[best_idx])

            sentiment, sentiment_confidence = get_sentiment(user_input)
            urgency = get_urgency(user_input, intent, sentiment)

        # Don't pretend that low model probability is certainty.
        if intent_confidence < 0.55:
            routing_status = "Needs Review"
        else:
            routing_status = "Auto-Routable"

        st.subheader(f"Intent: {intent.replace('_', ' ').title()}")
        st.write(f"**Sentiment:** {sentiment} ({sentiment_confidence * 100:.1f}%)")
        st.write(f"**Urgency:** {urgency}")
        st.write(f"**Routing:** {routing_status}")

        st.write("**Intent confidence**")
        st.progress(intent_confidence)
        st.caption(f"Model confidence: {intent_confidence * 100:.1f}%")

        # Show the model's top alternatives for transparency.
        top_indices = probabilities.argsort()[::-1][:3]
        st.write("**Top intent predictions**")
        for idx in top_indices:
            st.write(
                f"• {str(classes[idx]).replace('_', ' ').title()}: "
                f"{probabilities[idx] * 100:.1f}%"
            )

        if intent == "billing":
            st.info("💳 Suggested route: Payments / Billing")
        elif intent == "card":
            st.info("💳 Suggested route: Card Support")
        elif intent == "transfer":
            st.info("🔄 Suggested route: Transfers")
        else:
            st.info("👤 Suggested route: Account Support")

        if voice_enabled:
            try:
                speak(
                    f"Intent: {intent}. Sentiment: {sentiment}. "
                    f"Urgency: {urgency}."
                )
            except Exception:
                st.warning("Voice generation is unavailable, but the analysis completed successfully.")

        result_text = (
            f"**Intent:** {intent.replace('_', ' ').title()}  \n"
            f"**Sentiment:** {sentiment}  \n"
            f"**Urgency:** {urgency}  \n"
            f"**Routing:** {routing_status}"
        )
        st.session_state.messages.append({"role": "assistant", "content": result_text})
