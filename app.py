import base64
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from gtts import gTTS
from transformers import pipeline

from database import fetch_recent_analyses, init_database, insert_analysis, is_database_configured

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "model.pkl"
METRICS_PATH = BASE / "metrics.json"

st.set_page_config(page_title="AI Customer Insight Analyzer", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.metric-card {padding: 18px; border-radius: 14px; border: 1px solid rgba(128,128,128,.25);}
.small-muted {opacity: .7; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading intent model...")
def load_intent_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource(show_spinner="Loading sentiment model...")
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

intent_model = load_intent_model()
sentiment_model = load_sentiment_model()


def get_sentiment(text: str):
    result = sentiment_model(text, truncation=True)[0]
    label_map = {"label_0": "Negative", "label_1": "Neutral", "label_2": "Positive", "negative": "Negative", "neutral": "Neutral", "positive": "Positive"}
    label = label_map.get(str(result["label"]).lower(), str(result["label"]).title())
    return label, float(result["score"])


def get_urgency(text: str, intent: str, sentiment: str):
    text_lower = text.lower()
    high_terms = ["urgent", "urgently", "immediately", "right now", "stolen", "fraud", "fraudulent", "unauthorized", "scammed", "can't access", "cannot access", "locked out"]
    medium_terms = ["asap", "soon", "as soon as possible", "quickly", "today", "time sensitive", "time-sensitive", "failed", "not working", "blocked", "missing", "wrong", "charged twice", "declined", "pending", "issue", "problem"]
    if any(term in text_lower for term in high_terms): return "High"
    if any(term in text_lower for term in medium_terms): return "Medium"
    if sentiment == "Negative" and intent in {"billing", "transfer", "card"}: return "Medium"
    return "Low"


def speak(text: str):
    audio_path = BASE / "voice.mp3"
    tts = gTTS(text=text, lang="en")
    tts.save(audio_path)
    audio_bytes = audio_path.read_bytes()
    b64 = base64.b64encode(audio_bytes).decode()
    st.markdown(f'<audio controls autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>', unsafe_allow_html=True)


def dashboard():
    st.title("📊 Customer Intelligence Dashboard")
    st.caption("Live analytics from MySQL • Power BI-ready data model")
    if not is_database_configured():
        st.warning("MySQL is not connected. Add MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD and MYSQL_DATABASE to enable live analytics.")
        st.info("The AI analyzer still works without MySQL. Once connected, every analysis is stored automatically.")
        return
    try:
        init_database()
        df = fetch_recent_analyses(500)
    except Exception as exc:
        st.error(f"MySQL connection failed: {exc}")
        return
    if df.empty:
        st.info("No customer analyses have been stored yet. Run an analysis from the Analyzer tab.")
        return
    total = len(df)
    high = int((df["urgency"] == "High").sum())
    negative = int((df["sentiment"] == "Negative").sum())
    auto = int((df["routing_status"] == "Auto-Routable").sum())
    avg_conf = float(df["intent_confidence"].mean() * 100)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Analyses", f"{total:,}")
    c2.metric("High Urgency", f"{high:,}")
    c3.metric("Negative", f"{negative:,}")
    c4.metric("Avg Confidence", f"{avg_conf:.1f}%")
    c5.metric("Auto-Routable", f"{auto / total * 100:.1f}%")
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        intent_counts = df["intent"].value_counts().reset_index()
        intent_counts.columns = ["intent", "count"]
        st.plotly_chart(px.bar(intent_counts, x="intent", y="count", title="Customer Intent Volume"), use_container_width=True)
    with col2:
        sentiment_counts = df["sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["sentiment", "count"]
        st.plotly_chart(px.pie(sentiment_counts, names="sentiment", values="count", title="Sentiment Distribution"), use_container_width=True)
    col3, col4 = st.columns(2)
    with col3:
        urgency_counts = df["urgency"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).reset_index()
        urgency_counts.columns = ["urgency", "count"]
        st.plotly_chart(px.bar(urgency_counts, x="urgency", y="count", title="Urgency Breakdown"), use_container_width=True)
    with col4:
        routing_counts = df["routing_status"].value_counts().reset_index()
        routing_counts.columns = ["routing_status", "count"]
        st.plotly_chart(px.bar(routing_counts, x="routing_status", y="count", title="Routing Status"), use_container_width=True)
    st.subheader("Recent Customer Analyses")
    display = df.copy()
    display["intent_confidence"] = (display["intent_confidence"] * 100).round(1).astype(str) + "%"
    st.dataframe(display, use_container_width=True, hide_index=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Export Analytics CSV", csv, "customer_analytics.csv", "text/csv")


def analyzer():
    st.title("🤖 AI Customer Insight Analyzer")
    st.caption("Hybrid NLP system for customer intent, sentiment, and urgency analysis")
    with st.sidebar:
        st.subheader("Model")
        st.write("**Intent:** TF-IDF + Logistic Regression")
        st.write("**Sentiment:** RoBERTa (3-class pretrained)")
        st.write("**Urgency:** Rule-based business logic")
        if METRICS_PATH.exists():
            metrics = json.loads(METRICS_PATH.read_text())
            st.divider(); st.subheader("Offline evaluation")
            st.metric("Test accuracy", f"{metrics['accuracy'] * 100:.2f}%")
            st.metric("Macro F1", f"{metrics['macro_f1'] * 100:.2f}%")
    voice_enabled = st.toggle("🔊 Enable AI Voice", value=False)
    if "messages" not in st.session_state: st.session_state.messages = []
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []; st.rerun()
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])
    user_input = st.chat_input("💬 Type a customer message...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing intent, sentiment and urgency..."):
                probabilities = intent_model.predict_proba([user_input])[0]
                classes = intent_model.classes_
                best_idx = probabilities.argmax()
                intent = str(classes[best_idx])
                intent_confidence = float(probabilities[best_idx])
                sentiment, sentiment_confidence = get_sentiment(user_input)
                urgency = get_urgency(user_input, intent, sentiment)
            routing_status = "Needs Review" if intent_confidence < 0.55 else "Auto-Routable"
            st.subheader(f"Intent: {intent.replace('_', ' ').title()}")
            st.write(f"**Sentiment:** {sentiment} ({sentiment_confidence * 100:.1f}%)")
            st.write(f"**Urgency:** {urgency}")
            st.write(f"**Routing:** {routing_status}")
            st.write("**Intent confidence**"); st.progress(intent_confidence); st.caption(f"Model confidence: {intent_confidence * 100:.1f}%")
            st.write("**Top intent predictions**")
            for idx in probabilities.argsort()[::-1][:3]: st.write(f"• {str(classes[idx]).replace('_', ' ').title()}: {probabilities[idx] * 100:.1f}%")
            route_map = {"billing": "💳 Suggested route: Payments / Billing", "card": "💳 Suggested route: Card Support", "transfer": "🔄 Suggested route: Transfers"}
            st.info(route_map.get(intent, "👤 Suggested route: Account Support"))
            if voice_enabled:
                try: speak(f"Intent: {intent}. Sentiment: {sentiment}. Urgency: {urgency}.")
                except Exception: st.warning("Voice generation is unavailable, but the analysis completed successfully.")
            if is_database_configured():
                try:
                    init_database()
                    insert_analysis(user_input, intent, intent_confidence, sentiment, sentiment_confidence, urgency, routing_status)
                    st.success("✓ Analysis saved to MySQL")
                except Exception as exc:
                    st.warning(f"Analysis completed, but MySQL save failed: {exc}")
            result_text = f"**Intent:** {intent.replace('_', ' ').title()}  \n**Sentiment:** {sentiment}  \n**Urgency:** {urgency}  \n**Routing:** {routing_status}"
            st.session_state.messages.append({"role": "assistant", "content": result_text})


if "page" not in st.session_state: st.session_state.page = "Analyzer"
with st.sidebar:
    st.markdown("## AI Customer Insight")
    st.session_state.page = st.radio("Navigation", ["Analyzer", "Dashboard"], index=0 if st.session_state.page == "Analyzer" else 1)
    st.divider()
    db_status = "🟢 MySQL Connected" if is_database_configured() else "🟡 MySQL Not Configured"
    st.caption(db_status)

if st.session_state.page == "Dashboard": dashboard()
else: analyzer()
