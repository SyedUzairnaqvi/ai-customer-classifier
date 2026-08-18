import base64
import json
import time
import uuid
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from gtts import gTTS
from transformers import pipeline

from batch_processor import classify_batch, normalize_messages
from database import (
    bulk_insert_analyses,
    create_batch,
    fetch_batch_history,
    fetch_dashboard_data,
    fetch_recent_analyses,
    init_database,
    insert_analysis,
    is_database_configured,
    update_batch,
)

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "model.pkl"
METRICS_PATH = BASE / "metrics.json"
MAX_BATCH_ROWS = 50_000

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
    result = sentiment_model(text, truncation=True, max_length=512)[0]
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
    st.caption("Live analytics from MySQL • Power BI-ready data model • Built for high-volume customer analysis")
    if not is_database_configured():
        st.warning("MySQL is not connected. Add MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD and MYSQL_DATABASE to enable live analytics.")
        return
    try:
        init_database()
        summary = fetch_dashboard_data()
        df = fetch_recent_analyses(5000)
    except Exception as exc:
        st.error(f"MySQL connection failed: {exc}")
        return
    total = int(summary["total"])
    if total == 0:
        st.info("No customer analyses have been stored yet. Run an analysis or upload a batch from the Batch Analyzer tab.")
        return

    high = int(summary["high"])
    negative = int(summary["negative"])
    auto = int(summary["auto_routable"])
    avg_conf = float(summary["avg_confidence"] * 100)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Analyses", f"{total:,}")
    c2.metric("High Urgency", f"{high:,}")
    c3.metric("Negative", f"{negative:,}")
    c4.metric("Avg Confidence", f"{avg_conf:.1f}%")
    c5.metric("Auto-Routable", f"{auto / total * 100:.1f}%")
    st.caption("KPIs and charts use the full MySQL history. The table below shows the latest 5,000 records for responsive UI.")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.bar(summary["intent"], x="intent", y="count", title="Customer Intent Volume"), use_container_width=True)
    with col2:
        st.plotly_chart(px.pie(summary["sentiment"], names="sentiment", values="count", title="Sentiment Distribution"), use_container_width=True)
    col3, col4 = st.columns(2)
    with col3:
        urgency_counts = summary["urgency"].set_index("urgency").reindex(["High", "Medium", "Low"]).fillna(0).reset_index()
        st.plotly_chart(px.bar(urgency_counts, x="urgency", y="count", title="Urgency Breakdown"), use_container_width=True)
    with col4:
        st.plotly_chart(px.bar(summary["routing"], x="routing_status", y="count", title="Routing Status"), use_container_width=True)

    st.subheader("Recent Customer Analyses")
    display = df.copy()
    for numeric_col in ["intent_confidence", "sentiment_confidence"]:
        if numeric_col in display.columns:
            display[numeric_col] = pd.to_numeric(display[numeric_col], errors="coerce")
            display[numeric_col] = display[numeric_col].mul(100).round(1).map(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
    st.dataframe(display, use_container_width=True, hide_index=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Export Recent Analytics CSV", csv, "customer_analytics_recent.csv", "text/csv")


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


def batch_analyzer():
    st.title("⚡ High-Volume Customer Analyzer")
    st.caption("Batch NLP + bulk MySQL ingestion for up to 50,000 customer messages per upload")
    if not is_database_configured():
        st.error("MySQL must be connected before running a batch.")
        return

    st.info("Upload a CSV with a customer-message column. Intent runs in bulk, RoBERTa sentiment runs in batches, urgency/routing are applied, and results are bulk-inserted into MySQL.")
    uploaded = st.file_uploader("Upload customer messages CSV", type=["csv"], help="Recommended format: customer_id,message. Maximum 50,000 non-empty messages per upload.")

    if uploaded is not None:
        try:
            df_input = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            return
        if df_input.empty:
            st.warning("The CSV is empty.")
            return
        preferred = [c for c in df_input.columns if c.lower() in {"message", "text", "customer_message", "feedback", "comment"}]
        message_column = st.selectbox("Message column", list(df_input.columns), index=list(df_input.columns).index(preferred[0]) if preferred else 0)
        clean = normalize_messages(df_input, message_column)
        if len(clean) > MAX_BATCH_ROWS:
            st.error(f"This upload has {len(clean):,} messages. Maximum is {MAX_BATCH_ROWS:,} per batch.")
            return
        st.write(f"**{len(clean):,} messages ready for processing**")
        st.dataframe(clean[[message_column]].head(10), use_container_width=True, hide_index=True)
        batch_size = st.select_slider("NLP batch size", options=[8, 16, 32, 64, 96, 128], value=32, help="Higher values can be faster when enough RAM is available. Reduce if memory becomes tight.")
        start = st.button("🚀 Process & Store Batch", type="primary", use_container_width=True)

        if start:
            batch_id = f"B{time.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
            texts = clean[message_column].tolist()
            create_batch(batch_id, uploaded.name, len(texts))
            progress = st.progress(0, text="Starting batch...")
            status = st.empty()
            started = time.perf_counter()
            total_inserted = 0
            try:
                for start_idx in range(0, len(texts), batch_size):
                    chunk = texts[start_idx:start_idx + batch_size]
                    rows = classify_batch(chunk, intent_model, sentiment_model, get_urgency, batch_size=batch_size)
                    inserted = bulk_insert_analyses(rows, batch_id, chunk_size=1000)
                    total_inserted += inserted
                    done = start_idx + len(chunk)
                    elapsed = max(time.perf_counter() - started, 0.001)
                    rate = done / elapsed
                    eta = (len(texts) - done) / rate if rate else 0
                    progress.progress(done / len(texts), text=f"Processed {done:,}/{len(texts):,}")
                    status.caption(f"Stored {total_inserted:,} rows • {rate:,.1f} messages/sec • ETA {eta/60:.1f} min")
                    update_batch(batch_id, done, "Processing" if done < len(texts) else "Completed")
                elapsed = max(time.perf_counter() - started, 0.001)
                update_batch(batch_id, total_inserted, "Completed")
                progress.progress(1.0, text="Batch completed")
                st.success(f"✓ {total_inserted:,} customer messages analyzed and stored in MySQL in {elapsed/60:.1f} minutes.")
                st.caption(f"Batch ID: {batch_id} • Average throughput: {total_inserted/elapsed:,.1f} messages/sec")
            except Exception as exc:
                update_batch(batch_id, total_inserted, "Failed")
                st.error(f"Batch stopped after {total_inserted:,} rows: {exc}")

    st.divider()
    st.subheader("Batch History")
    try:
        init_database()
        history = fetch_batch_history(20)
        if history.empty:
            st.caption("No batch jobs yet.")
        else:
            st.dataframe(history, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(f"Could not load batch history: {exc}")


if "page" not in st.session_state: st.session_state.page = "Analyzer"
with st.sidebar:
    st.markdown("## AI Customer Insight")
    pages = ["Analyzer", "Batch Analyzer", "Dashboard"]
    st.session_state.page = st.radio("Navigation", pages, index=pages.index(st.session_state.page) if st.session_state.page in pages else 0)
    st.divider()
    db_status = "🟢 MySQL Connected" if is_database_configured() else "🟡 MySQL Not Configured"
    st.caption(db_status)
    st.caption("High-volume capacity: 50,000 messages/batch")

if st.session_state.page == "Dashboard": dashboard()
elif st.session_state.page == "Batch Analyzer": batch_analyzer()
else: analyzer()
