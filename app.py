import streamlit as st
from transformers import pipeline
from gtts import gTTS
import base64
import time

# Load model
classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# 🔊 Speak (auto-play)
def speak(text):
    tts = gTTS(text)
    tts.save("voice.mp3")

    with open("voice.mp3", "rb") as f:
        audio_bytes = f.read()
        b64 = base64.b64encode(audio_bytes).decode()

        st.markdown(f"""
        <audio autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """, unsafe_allow_html=True)

# Page config
st.set_page_config(page_title="AI Chatbot", layout="centered")

# 🎨 UI Styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
.big-title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
}
.subtitle {
    text-align: center;
    color: #cbd5f5;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🚀 AI Customer Insight</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart Customer Message Analyzer</div>', unsafe_allow_html=True)

# 🔊 Voice toggle
voice_enabled = st.toggle("🔊 Enable AI Voice")

# 🧠 Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("💬 Type your message...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    text = user_input
    text_lower = text.lower()

    # 🔧 CATEGORY RULES
    billing_words = [
        "payment", "refund", "charged", "billing",
        "subscription", "invoice", "transaction", "deducted"
    ]

    bug_words = [
        "error", "crash", "bug", "not working",
        "failed", "freeze", "stuck", "issue", "login"
    ]

    if any(word in text_lower for word in billing_words):
        pred = "billing"
        confidence = 100

    elif any(word in text_lower for word in bug_words):
        pred = "bug"
        confidence = 100

    else:
        result = classifier(text)[0]
        label = result["label"]
        confidence = round(result["score"] * 100, 2)
        pred = "positive" if label == "POSITIVE" else "negative"

    # 🎯 SENTIMENT + URGENCY
    sentiment = "🟢 Positive"
    urgency = "🟢 Low"

    if pred in ["bug", "billing"]:
        sentiment = "🟡 Needs Attention"
        urgency = "⚡ Medium"

    if any(word in text_lower for word in ["urgent", "asap", "immediately", "now"]):
        urgency = "🚀 High"
        sentiment = "🟠 Important"

    if any(word in text_lower for word in ["worst", "terrible", "hate", "awful"]):
        sentiment = "🟠 Strong Negative"
        urgency = "🚀 High"

    # 🎨 COLORS (no red)
    if pred == "billing":
        icon, color = "💳", "#f59e0b"
    elif pred == "bug":
        icon, color = "🐞", "#8b5cf6"
    elif pred == "positive":
        icon, color = "😊", "#22c55e"
    else:
        icon, color = "😐", "#3b82f6"

    # 🤖 Assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            time.sleep(1)

        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {color}, #1e293b);
            padding: 20px;
            border-radius: 18px;
            text-align: center;
            box-shadow: 0 0 20px {color};
        ">
            <h2>{icon} {pred.upper()}</h2>
            <p>{sentiment}</p>
            <p>{urgency}</p>
        </div>
        """, unsafe_allow_html=True)

        # 🔮 Confidence
        st.progress(confidence / 100)
        st.write(f"🔮 Confidence: {confidence}%")

        # 🔊 Voice
        if voice_enabled:
            speak(f"Category is {pred}. Sentiment is {sentiment}. Urgency is {urgency}.")

        # 🎭 Reactions
        if pred == "bug":
            st.info("🛠️ Something needs fixing!")
        elif pred == "billing":
            st.warning("💸 Check your payment details!")
        elif pred == "positive":
            st.success("🎉 Everything looks great!")
        else:
            st.info("🧐 Seems normal!")

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"{pred} | {sentiment} | {urgency}"
    })