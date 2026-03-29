import streamlit as st
import joblib

model = joblib.load("model.pkl")

# st.title("AI Customer Insight Classifier")

# text = st.text_area("Enter message")
st.set_page_config(page_title="AI Classifier", layout="centered")

st.title("🚀 AI Customer Insight Classifier")
st.markdown("Analyze customer messages with ML")

text = st.text_area("💬 Enter customer message:")

if st.button("Analyze"):
    if text:
        pred = model.predict([text])[0]

        if pred in ["bug", "billing"]:
            sentiment = "🔴 Negative"
            urgency = "⚡ High"
        else:
            sentiment = "🟢 Positive"
            urgency = "✅ Low"

        st.success(f"Category: {pred}")
        st.info(f"Sentiment: {sentiment}")
        st.warning(f"Urgency: {urgency}")
    else:
        st.write("Enter text")
# if st.button("Analyze"):
#     if text:
#         pred = model.predict([text])[0]

#         if pred in ["bug", "billing"]:
#             sentiment = "Negative"
#         else:
#             sentiment = "Positive"

#         st.write("Category:", pred)
#         st.write("Sentiment:", sentiment)
else:
    st.write("Enter text")