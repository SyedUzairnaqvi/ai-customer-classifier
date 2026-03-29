import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_csv("data.csv")

# Clean
df.dropna(inplace=True)

# Model pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=200))
])

# Train
model.fit(df["text"], df["label"])

# Save
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved!")