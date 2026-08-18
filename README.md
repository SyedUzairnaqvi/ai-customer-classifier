# AI Customer Insight Analyzer

A hybrid NLP customer-support analytics application with **ML + MySQL + Streamlit dashboard + Power BI-ready analytics**.

## Live Demo

**Streamlit App:** https://ai-customer-classifier-v6yuiq2aqqdhooajzs3eum.streamlit.app/

**Current release:** FINAL — this repository uses `ai_customer_insight_analyzer_FINAL` as the main/root version.

## Core AI

For every customer message the system produces:

- **Intent:** account, billing, card, or transfer
- **Sentiment:** negative, neutral, or positive
- **Urgency:** high, medium, or low
- **Routing:** Auto-Routable or Needs Review
- **Confidence:** intent probability and top alternative predictions

The model stack is TF-IDF + Logistic Regression for intent, RoBERTa for sentiment, and transparent business rules for urgency.

## Analytics architecture

```text
Customer Message
       |
       v
AI Analyzer
(Intent + Sentiment + Urgency + Routing)
       |
       +----------------------+
       |                      |
       v                      v
Streamlit UI             MySQL Database
Analyzer + Dashboard     customer_analyses
                              |
                              v
                     Power BI-ready SQL Views
                              |
                              v
                       Power BI Dashboard
```

## MySQL

The app now persists successful analyses to MySQL when these environment variables/secrets are configured:

- `MYSQL_HOST`
- `MYSQL_PORT`
- `MYSQL_USER`
- `MYSQL_PASSWORD`
- `MYSQL_DATABASE`

Run `schema.sql` once against the database. It creates `customer_analyses` plus the Power BI views:

- `powerbi_customer_analytics`
- `powerbi_daily_summary`

The app also calls the schema initialization safely on startup/use, so the tables/views are created automatically when the database user has permission.

For deployment, use environment variables or Streamlit Secrets. **Never commit real database credentials.** See `.env.example`.

## Streamlit dashboard

The sidebar now has two modes:

1. **Analyzer** — run the AI classifier and save results to MySQL.
2. **Dashboard** — live KPI cards, intent volume, sentiment, urgency, routing charts, recent records, and CSV export.

Dashboard KPIs include total analyses, high-urgency cases, negative sentiment, average intent confidence, and auto-routable rate.

## Power BI

See `powerbi/README.md` for the recommended Power BI model. Connect Power BI Desktop to the same MySQL database and import the two views listed above.

Suggested report pages/visuals:

- Total analyses
- High urgency cases
- Negative sentiment
- Average confidence
- Auto-routable rate
- Intent distribution
- Sentiment distribution
- Urgency distribution
- Daily volume
- Routing status
- Customer-level analysis table

## Model evaluation

The included intent model was trained on 10,003 BANKING77 training examples and evaluated on 3,080 untouched test examples.

- **Accuracy: 92.89%**
- **Macro F1: 92.89%**

These are intent-routing metrics, not production performance guarantees.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the pretrained RoBERTa sentiment model from Hugging Face.

## Project structure

```text
app.py                 # Streamlit analyzer + dashboard
model.pkl              # trained intent model
metrics.json           # evaluation metrics
database.py            # MySQL connection + persistence + analytics queries
schema.sql             # MySQL tables + Power BI views
powerbi/README.md      # Power BI connection/model guide
.env.example           # safe configuration template
requirements.txt       # runtime dependencies
train.py               # model training
download_data.py       # BANKING77 download helper
```
