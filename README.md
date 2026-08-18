# AI Customer Insight Analyzer

A high-volume hybrid NLP customer-support analytics platform with **ML + MySQL/TiDB Cloud + Streamlit + Power BI**.

## Live Demo

**Streamlit App:** https://ai-customer-classifier-v6yuiq2aqqdhooajzs3eum.streamlit.app/

## Core AI

For every customer message the system produces:

- **Intent:** account, billing, card, or transfer
- **Sentiment:** negative, neutral, or positive
- **Urgency:** high, medium, or low
- **Routing:** Auto-Routable or Needs Review
- **Confidence:** intent probability and top alternative predictions

The model stack is TF-IDF + Logistic Regression for intent, RoBERTa for sentiment, and transparent business rules for urgency.

## High-volume processing

The **Batch Analyzer** accepts CSV uploads containing up to **50,000 non-empty customer messages per batch**.

```text
CSV (10k–50k messages)
          |
          v
Message validation / cleaning
          |
          v
TF-IDF + Logistic Regression (bulk inference)
          |
          v
RoBERTa sentiment (batched inference)
          |
          v
Urgency + routing rules
          |
          v
Bulk MySQL inserts (1,000-row chunks)
          |
          v
Streamlit Dashboard + Power BI
```

Features:

- Batch progress bar and throughput/ETA
- Configurable NLP batch size
- Bulk `executemany` database writes instead of one INSERT per message
- Batch IDs and processing history
- Indexed MySQL storage
- Dashboard reads a responsive latest-5,000 window while MySQL retains the full history
- Power BI views remain backed by the full database table

**Important:** 50,000-row capacity is a software batch limit, not a guarantee of instant processing. RoBERTa inference time depends on the Streamlit/CPU resources available. Increasing the NLP batch size can improve throughput when sufficient memory is available.

## Analytics architecture

```text
Customer Message / CSV Batch
            |
            v
 AI Analyzer + Batch Analyzer
            |
            +----------------------+
            |                      |
            v                      v
      Streamlit UI           MySQL / TiDB Cloud
      Analyzer + Dashboard   customer_analyses
                             analysis_batches
                                  |
                                  v
                         Power BI-ready views
                                  |
                                  v
                           Power BI Dashboard
```

## MySQL

Configure:

- `MYSQL_HOST`
- `MYSQL_PORT`
- `MYSQL_USER`
- `MYSQL_PASSWORD`
- `MYSQL_DATABASE`

The app initializes the base schema and `batch_schema.sql` automatically when the database user has DDL permissions.

Batch storage adds:

- `customer_analyses.batch_id`
- `analysis_batches`
- `powerbi_batch_summary`

For deployment, use environment variables or Streamlit Secrets. **Never commit real database credentials.**

## Streamlit modes

1. **Analyzer** — single-message AI analysis and MySQL persistence.
2. **Batch Analyzer** — high-volume CSV processing up to 50,000 messages/batch.
3. **Dashboard** — live KPIs, intent volume, sentiment, urgency, routing, recent records, and CSV export.

## Power BI

Connect Power BI Desktop to the same MySQL/TiDB database. The main views are:

- `powerbi_customer_analytics`
- `powerbi_daily_summary`
- `powerbi_batch_summary`

The customer analytics view now includes the actual `message` and `batch_id` fields for detailed reporting.

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
app.py                 # Streamlit analyzer + batch analyzer + dashboard
batch_processor.py     # Batched NLP inference
model.pkl              # trained intent model
metrics.json           # evaluation metrics
database.py            # MySQL connection + bulk persistence + batch history
schema.sql             # base MySQL tables + Power BI views
batch_schema.sql       # scalable batch migration + batch Power BI view
powerbi/README.md      # Power BI connection/model guide
.env.example           # safe configuration template
requirements.txt       # runtime dependencies
train.py               # model training
download_data.py       # BANKING77 download helper
```
