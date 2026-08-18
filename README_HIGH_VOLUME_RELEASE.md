# High-Volume Release

- Supports up to 50,000 non-empty customer messages per CSV upload.
- Intent classification uses the existing TF-IDF + Logistic Regression model.
- Sentiment uses batched 3-class RoBERTa inference.
- Urgency and routing are computed per message.
- Results are inserted into MySQL in bulk chunks.
- Batch metadata records source, total rows, processed rows, and status.
- Full-history dashboard KPIs are computed by SQL aggregation; the UI table is capped at 5,000 recent rows.
- Power BI views are available for customer-level and daily/batch summaries.

For a true production-scale 50k workload, runtime speed depends on the Streamlit CPU/GPU allocation and database latency. The recommended acceptance test is a real 10k upload before claiming the deployment is fully validated.
