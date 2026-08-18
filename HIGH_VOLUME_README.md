# High-Volume Customer Analyzer

The application supports batch analysis of up to 50,000 non-empty customer messages per CSV upload.

## Processing pipeline

CSV upload → validation/normalization → batched TF-IDF + Logistic Regression intent inference → batched RoBERTa sentiment inference → rule-based urgency → routing decision → bulk MySQL ingestion → SQL-aggregated dashboard / Power BI views.

## Capacity notes

- Maximum: 50,000 messages per upload.
- NLP inference is performed in configurable batches to avoid loading all transformer outputs at once.
- Database writes are grouped into larger chunks to reduce connection/transaction overhead.
- MySQL keeps indexes on time, intent, sentiment, urgency, and batch ID for analytics queries.
- Dashboard KPIs and charts use SQL aggregation over the full history; only the latest 5,000 records are loaded into the interactive table.
- For production workloads above 50,000 messages, submit multiple batches or move the worker to a dedicated background-job service/GPU worker.

## Acceptance test

A deployment is considered complete only after a real CSV batch is processed successfully and the Batch History shows `Completed` with `processed_rows == total_rows`, followed by dashboard KPI verification.
