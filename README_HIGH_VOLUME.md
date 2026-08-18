## High-volume processing

The Batch Analyzer is designed for up to 50,000 non-empty messages per upload. It uses bounded NLP batches, batched RoBERTa inference, bulk MySQL writes, batch progress tracking, and SQL aggregation for dashboard KPIs.

For production loads beyond a single 50k upload, run multiple batches or move inference to a dedicated worker/GPU service. Actual throughput depends on CPU/GPU, Streamlit resources, transformer model load time, and MySQL network latency.
