# High-volume release checklist

- [x] CSV batch upload capped at 50,000 messages.
- [x] Empty messages removed before inference.
- [x] Intent inference is vectorized over each bounded chunk.
- [x] RoBERTa sentiment uses pipeline batching and truncation.
- [x] Results are written to MySQL in bulk transactions.
- [x] Batch metadata tracks total, processed, and status.
- [x] Batch ID is indexed for history and reporting.
- [x] Full-history dashboard KPIs use SQL aggregation.
- [x] Interactive dashboard table is bounded to 5,000 recent rows.
- [ ] Final production acceptance test: process an actual 10,000+ row CSV on the deployed app and verify Completed status and row count.

The final item cannot be truthfully checked from GitHub alone; it requires execution against the live deployment.
