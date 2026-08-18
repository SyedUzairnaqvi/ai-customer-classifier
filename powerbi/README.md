# Power BI integration

Connect Power BI Desktop to the same MySQL/TiDB Cloud database used by the Streamlit app.

## Recommended model

Use these MySQL views:

- `powerbi_customer_analytics` — one row per analyzed customer message, including the message text and batch ID.
- `powerbi_daily_summary` — full-history daily KPI aggregates.
- `powerbi_batch_summary` — batch-level processing/quality summary.

## Suggested dashboard

### KPI cards
- Total analyses
- High urgency cases
- Negative sentiment cases
- Average intent confidence
- Auto-routable rate

### Charts
- Analyses by intent
- Sentiment distribution
- Urgency distribution
- Daily analysis volume
- Intent confidence distribution
- Routing status split
- Batch volume

### Detail table
Show `created_at`, `message`, `intent`, confidence, sentiment, urgency, routing status and `batch_id`.

Power BI Desktop: **Get Data → MySQL database**, then select the views above.

The Streamlit dashboard uses SQL aggregation for full-history KPIs, so 10k–50k+ stored records do not need to be loaded into pandas just to calculate dashboard totals.
