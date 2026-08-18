# Power BI integration

Connect Power BI Desktop to the same MySQL database used by the Streamlit app.

## Recommended model

Use these MySQL views:

- `powerbi_customer_analytics` — one row per analyzed customer message.
- `powerbi_daily_summary` — daily KPI aggregates.

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

### Detail table
Show `created_at`, `message`, `intent`, confidence, sentiment, urgency and routing status.

Power BI Desktop: **Get Data → MySQL database**, then select the two views above.
