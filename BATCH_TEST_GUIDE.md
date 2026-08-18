# 10K acceptance test

1. Open the deployed app.
2. Select **Batch Analyzer**.
3. Upload `customer_messages_10000.csv`.
4. Select the `message` column.
5. Choose NLP batch size 128.
6. Click **Process & Store Batch**.
7. Confirm the progress reaches 10,000/10,000.
8. Confirm Batch History shows `Completed` and `processed_rows = 10000`.
9. Open Dashboard and confirm Total Analyses increased by 10,000.
10. Refresh Power BI and confirm the SQL view contains the new rows.

This is the runtime acceptance test for the high-volume release.
