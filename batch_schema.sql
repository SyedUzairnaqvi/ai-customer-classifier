ALTER TABLE customer_analyses ADD COLUMN IF NOT EXISTS batch_id VARCHAR(64) NULL;

CREATE TABLE IF NOT EXISTS analysis_batches (
    batch_id VARCHAR(64) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    source_name VARCHAR(255) NOT NULL,
    total_rows INT UNSIGNED NOT NULL,
    processed_rows INT UNSIGNED NOT NULL DEFAULT 0,
    status VARCHAR(30) NOT NULL DEFAULT 'Processing',
    PRIMARY KEY (batch_id),
    INDEX idx_batch_created_at (created_at),
    INDEX idx_batch_status (status)
);

CREATE INDEX IF NOT EXISTS idx_customer_analyses_batch_id ON customer_analyses (batch_id);

CREATE OR REPLACE VIEW powerbi_customer_analytics AS
SELECT
    analysis_id,
    created_at,
    DATE(created_at) AS analysis_date,
    intent,
    ROUND(intent_confidence * 100, 2) AS intent_confidence_pct,
    sentiment,
    ROUND(sentiment_confidence * 100, 2) AS sentiment_confidence_pct,
    urgency,
    routing_status,
    batch_id,
    message,
    CHAR_LENGTH(message) AS message_length
FROM customer_analyses;

CREATE OR REPLACE VIEW powerbi_batch_summary AS
SELECT
    batch_id,
    MIN(created_at) AS batch_created_at,
    COUNT(*) AS processed_analyses,
    SUM(urgency = 'High') AS high_urgency,
    SUM(sentiment = 'Negative') AS negative_count,
    ROUND(AVG(intent_confidence) * 100, 2) AS avg_intent_confidence_pct
FROM customer_analyses
WHERE batch_id IS NOT NULL
GROUP BY batch_id;
