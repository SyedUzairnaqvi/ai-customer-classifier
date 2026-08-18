CREATE TABLE IF NOT EXISTS customer_analyses (
    analysis_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    message TEXT NOT NULL,
    intent VARCHAR(100) NOT NULL,
    intent_confidence DECIMAL(6,5) NOT NULL,
    sentiment VARCHAR(30) NOT NULL,
    sentiment_confidence DECIMAL(6,5) NOT NULL,
    urgency VARCHAR(20) NOT NULL,
    routing_status VARCHAR(30) NOT NULL,
    PRIMARY KEY (analysis_id),
    INDEX idx_created_at (created_at),
    INDEX idx_intent (intent),
    INDEX idx_sentiment (sentiment),
    INDEX idx_urgency (urgency)
);

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
    CHAR_LENGTH(message) AS message_length
FROM customer_analyses;

CREATE OR REPLACE VIEW powerbi_daily_summary AS
SELECT
    DATE(created_at) AS analysis_date,
    COUNT(*) AS total_analyses,
    SUM(urgency = 'High') AS high_urgency,
    SUM(urgency = 'Medium') AS medium_urgency,
    SUM(urgency = 'Low') AS low_urgency,
    SUM(sentiment = 'Positive') AS positive_count,
    SUM(sentiment = 'Neutral') AS neutral_count,
    SUM(sentiment = 'Negative') AS negative_count,
    ROUND(AVG(intent_confidence) * 100, 2) AS avg_intent_confidence_pct
FROM customer_analyses
GROUP BY DATE(created_at);
