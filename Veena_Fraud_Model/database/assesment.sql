CREATE TABLE risk_assessments (
    id SERIAL PRIMARY KEY,
    claim_id INT REFERENCES claims(id),
    fraud_probability FLOAT,
    risk_level VARCHAR(20),
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);