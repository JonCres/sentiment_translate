# Data Dictionary

**Project:** Predictive CLTV Insights
**Industry:** Media & Entertainment
**Last Updated:** 2026-02-13

---

## Table of Contents

1. [Overview](#overview)
2. [Transaction Features](#transaction-features)
3. [RFM Features](#rfm-features)
4. [Behavioral Features](#behavioral-features)
5. [Survival Features](#survival-features)
6. [Prediction Outputs](#prediction-outputs)
7. [Derived Metrics](#derived-metrics)
8. [Data Types & Constraints](#data-types--constraints)
9. [Source Systems](#source-systems)

---

## Overview

This data dictionary documents all features, variables, and outputs in the Predictive CLTV Insights system. Features are organized into logical groups based on their purpose in the CLTV prediction pipeline.

**Feature Categories:**
- **Transaction Features**: Raw transaction data from source systems
- **RFM Features**: Recency, Frequency, Monetary summary statistics
- **Behavioral Features**: Engagement and content consumption metrics
- **Survival Features**: Subscription lifecycle data for contractual models
- **Prediction Outputs**: Model predictions and derived KPIs

---

## Transaction Features

Raw transaction data from source systems (e-commerce, billing, CRM).

| Feature | Type | Description | Valid Range | Example | Source |
|---------|------|-------------|-------------|---------|--------|
| **customer_id** | String | Unique customer identifier | Non-empty | "C_123456" | CRM |
| **transaction_id** | String | Unique transaction identifier | Non-empty | "TXN_789012" | Billing |
| **transaction_dt** | Datetime | Transaction timestamp (UTC) | 2010-01-01 to present | 2024-06-15T14:30:00Z | Billing |
| **amount_usd** | Float | Transaction monetary value (USD) | > 0 | 29.99 | Billing |
| **quantity** | Integer | Number of items purchased | ≥ 1 | 2 | Billing |
| **product_id** | String | Product/SKU identifier | Non-empty | "PROD_SVOD_MONTHLY" | Catalog |
| **payment_method** | Categorical | Payment method used | {credit_card, paypal, apple_pay, ...} | "credit_card" | Billing |
| **transaction_status** | Categorical | Transaction status | {completed, pending, failed, refunded} | "completed" | Billing |

**Business Rules:**
- Negative amounts indicate returns/refunds (filtered out in processing)
- Null customer_id indicates guest checkout (excluded from CLTV)
- Transactions older than 3 years archived for compliance

---

## RFM Features

Aggregated features summarizing customer purchase patterns (Recency, Frequency, Monetary).

| Feature | Type | Description | Calculation | Valid Range | Example |
|---------|------|-------------|-------------|-------------|---------|
| **customer_id** | String | Unique customer identifier | - | Non-empty | "C_123456" |
| **frequency** | Integer | Number of **repeat** purchases | COUNT(transactions) - 1 | ≥ 0 | 5 |
| **recency** | Float | Time between first and last purchase (days) | last_purchase_date - first_purchase_date | 0 to T | 180.5 |
| **T** | Float | Customer age (days since first purchase) | observation_period_end - first_purchase_date | > 0 | 365.0 |
| **monetary_value** | Float | Average transaction value for **repeat** customers (USD) | SUM(amount) / COUNT(repeat_transactions) | > 0 | 45.75 |
| **first_purchase_date** | Date | Date of first transaction | MIN(transaction_dt) | 2010-01-01 to present | 2023-01-15 |
| **last_purchase_date** | Date | Date of most recent transaction | MAX(transaction_dt) | ≥ first_purchase_date | 2024-07-10 |
| **total_revenue** | Float | Total lifetime spend (USD) | SUM(amount_usd) | > 0 | 1,234.56 |

**Business Definitions:**
- **Frequency**: Excludes first transaction (measures repeat behavior only)
  - Frequency = 0: One-time customer
  - Frequency = 3: Customer made 4 total purchases (1 initial + 3 repeats)
- **Recency**:
  - Recency = 0: Customer never made repeat purchase
  - Recency = T: Customer just made purchase at observation_period_end
  - Recency < T: Time since last purchase indicates dormancy
- **Monetary Value**: Only calculated for repeat customers (frequency > 0)
  - Excludes first transaction to model repeat purchase value

**Model Usage:**
- **BG/NBD**: Uses frequency, recency, T to predict future purchase frequency
- **Gamma-Gamma**: Uses frequency, monetary_value to predict transaction value
- **CLTV**: Combines BG/NBD purchase predictions × Gamma-Gamma value predictions

---

## Behavioral Features

Engagement and content consumption metrics (for XGBoost behavioral refinement).

| Feature | Type | Description | Calculation | Valid Range | Example | Source |
|---------|------|-------------|-------------|-------------|---------|--------|
| **customer_id** | String | Unique customer identifier | - | Non-empty | "C_123456" | - |
| **watch_time** | Integer | Total content watch time (minutes) | SUM(playback_duration) | ≥ 0 | 4,320 (3 days) | Streaming Analytics |
| **login_count** | Integer | Number of login sessions | COUNT(DISTINCT session_id) | ≥ 0 | 45 | Auth Service |
| **engagement_score** | Float | Composite engagement metric | Weighted sum of activities | 0.0 - 1.0 | 0.78 | Analytics Engine |
| **buffering_ratio** | Float | Buffering events per playback | buffering_events / total_plays | 0.0 - 1.0 | 0.05 | QoE Service |
| **active_days_count** | Integer | Number of unique active days | COUNT(DISTINCT DATE(activity_dt)) | ≥ 0 | 25 | Analytics Engine |
| **content_streams** | Integer | Number of content items consumed | COUNT(DISTINCT content_id) | ≥ 0 | 120 | Streaming Analytics |
| **avg_session_duration** | Float | Average session length (minutes) | AVG(session_end - session_start) | > 0 | 42.5 | Auth Service |
| **last_engagement_date** | Date | Most recent engagement activity | MAX(activity_dt) | - | 2024-12-01 | Analytics Engine |
| **genre_diversity** | Float | Shannon entropy of genre consumption | -Σ(p_i * log(p_i)) for genres | 0.0 - 3.0 | 1.85 | Streaming Analytics |

**Feature Definitions:**

**watch_time**: Total minutes of content consumed
- Calculated from streaming logs (playback_start to playback_stop)
- Excludes paused time and buffering time
- High watch time indicates engagement (typical: 1000-5000 min/month for active users)

**engagement_score**: Composite metric (0-1 scale)
- Formula: `0.4 * watch_time_normalized + 0.3 * login_frequency + 0.3 * content_diversity`
- Score > 0.7: Highly engaged
- Score 0.4-0.7: Moderately engaged
- Score < 0.4: At-risk

**buffering_ratio**: Quality of Experience (QoE) metric
- High ratio (> 0.1) indicates poor experience → churn risk
- Low ratio (< 0.05) indicates good experience → retention

**genre_diversity**: Measures breadth of content consumption
- 0: Consumes only one genre (low engagement)
- 1-2: Moderate diversity (typical)
- >2.5: High diversity (exploratory behavior)

**Model Usage:**
- **XGBoost Churn Prediction**: Uses all behavioral features to predict short-term churn risk
- **Binge-and-Burnout Detection**: High watch_time + low active_days_count = binge pattern

---

## Survival Features

Subscription lifecycle data for contractual business models (SVOD, SaaS).

| Feature | Type | Description | Calculation | Valid Range | Example | Source |
|---------|------|-------------|-------------|-------------|---------|--------|
| **subscription_id** | String | Unique subscription identifier | - | Non-empty | "SUB_789012" | Subscription DB |
| **customer_id** | String | Customer identifier (may = subscription_id) | - | Non-empty | "C_123456" | Subscription DB |
| **start_date** | Date | Subscription activation date | - | 2010-01-01 to present | 2023-06-01 | Subscription DB |
| **end_date** | Date | Subscription termination date | - | ≥ start_date or NULL | 2024-08-15 | Subscription DB |
| **status** | Categorical | Subscription status | - | {active, churned, cancelled, paused} | "churned" | Subscription DB |
| **T** | Integer | Subscription duration (days or months) | (end_date or observation_date) - start_date | > 0 | 440 (days) | Derived |
| **E** | Integer | Event indicator (churn) | 1 if churned, 0 if active/censored | 0 or 1 | 1 | Derived |
| **subscription_tier** | Categorical | Subscription level | - | {basic, premium, elite} | "premium" | Subscription DB |
| **subscription_tier_encoded** | Integer | Numeric encoding of tier | Ordinal encoding | 1, 2, 3 | 2 | Derived |
| **monthly_fee** | Float | Subscription price (USD/month) | - | > 0 | 14.99 | Pricing Catalog |

**Survival Analysis Concepts:**

**T (Duration)**:
- For churned subscriptions: Time from start_date to end_date
- For active subscriptions: Time from start_date to observation_period_end (censored)

**E (Event Indicator)**:
- E = 1: Customer churned (event observed)
- E = 0: Customer still active (right-censored, event not yet observed)

**Censoring**: Active customers are "censored" because we don't know when they'll churn.
Survival models handle censored observations statistically.

**Model Usage:**
- **Weibull AFT**: Uses T, E, and covariates (tier, demographics) to predict time-until-churn
- **sBG (Shifted Beta-Geometric)**: Uses cohort-level retention curves for churn prediction
- **Contractual CLTV**: CLTV = monthly_fee × expected_lifetime (from survival model)

---

## Prediction Outputs

Model predictions and confidence intervals.

| Feature | Type | Description | Model | Valid Range | Example |
|---------|------|-------------|-------|-------------|---------|
| **customer_id** | String | Customer identifier | - | Non-empty | "C_123456" |
| **prediction_date** | Date | Date of prediction generation | - | - | 2024-12-01 |
| **clv_12mo** | Float | Predicted 12-month CLTV (USD) | BG/NBD + Gamma-Gamma | ≥ 0 | 425.00 |
| **clv_24mo** | Float | Predicted 24-month CLTV (USD) | BG/NBD + Gamma-Gamma | ≥ 0 | 780.50 |
| **clv_36mo** | Float | Predicted 36-month CLTV (USD) | BG/NBD + Gamma-Gamma | ≥ 0 | 1,050.25 |
| **clv_95_ci_lower** | Float | Lower bound of 95% CI for CLTV | Parametric bootstrap | ≥ 0 | 350.00 |
| **clv_95_ci_upper** | Float | Upper bound of 95% CI for CLTV | Parametric bootstrap | ≥ clv_12mo | 500.00 |
| **p_alive** | Float | Probability customer is active | BG/NBD | 0.0 - 1.0 | 0.85 |
| **p_churn_30day** | Float | 30-day churn probability | BG/NBD + XGBoost | 0.0 - 1.0 | 0.12 |
| **p_churn_90day** | Float | 90-day churn probability | BG/NBD extrapolation | 0.0 - 1.0 | 0.18 |
| **expected_transactions_12mo** | Float | Expected purchases in 12 months | BG/NBD | ≥ 0 | 8.5 |
| **expected_value_per_transaction** | Float | Expected avg transaction value (USD) | Gamma-Gamma | > 0 | 50.00 |
| **clv_segment** | Categorical | Customer value segment | Quintile binning | {Q1_Low, Q2_Fair, Q3_Medium, Q4_High, Q5_Elite} | "Q4_High" |
| **percentile_rank** | Integer | CLTV percentile rank | Rank / total_customers * 100 | 0 - 100 | 85 |

**Prediction Interpretation:**

**clv_12mo**: Expected revenue from customer over next 12 months
- Accounts for: (1) probability customer is still active, (2) expected purchase frequency, (3) expected transaction value, (4) time value of money (1% monthly discount)
- Low CLV (<$100): Consider customer acquisition cost (CAC) efficiency
- Medium CLV ($100-$500): Target for retention campaigns
- High CLV (>$500): VIP/white-glove treatment

**p_alive**: Probability customer hasn't "died" (churned silently)
- p_alive > 0.8: Active customer, likely to transact again
- p_alive 0.5-0.8: Dormant but recoverable
- p_alive < 0.5: Likely churned, win-back campaign needed

**p_churn_30day**: Short-term churn risk (30-day horizon)
- p_churn < 0.1: Low risk
- p_churn 0.1-0.3: Moderate risk, monitor
- p_churn > 0.3: High risk, immediate intervention

**clv_segment**:
- Q1_Low (0-20th percentile): Low-value customers
- Q2_Fair (20-40th): Below-average value
- Q3_Medium (40-60th): Average value
- Q4_High (60-80th): Above-average value
- Q5_Elite (80-100th): Top 20%, whales

---

## Derived Metrics

Metrics calculated from predictions for business reporting.

| Metric | Calculation | Description | Target | Example |
|--------|-------------|-------------|--------|---------|
| **Total CLV** | SUM(clv_12mo) | Total revenue potential across customer base | - | $5.2M |
| **Average CLV** | AVG(clv_12mo) | Mean customer lifetime value | $400+ | $425 |
| **CLV Variance** | STDEV(clv_12mo) | Customer value heterogeneity | - | $150 |
| **Whale Concentration** | (Top 10% CLV / Total CLV) * 100 | % of revenue from top customers | <40% | 35% |
| **Churn Risk Count** | COUNT(p_churn_30day > 0.3) | Number of high-risk customers | <10% of base | 1,200 |
| **Average Churn Risk** | AVG(p_churn_30day) | Mean churn probability | <15% | 12% |
| **LTV:CAC Ratio** | Average CLV / Customer Acquisition Cost | Unit economics efficiency | >3:1 | 3.5:1 |
| **Customer Lifetime (months)** | T / 30 | Average customer age | - | 12.5 |

---

## Data Types & Constraints

### Data Types

| Type | Python | Polars | PostgreSQL | Description |
|------|--------|--------|------------|-------------|
| **String** | str | pl.Utf8 | VARCHAR | Text data, IDs |
| **Integer** | int | pl.Int64 | BIGINT | Whole numbers |
| **Float** | float | pl.Float64 | DOUBLE PRECISION | Decimal numbers |
| **Date** | datetime.date | pl.Date | DATE | Calendar date (no time) |
| **Datetime** | datetime.datetime | pl.Datetime | TIMESTAMP | Date + time (UTC) |
| **Categorical** | str (enum) | pl.Categorical | VARCHAR | Predefined categories |
| **Boolean** | bool | pl.Boolean | BOOLEAN | True/False |

### Constraints

**NOT NULL**: customer_id, transaction_dt, amount_usd (in raw transactions)
**UNIQUE**: transaction_id, subscription_id
**CHECK**: amount_usd > 0, frequency >= 0, T > 0
**FOREIGN KEYS**: customer_id references customers table (if normalized)

---

## Source Systems

| System | Type | Refresh Frequency | Data Coverage | Access Method |
|--------|------|-------------------|---------------|---------------|
| **Billing System** | PostgreSQL | Real-time | Transactions, payments | JDBC |
| **CRM** | Salesforce | Daily | Customer profiles, acquisition | API |
| **Streaming Analytics** | BigQuery | Hourly | Engagement, watch time | SQL |
| **Auth Service** | DynamoDB | Real-time | Login sessions | DynamoDB Streams |
| **Subscription DB** | MySQL | Real-time | Subscription lifecycle | CDC (Debezium) |
| **QoE Service** | Elasticsearch | 15-min | Buffering, quality metrics | REST API |

**Data Lineage:**
```
Raw Sources → Kedro data_processing Pipeline → Feature Store (Delta Lake) → Feast → Model Training/Inference
```

**Data Freshness:**
- **Training**: Uses historical data up to observation_period_end (batch)
- **Inference**: Features refreshed hourly (near real-time)
- **Feast Materialization**: Daily at 2 AM UTC

---

## Glossary

**CLTV**: Customer Lifetime Value - total net profit from a customer over their lifetime
**RFM**: Recency, Frequency, Monetary - customer segmentation framework
**Churn**: Customer attrition (stopping purchases or cancelling subscription)
**Censoring**: In survival analysis, when event (churn) hasn't occurred by observation end
**Observation Period**: Time window for calculating RFM features (e.g., 12 months)
**Whale**: High-value customer in top 10% of CLV distribution
**Binge-and-Burnout**: Pattern of intense engagement followed by abrupt churn

---

## Additional Resources

- [Lifetimes Library Documentation](https://lifetimes.readthedocs.io/)
- [RFM Analysis Guide](https://en.wikipedia.org/wiki/RFM_(market_research))
- [Survival Analysis Primer](https://lifelines.readthedocs.io/)
- Project-specific: `docs/technical_design.md` (Section 4: Data Architecture)
