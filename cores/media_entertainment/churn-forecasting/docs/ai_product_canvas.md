# AI Product Canvas: Churn Forecasting AI Core

## Section 1: AI Core Overview

**Industry**  
Media & Entertainment (Streaming Video, Audio Services, Gaming Platforms)

**AI Core Name**  
Churn Forecasting Engine

**External Name (Marketing punch)**  
Subscriber Churn Radar™

**Primary Function**  
Forecasting customer churn by detecting early signs of disengagement through holistic analysis of user behavior patterns, content consumption trajectories, quality-of-experience metrics, and social/peer influences across 30/60/90-day prediction horizons.

**How it helps**  
Transforms reactive "firefighting" retention cultures into proactive intervention strategies by identifying at-risk subscribers 30-90 days before cancellation—enabling personalized content recommendations, targeted incentives, and technical support interventions that prevent voluntary churn and recover involuntary churn from payment failures.

**Models to Apply**
- **Ensemble Methods**: Random Forest, XGBoost, LightGBM, CatBoost in soft voting configuration for robust tabular feature prediction
- **Deep Learning**: CNN-BiLSTM with Multi-Head Self-Attention for temporal pattern extraction from raw event sequences
- **Advanced Architectures**:
  - Hybrid Graph Attention Networks (GAT + MLP) capturing peer-influenced churn propagation in social viewing contexts
  - TBformer: Multi-scale Transformer with time-behavior attention for multi-modal behavior fusion
  - CCP-Net: Coordinate Attention Mechanism for spatial-temporal feature weighting
- **Class Balancing**: SMOTE, ADASYN, and weighted loss functions addressing 5-15% typical churn rates
- **Explainability**: SHAP (global feature importance) + LIME (local instance explanations) for marketing transparency

**Inputs**
- **Data Sources**: 
  - Internal: Viewing logs, subscription systems, payment gateways, CDN/QoE telemetry, customer service platforms, device telemetry
  - External: Content metadata APIs, competitive pricing feeds, seasonal/holiday calendars
- **Formats**: Structured tabular (subscription data), semi-structured event streams (viewing logs), unstructured (support tickets for NLP sentiment)
- **Frequency**: Real-time event streams (QoE, interactions), daily batch (subscription status, payment outcomes), weekly model retraining cycles

**Outputs**
- **Deliverables**: 
  - Per-subscriber churn probability scores (30/60/90-day horizons)
  - Risk-tier classifications (Low/Medium/High/Critical)
  - Feature attribution reports identifying top churn drivers
  - Personalized intervention recommendations (content, pricing, support)
  - Cohort-level churn analytics by acquisition channel, device, content preference
- **Format**: REST API (JSON), real-time event stream (Kafka), dashboard visualizations (Tableau/Looker)
- **Consumption**: 
  - Marketing automation platforms (Braze, Iterable) for triggered campaigns
  - CRM systems (Salesforce) for customer success workflows
  - Product analytics dashboards for content strategy decisions
  - Payment recovery systems for involuntary churn intervention

**Business Outcomes**
- 15-27% reduction in gross churn rate (4.1% → 3.0-3.5%)
- 86% increase in payment failure recovery rate (35% → 65%)
- 71% reduction in time-to-intervention (35 days → 10 days)
- $86.2M annual net revenue protection on 5M subscriber base
- 15 percentage point increase in subscribers achieving 3hr/week watch time threshold (65% → 80%)

**Business Impact KPIs**
| KPI Category | Primary KPIs (Direct) | Secondary KPIs (Indirect) | Leading vs. Lagging |
|--------------|------------------------|----------------------------|---------------------|
| **Retention** | Gross churn rate, Net churn rate | Retention rate by cohort | Lagging |
| **Engagement** | Watch time/day, 3hr/week threshold achievement | Content completion rate | Leading |
| **Revenue** | ARPU stability, Payment recovery rate | Ad-tier conversion rate | Lagging/Leading |
| **Operational** | Intervention success rate, Time-to-intervention | Support ticket volume | Leading |
| **Model Quality** | AUC-ROC (0.9626 target), Precision/Recall | Feature drift detection | Leading |

## Section 2: Problem Definition

**What is the problem?**  
The streaming media industry faces accelerating subscriber churn in a saturated market (339M+ total subscriptions) with declining growth rates (11% YoY in 2025 vs. 22% in 2023). Subscribers exhibit "content chasing" behavior—subscribing for specific shows then canceling—while experiencing subscription fatigue across 4.3 services per household. Current retention strategies are reactive (triggered at cancellation) rather than proactive, missing the 30-90 day pre-churn behavioral decay window when interventions are most effective.

**Why is it a problem?**  
- **Revenue Impact**: 4.1% average monthly gross churn on 5M subscribers = 205,000 cancellations/month = $22.5M annual revenue loss at $15 ARPU
- **CAC Inefficiency**: Customer acquisition costs ($80-120) exceed first-year revenue for churned subscribers, creating negative unit economics
- **Market Saturation**: With 78% of households subscribing to ≥3 services, growth requires stealing share rather than expanding market—making retention 5x more cost-effective than acquisition
- **Platform Loyalty Erosion**: Viewers prioritize specific content over platforms (73% would cancel if favorite show moved), requiring continuous engagement to maintain stickiness

**Whose problem is it?**  
- **Primary Owner**: Chief Revenue Officer / Chief Customer Officer
- **Business Units**: 
  - Product Team (engagement features, QoE optimization)
  - Marketing (retention campaigns, win-back programs)
  - Content Strategy (acquisition/production decisions based on retention impact)
  - Customer Success (proactive support interventions)
- **Affected Stakeholders**: 
  - Finance (revenue forecasting accuracy)
  - Engineering (QoE infrastructure investments)
  - Executive Leadership (subscriber growth targets, valuation metrics)

## Section 3: Hypothesis & Testing Strategy

**What will be tested?**
1. **Hypothesis A**: Subscribers exhibiting >40% decline in watch time over 30 days AND >7 days of inactivity have 8.2x higher churn probability within 60 days than baseline population
2. **Hypothesis B**: Personalized content recommendations triggered at 65% churn probability threshold increase 90-day retention by 24% compared to generic retention offers
3. **Hypothesis C**: Proactive payment retry workflows triggered within 24 hours of failure recover 65% of involuntary churn cases versus 35% with standard 72-hour retry cycles

**Expected responses for each hypothesis**
| Hypothesis | Success Criteria | Confidence Threshold | Minimum Sample Size |
|------------|------------------|----------------------|---------------------|
| A | AUC-ROC >0.92 for combined signal | 95% CI, p<0.01 | 50,000 subscribers |
| B | 20%+ relative lift in 90-day retention | 90% CI, p<0.05 | 100,000 subscribers (50k test/50k control) |
| C | 25%+ absolute increase in payment recovery | 95% CI, p<0.01 | 25,000 failed payments |

**Strategy**
- **Phase 1 (Weeks 1-4)**: Retrospective validation on historical churn data to confirm predictive power of behavioral signals
- **Phase 2 (Weeks 5-12)**: Small-scale A/B test (5% of at-risk population) with control group receiving standard retention flows
- **Phase 3 (Weeks 13-24)**: Gradual rollout to 25% → 50% → 100% of at-risk population with continuous monitoring
- **Measurement Framework**: 
  - Primary metric: 90-day retention rate difference between treatment/control
  - Guardrail metrics: Revenue per subscriber, support ticket volume, content consumption diversity
  - Holdout analysis: 5% permanent control group for long-term incrementality measurement

## Section 4: Solution Design

**What will be the solution?**  
A hybrid AI architecture combining three complementary modeling approaches within a Lambda processing framework:

1. **Speed Layer (Real-Time)**: Lightweight XGBoost model processing event streams for immediate intervention triggers (e.g., payment failure + buffering spike → instant support alert)
2. **Batch Layer (Deep Learning)**: CNN-BiLSTM with Multi-Head Self-Attention trained on 90-day behavioral sequences to detect subtle engagement decay patterns invisible to tabular models
3. **Graph Layer (Relational)**: Hybrid Graph Attention Network modeling peer influence effects—critical for social viewing platforms where churn propagates through friend networks

The system ingests raw event streams into an Activity Schema data warehouse, transforms them into both fixed-length tensors (for deep learning) and engineered features (for ensembles), then fuses predictions via weighted ensemble with attention-based weighting.

**Type of Solution**  
Hybrid approach combining:
- Classical ML (Ensemble Methods for tabular features)
- Deep Learning (CNN-BiLSTM for sequential behavior)
- Graph Neural Networks (for social/peer influence)
- Time Series Analysis (for temporal decay patterns)

**Expected Output**  
```json
{
  "subscriber_id": "SUB_982341",
  "prediction_timestamp": "2026-01-29T08:30:00Z",
  "churn_probabilities": {
    "30_day": 0.74,
    "60_day": 0.86,
    "90_day": 0.93
  },
  "risk_classification": "Critical",
  "churn_risk_score": 91,
  "serial_churner_flag": false,
  "behavioral_signals": {
    "watch_time_trend_30d_pct": -62,
    "days_since_last_view": 12,
    "avg_session_duration_min": 18,
    "content_completion_rate": 0.34,
    "buffering_ratio_increase_pct": 215
  },
  "top_churn_drivers": [
    {"feature": "watch_time_decline_30d", "shap_value": -0.42, "impact": "High"},
    {"feature": "days_inactive", "shap_value": -0.31, "impact": "High"},
    {"feature": "low_content_completion", "shap_value": -0.18, "impact": "Medium"}
  ],
  "recommended_interventions": [
    {"action": "email_personalized_content", "priority": 1, "content_ids": ["S12345", "M67890"]},
    {"action": "offer_discount_30pct", "priority": 2, "valid_days": 7},
    {"action": "proactive_support_contact", "priority": 3, "reason": "technical_frustration"}
  ],
  "subscriber_context": {
    "tenure_months": 18,
    "plan_tier": "Premium",
    "primary_device": "Smart TV",
    "preferred_genre": "Drama",
    "acquisition_cohort": "2024-07"
  },
  "model_metadata": {
    "version": "v3.2.1",
    "accuracy": 0.958,
    "auc_roc": 0.9626,
    "prediction_horizon_days": 30
  }
}
```

## Section 5: Data

**Source**
- **Internal**: 
  - Viewing logs (event-level: play/pause/seek/completion)
  - Subscription management system (status changes, plan tiers)
  - Payment gateway (transaction success/failure, retries)
  - CDN/QoE telemetry (buffering events, startup time, bitrate)
  - Customer service platform (tickets, chat logs, sentiment)
  - Device telemetry (app crashes, OS versions, screen sizes)
- **External**:
  - Content metadata APIs (genre taxonomy, release dates, original vs. licensed)
  - Competitive intelligence feeds (pricing changes, content launches)
  - Seasonal/holiday calendars (sports events, award shows)

**Inputs**  
*Mandatory Data Skeleton (Minimum Viable Dataset)*:
| Variable Category | Mandatory Variables | Data Type | Criticality |
|-------------------|---------------------|-----------|-------------|
| **Identifiers** | `subscriber_id` (hashed) | String | Critical: Primary key for entity resolution |
| **Temporal** | `event_timestamp` (ISO 8601) | Timestamp | Critical: Sequence ordering for LSTM tensors |
| **Behavioral** | `session_duration`, `activity_count`, `content_completion` | Integer/Float | High: Core engagement proxies |
| **Contractual** | `subscription_status`, `tenure_days`, `plan_tier` | Boolean/Integer/String | Critical: Target variable + segmentation |
| **Transactional** | `payment_success_flag`, `transaction_amount` | Boolean/Float | Critical: Involuntary vs. voluntary churn differentiation |
| **Technical QoE** | `service_error_count`, `rebuffer_ratio` | Integer/Float | Critical: "Frustration churn" predictor |

*Optional but High-Value Variables*:
- Content diversity index (0.0-1.0 scale)
- Social graph degree (friend connections, watch parties)
- Sentiment score from NLP analysis of support interactions
- Device ecosystem metadata (Smart TV 60%, Mobile 35%, Desktop 5%)
- Marketing response signals (email/push engagement)

**Quality**
- **Completeness**: <5% missing values for mandatory variables; imputation via forward-fill for time-series gaps
- **Accuracy**: Payment success flags validated against bank settlement reports; QoE metrics calibrated against synthetic monitoring
- **Consistency**: Schema enforcement via Apache Iceberg with column-level validation rules; automated drift detection alerts
- **Timeliness**: 
  - Real-time streams: <60 second latency for intervention triggers
  - Batch features: Daily refresh with 4-hour SLA
  - Model retraining: Weekly cadence with 90-day rolling window

**Access vs. Availability**
- **Access**: Managed via Kedro DataCatalog with role-based permissions; PII fields (email, name) require explicit approval; production models access only anonymized/hashed identifiers
- **Availability**: 
  - Historical depth: Minimum 6 months required (12 months optimal for seasonality capture)
  - Uptime SLA: 99.95% for prediction API; 99.5% for batch feature pipeline
  - Real-time vs. batch: Hybrid architecture supporting both event-driven triggers and scheduled predictions

**PII Gatekeeping**
- All raw data undergoes mandatory PII masking before entering feature store:
  - Direct identifiers (email, phone, name) removed at ingestion
  - IPs anonymized via /24 truncation
  - Device IDs hashed with salted SHA-256
  - Presidio NLP library scans unstructured text for residual PII
- Feature store contains only anonymized behavioral aggregates and hashed identifiers
- Model training occurs exclusively on de-identified data; production inference uses tokenized subscriber IDs

**Process/Transformation**
1. **Ingestion**: Event streams → Kafka → Flink (real-time validation) → Delta Lake (bronze layer)
2. **Validation**: Great Expectations rules applied per data domain (e.g., session_duration > 0, payment_amount ≥ 0)
3. **Feature Engineering**:
   - *For Ensembles*: Sliding window aggregations (7/14/30-day RFM metrics), trend slopes, ratios (completion rate)
   - *For Deep Learning*: Fixed-length sequence construction (90-day windows → 3D tensors: Samples × Time Steps × Features), Min-Max normalization
   - *For Graph Models*: Synthetic similarity graphs based on content co-consumption patterns
4. **Class Balancing**: SMOTE applied ONLY to training set (not validation/test) with k=5 neighbors for synthetic minority examples
5. **Serving**: Curated features published to Feast feature store with point-in-time correctness guarantees

**Outputs**
| Output Category | Specific Predictions | Format | Granularity | Update Frequency |
|-----------------|----------------------|--------|-------------|------------------|
| Churn Probability | 30/60/90-day risk scores | Numeric (0-1) | Per subscriber | Daily (batch), Real-time (triggers) |
| Risk Segmentation | Low/Medium/High/Critical tiers | Categorical | Per subscriber | Daily |
| Feature Attribution | Top 5 churn drivers with SHAP values | JSON + Visualization | Per subscriber | Weekly (on-demand) |
| Intervention Queue | Prioritized actions by urgency | Event stream | Per subscriber | Real-time |
| Cohort Analytics | Churn rate by acquisition channel/device | Aggregated % | Cohort-level | Monthly |
| Model Monitoring | Accuracy, AUC, feature drift metrics | Time series | Model-level | Hourly |

**Test/Train/Validation Split**
- **Temporal Split**: Critical for time-series data to prevent lookahead bias
  - Training: T-365 to T-90 days
  - Validation: T-90 to T-30 days (hyperparameter tuning)
  - Test: T-30 to T-0 days (final evaluation)
- **Stratified Sampling**: Applied within time windows to maintain churn class distribution (5-15% positive class)
- **Cohort Holdout**: 5% of newest acquisition cohorts held out entirely to test generalization to unseen user segments
- **Backtesting Protocol**: Rolling origin evaluation with 30-day steps to simulate real-world deployment conditions

## Section 6: Actors & Stakeholders

**Who is your client?**  
Chief Revenue Officer (CRO) with P&L responsibility for subscriber retention and lifetime value

**Who are your stakeholders?**
| Stakeholder Group | Role | Involvement Level |
|-------------------|------|-------------------|
| **Data Engineering** | Build/maintain pipelines, feature store | High (co-development) |
| **ML Engineering** | Model deployment, MLOps infrastructure | High (implementation) |
| **Product Management** | Define intervention workflows, UX | Medium (requirements) |
| **Marketing** | Execute retention campaigns, measure lift | High (consumption) |
| **Content Strategy** | Inform acquisition/production decisions | Medium (analytics) |
| **Customer Success** | Proactive support interventions | Medium (consumption) |
| **Legal/Compliance** | Privacy review, regulatory adherence | Low (gatekeeping) |
| **Finance** | Revenue forecasting, ROI validation | Medium (measurement) |

**Who is your sponsor?**  
Chief Executive Officer (CEO) with direct accountability for subscriber growth targets and market valuation metrics

**Who will use the solution?**
- Marketing Operations Specialists: Daily review of high-risk subscriber lists for campaign execution
- Customer Success Managers: Weekly outreach to Critical-tier subscribers with personalized retention offers
- Product Analysts: Monthly cohort analysis to identify engagement friction points requiring UX improvements
- Content Acquisition Leads: Quarterly review of content-specific churn patterns to guide licensing decisions

**Who will be impacted by it?**
- **Subscribers**: Receive more relevant content recommendations and timely support interventions; reduced exposure to generic promotional spam
- **Customer Support Agents**: Shift from reactive cancellation handling to proactive retention conversations (30% reduction in cancellation-related tickets)
- **Content Creators/Partners**: Increased visibility for content demonstrating high retention impact (influencing future production investments)
- **Competitors**: Market-wide elevation of retention standards forcing industry-wide innovation in engagement strategies

## Section 7: Actions & Campaigns

**Which actions will be triggered?**
| Action Type | Trigger Condition | Automation Level | Example |
|-------------|-------------------|------------------|---------|
| **Automated** | Churn probability >85% + payment failure | Fully automated | Instant SMS: "We noticed payment issue—update card to keep watching [Show X]" |
| **Automated** | Buffering ratio >5% for 3+ sessions | Fully automated | App notification: "Experiencing playback issues? Try these fixes →" + CDN optimization |
| **Human-in-loop** | Churn probability 65-85% + high ARPU | Semi-automated | CS agent receives alert with subscriber context + recommended talking points |
| **Human-in-loop** | Serial churner pattern detected | Semi-automated | Retention specialist reviews history before offering annual plan incentive |
| **Batch Campaign** | Cohort-level churn risk spike | Scheduled | Email campaign to "Sports fans" cohort before NFL season kickoff |

**Which campaigns?**
1. **Proactive Content Re-engagement**: 
   - *Trigger*: Watch time decline + no new content engagement in 14 days
   - *Action*: Personalized email highlighting new releases in preferred genres + "Continue Watching" carousel
   - *Expected Lift*: 24% increase in 30-day retention vs. control

2. **Payment Recovery Workflow**:
   - *Trigger*: Payment failure event
   - *Action*: Multi-channel retry sequence (email → SMS → in-app) with simplified payment update flow + 48-hour grace period
   - *Expected Lift*: 65% payment recovery rate (vs. 35% baseline)

3. **Tier Migration Program**:
   - *Trigger*: High churn risk + price sensitivity signals (browsing cheaper plans)
   - *Action*: Offer migration to ad-supported tier with 30-day premium trial of new content
   - *Expected Lift*: 70% conversion rate for at-risk subscribers; 42% 90-day retention post-migration

4. **Technical QoE Intervention**:
   - *Trigger*: Buffering ratio >3% + session abandonment within 60 seconds
   - *Action*: Proactive support ticket creation + CDN optimization for subscriber's region + compensation offer (free month)
   - *Expected Lift*: 58% reduction in churn within 30 days for affected subscribers

5. **Live Event Retention Campaign**:
   - *Trigger*: Churn risk 45-65% + upcoming major event in preferred category (sports/awards)
   - *Action*: Personalized notification: "Don't miss [Event]—your subscription renews in 3 days" + exclusive pre-show content
   - *Expected Lift*: 31% reduction in churn during event week vs. non-event weeks

## Section 8: KPIs & Evaluation

**How to evaluate the model?**
| Metric Category | Specific Metrics | Target Threshold | Purpose |
|-----------------|------------------|------------------|---------|
| **Discrimination** | AUC-ROC | ≥0.95 | Overall ranking quality |
| **Classification** | Precision @ Top 10% | ≥85% | Minimize false positives in high-cost interventions |
| **Classification** | Recall @ 80% Precision | ≥88% | Capture majority of true churners |
| **Calibration** | Brier Score | ≤0.08 | Probability reliability for business decisions |
| **Business** | Intervention Success Rate | ≥42% | % of at-risk subscribers retained post-intervention |
| **Business** | Revenue Protected per $1 Spent | ≥$8.62 | ROI of retention marketing spend |

**Which metrics should be aligned?**
- Model AUC-ROC → Gross churn rate reduction (business outcome)
- Precision → Marketing campaign efficiency (cost per retained subscriber)
- Recall → Total subscribers saved (revenue protection)
- Feature attribution quality → Content strategy decisions (production ROI)

**How much uncertainty can we handle?**
- **Prediction Confidence**: Minimum 70% probability threshold for high-cost interventions (discounts); 50% threshold for low-cost interventions (content recommendations)
- **Decision Boundaries**: Dynamic thresholds based on subscriber value tier:
  - Whale subscribers (top 5% ARPU): Trigger at 45% churn probability
  - Mid-tier subscribers: Trigger at 65% churn probability  
  - Low-value subscribers: Trigger at 80% churn probability
- **Risk Tolerance**: 
  - False positive tolerance: 15% (acceptable waste in retention spend)
  - False negative tolerance: 12% (acceptable missed retention opportunities)
  - Business context adjustment: Higher tolerance during content droughts; lower tolerance during major event windows

**A/B Testing Strategy**
| Component | Control Group | Treatment Group | Duration | Success Criteria |
|-----------|---------------|-----------------|----------|------------------|
| **Model Version** | v2.1 (XGBoost only) | v3.2 (Hybrid CNN-BiLSTM + GAT) | 8 weeks | 5%+ lift in AUC-ROC |
| **Intervention Timing** | Trigger at 14 days pre-churn | Trigger at 30 days pre-churn | 12 weeks | 15%+ lift in 90-day retention |
| **Offer Type** | Generic 20% discount | Personalized content + targeted discount | 10 weeks | 20%+ lift in intervention success rate |
| **Measurement Protocol** | | | | |
| - Minimum detectable effect: 3% relative lift in retention | | | |
| - Statistical power: 80% (β=0.2) | | | |
| - Significance threshold: p<0.05 (two-tailed) | | | |
| - Guardrail monitoring: Revenue per subscriber, support volume | | | |

## Section 9: Value & Risk

**What is the size of the problem?**
- **Annual Revenue at Risk**: $22.5M on 5M subscriber base (4.1% monthly churn × $15 ARPU × 12 months)
- **Subscriber Impact**: 2.46M cancellations annually requiring replacement via costly acquisition ($80-120 CAC)
- **Market Context**: Industry-wide $14B annual revenue loss from preventable churn across streaming sector

**What is the baseline?**
- Current retention approach: Reactive cancellation flows with 18% win-back success rate
- Average time-to-intervention: 35 days pre-churn (too late for meaningful engagement recovery)
- Model performance baseline: 82-88% accuracy, 0.85 AUC-ROC using logistic regression on aggregated features
- Payment recovery rate: 35% for failed transactions

**What is the uplift/savings?**
| Scenario | Gross Churn Rate | Subscribers Saved/Year | Revenue Protected | Net Impact (After $11M Investment) |
|----------|------------------|------------------------|-------------------|-----------------------------------|
| **Conservative** | 3.8% (-7.3%) | 180,000 | $32.4M | +$21.4M (195% ROI) |
| **Expected** | 3.2% (-22%) | 540,000 | $97.2M | +$86.2M (783% ROI) |
| **Optimistic** | 2.8% (-31.7%) | 840,000 | $151.2M | +$140.2M (1,275% ROI) |

*Additional Value Streams*:
- Watch time increase (55 → 70 min/day) driving 750,000 additional retained subscribers = $135M revenue protection
- Payment recovery improvement (35% → 65%) saving 180,000 involuntary churn cases = $32.4M revenue protection
- Reduced support costs from proactive interventions: $4.2M annual savings

**What are the risks?**
| Risk Category | Specific Risk | Probability | Impact | Mitigation Strategy |
|---------------|---------------|-------------|--------|---------------------|
| **Technical** | Concept drift from changing viewing patterns | Medium | High | Weekly retraining + automated drift detection alerts |
| **Technical** | Feature pipeline failure causing stale predictions | Low | Critical | Dual pipeline architecture with fallback to last-known-good features |
| **Business** | Over-intervention causing subscriber fatigue | Medium | Medium | Caps on intervention frequency (max 2/week per subscriber) |
| **Business** | Discounting eroding ARPU without retention lift | Medium | High | Tiered intervention strategy prioritizing content over discounts |
| **Regulatory** | GDPR/CCPA compliance for behavioral tracking | High | Medium | Privacy-by-design architecture; explicit consent for personalization |
| **Reputational** | Algorithmic bias against demographic segments | Low | Critical | Fairness audits using AIF360; demographic parity constraints in training |

**What might these risks block?**
- **Worst-case scenario**: Model degradation causing false high-risk flags → excessive discounting → $15M revenue loss + subscriber expectation inflation
- **Mitigation**: 
  - Shadow mode deployment for first 30 days (predictions generated but not acted upon)
  - Gradual rollout with kill switch triggered at <5% intervention success rate
  - Continuous monitoring of ARPU impact by intervention cohort
  - Independent fairness audit before full deployment

## Section 10: Theoretical Foundations of the AI Core

**Domain-Specific Frameworks**
- **Survival Analysis**: Cox Proportional Hazards models adapted for time-to-churn prediction with time-varying covariates (engagement metrics)
- **RFM Framework Extension**: Recency/Frequency/Monetary analysis enhanced with behavioral velocity metrics (session decay rates, content diversity trends)
- **Frustration-Effort Model**: Technical QoE metrics (buffering, errors) quantified as "frustration events" that accelerate churn hazard rates
- **Social Contagion Theory**: Churn propagation modeled as infection process through social graphs (validated in MMORPG research showing 3.2x higher churn risk when 2+ friends churn)
- **Content Chasing Economics**: Behavioral economics framework modeling subscription decisions as rational responses to content release calendars rather than platform loyalty

**Machine Learning Foundations**
- **Ensemble Methods**: 
  - Gradient boosting (XGBoost/LightGBM) for high-dimensional tabular feature spaces with built-in handling of mixed data types
  - Feature importance via SHAP values providing game-theoretic attribution for business interpretability
- **Deep Learning for Sequences**:
  - CNN layers for local pattern detection (e.g., "micro-churn": login without playback for 3 consecutive days)
  - BiLSTM layers for long-term dependency modeling (6-month engagement decay trajectories)
  - Multi-Head Self-Attention for weighting critical events (payment failure carries 4.7x more weight than routine login)
- **Graph Neural Networks**:
  - Graph Attention Networks (GAT) learning attention coefficients for neighbor influence in churn propagation
  - Message passing framework aggregating signals from similar users based on content co-consumption patterns
- **Class Imbalance Handling**:
  - SMOTE generating synthetic minority examples in feature space rather than raw data space
  - Focal loss functions dynamically weighting hard-to-classify examples during training

## Section 11: Data Architecture & Engineering

### 11.1 Mandatory Data Variables: The Skeleton

| Variable Category | Abstract Definition | Mandatory Variables | Data Type | Criticality & Usage |
|-------------------|---------------------|---------------------|-----------|---------------------|
| **Entity Identity** | $E_{id}$ | `subscriber_id` (hashed) | String | Critical: Primary key for joining disparate logs |
| **Temporal Anchor** | $T_{event}$ | `event_timestamp` | Timestamp (ISO 8601) | Critical: Sequence ordering for BiLSTM tensors |
| **Subscription State** | $Y_{state}$ | `subscription_status` | Binary (0=Active, 1=Churned) | Critical: Target label for supervised learning |
| **Value Exchange** | $V_{exchange}$ | `transaction_amount`, `payment_success_flag` | Float, Boolean | Critical: Differentiates involuntary vs. voluntary churn |
| **Interaction Vector** | $I_{act}$ | `session_duration`, `activity_count`, `content_completion_rate` | Integer, Float | High: Primary engagement proxies; decreasing trends signal churn |
| **Quality of Experience** | $Q_{exp}$ | `service_error_count`, `rebuffer_ratio` | Integer, Float (%) | Critical: "Frustration churn" predictor; high error rates correlate with immediate exit |
| **Tenure Duration** | $T_{duration}$ | `account_tenure_days` | Integer | High: Survival analysis baseline; hazard rates highest in months 0-3 |

### 11.2 Data Transformation and Engineering

**For Deep Learning (Sequence Generation)**
- **Tensor Construction**: Raw event logs → fixed-length sequences (90 days) → 3D tensors [Samples × Time Steps × Features]
  - Short histories: Zero-padding with masking layers to ignore padded timesteps
  - Long histories: Truncation to most recent 90 days (optimal window per ablation studies)
- **Normalization**: Min-Max scaling per feature to [0,1] range preventing gradient explosion during backpropagation
- **Temporal Embeddings**: Learned embeddings for time-of-day and day-of-week capturing circadian viewing patterns

**For Ensemble Methods (Feature Aggregation)**
- **Sliding Window Aggregations**:
  - Recency: Days since last session (log-transformed)
  - Frequency: Sessions per 7/14/30 days (with trend slopes)
  - Monetary: Spend velocity (rolling 30-day sum)
  - Behavioral: Watch time decay rate (linear regression slope over 30 days)
- **Categorical Encoding**: 
  - High-cardinality (device_type): Target encoding with smoothing
  - Low-cardinality (plan_tier): One-hot encoding
- **Interaction Features**: Cross-terms capturing synergistic effects (e.g., `buffering_ratio × session_abandonment_rate`)

**Class Balancing Protocol**
- SMOTE applied ONLY to training set after temporal split
- k=5 nearest neighbors in feature space for synthetic example generation
- ADASYN variant used for borderline examples near decision boundary
- Validation/test sets remain unmodified to reflect real-world class distribution

### 11.3 Online Dataset Search and Analysis

| Dataset Name | Domain | URL/Source | Relevance to AI Core |
|--------------|--------|------------|----------------------|
| **KKBox Churn Prediction** | Music Streaming | [Kaggle](https://www.kaggle.com/c/kkbox-churn-prediction-challenge) | High: Daily user logs with song play counts ideal for LSTM sequence modeling; achieved 0.9626 AUC with hybrid architectures |
| **Telco Customer Churn** | Subscription Services | [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) | Medium: Standard benchmark for tabular ensemble methods; limited temporal depth for deep learning |
| **Mobile Game Churn** | Gaming (F2P) | [Kaggle](https://www.kaggle.com/datasets/dimitaryanev/mobilechurndataxlsx) | High: Level completion + in-game currency data perfect for "whale" detection and engagement decay modeling |
| **Netflix Prize (Synthetic)** | Video Streaming | [Academic repositories] | Medium: Viewing history patterns valuable despite age; requires augmentation with modern QoE metrics |
| **SaaS Churn (AWS Sample)** | General Subscription | [AWS GitHub](https://github.com/aws-samples/real-time-churn-prediction) | High: Production-ready pipeline architecture for real-time inference using Lambda/SageMaker |

### 11.4 Gap Analysis and Deductive Engineering

| Gap Scenario | Deductive Engineering Approach | Validation Method |
|--------------|-------------------------------|-------------------|
| **Missing explicit churn label** | Construct "Quiet Churn" label: If `last_event_timestamp < (current_date - 30 days)` → mark as churned | Compare against actual cancellations where available; optimize inactivity window via ROC analysis |
| **Missing QoE metrics** | Infer technical friction from "Short Session Behavior": Sessions <10 seconds with no content playback → flag as technical failure | Correlate with app crash logs where available; validate via support ticket text mining |
| **Missing social graph** | Build synthetic similarity graph based on content co-consumption: Jaccard similarity of watched titles → edge weights | Validate against actual friend connections in platforms with social features; measure lift in prediction accuracy |
| **Missing content metadata** | Derive genre preferences via collaborative filtering: Matrix factorization on user-title interactions → latent genre vectors | Compare against explicit genre tags where available; use reconstruction error as quality metric |
| **GDPR-restricted demographics** | Create behavioral segments via clustering: K-means on viewing times (morning commuter vs. late-night binger) | Validate segment stability over time; measure predictive power for churn vs. static demographics |

## Section 12: Impact & Measurement

**What is the impact?**
- **Revenue Protection**: $86.2M net annual impact on 5M subscriber base (783% ROI)
- **Subscriber Retention**: 540,000 additional retained subscribers annually (+22% reduction in gross churn)
- **Operational Efficiency**: 71% reduction in time-to-intervention (35 → 10 days); 30% reduction in cancellation-related support tickets
- **Strategic Enablement**: Data-driven content acquisition decisions increasing content ROI by 18% (measured via retention impact per $1M spend)

**Where can you see the improvement?**
| Measurement Channel | Specific Metrics | Update Frequency | Audience |
|---------------------|------------------|------------------|----------|
| **Executive Dashboard** | Gross/net churn rate, revenue protected, ROI | Daily | C-Suite |
| **Marketing Ops Console** | Intervention success rate, cost per retained subscriber | Real-time | Marketing team |
| **Product Analytics** | Watch time trends, content completion rates by cohort | Daily | Product managers |
| **Model Monitoring** | AUC-ROC drift, feature importance stability, prediction latency | Hourly | ML engineers |
| **Customer Feedback** | NPS changes among intervened vs. control groups | Quarterly | Customer success |

**Success Criteria**
| Metric | Baseline (2025) | Target (12 Months) | Measurement Frequency | Owner |
|--------|-----------------|---------------------|------------------------|-------|
| Gross Churn Rate | 4.1% | 3.2% (-22%) | Monthly | CRO |
| Model AUC-ROC | 0.85 | 0.9626 | Weekly | ML Lead |
| Intervention Success Rate | 18% | 42% (+133%) | Weekly | Marketing Ops |
| Payment Recovery Rate | 35% | 65% (+86%) | Daily | Finance |
| 3hr/Week Threshold Achievement | 65% | 80% (+15pp) | Weekly | Product |
| Time-to-Intervention | 35 days | 10 days (-71%) | Daily | Engineering |
| ARPU Impact (Net) | Baseline | +$0.85/subscriber | Monthly | Finance |
| False Positive Rate | N/A | <15% | Weekly | ML Lead |
| System Uptime | 99.5% | 99.95% | Hourly | DevOps |
---