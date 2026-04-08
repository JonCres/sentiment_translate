# AI Product Canvas: A Unified Framework for AI Core Development

## Section 1: AI Core Overview

### Industry

Marketing (B2B Focus)

### AI Core Name

Voice of the Client

### External Name (Marketing punch)

Strategic Account Intelligence

### Primary Function

Analyze B2B customer feedback and NPS surveys using advanced NLP and sentiment analysis to extract strategic insights about client satisfaction, loyalty drivers, detractors, and improvement opportunities, enabling targeted relationship management and service optimization.

### How it helps

Converts NPS feedback into actionable intelligence by identifying themes in promoter/detractor responses, uncovering root causes of satisfaction scores, and highlighting specific improvement areas to strengthen client relationships and drive advocacy.

### Models to Apply

* **Zero-Shot Classification (DeBERTa/BART):** For taxonomy mapping without massive labeled datasets.
* **BERTopic:** For unsupervised theme discovery.
* **Aspect-Based Sentiment Analysis (PyABSA):** To disentangle mixed feedback.
* **KeyBERT/SpaCy:** For entity and keyphrase extraction.
* **XGBoost/LightGBM:** For churn prediction using tabular and sentiment features.
* **SHAP:** For model interpretability and driver analysis.

### Inputs

| Input Category | Specific Variables | Data Type | Frequency | Volume | Priority |
|----------------|-------------------|-----------|-----------|--------|----------|
| NPS Survey Responses | NPS score (0-10), timestamp, response ID | Numeric + Metadata | Quarterly/Bi-annual | 500-5K/quarter | Critical |
| Open-Ended Feedback | Free-text "Why did you give this score?" responses | Text | Quarterly/Bi-annual | 400-4K/quarter | Critical |
| Client Metadata | Client ID, industry, segment, ARR/ACV, account tier | Structured | Daily sync | Reference data | Critical |
| Account History | Previous NPS scores, renewal dates, tenure | Time-series | Daily sync | Reference data | High |
| Relationship Data | Account manager ID, CSM ID, executive sponsor | Structured | Weekly sync | Reference data | High |
| Support Interaction Data | Ticket count, resolution time, escalations | Numeric + Categorical | Real-time | Event stream | Medium |
| Engagement Metrics | Training completion, portal logins, event attendance | Behavioral | Daily | Event stream | Medium |

### Outputs

| Output Category | Specific Predictions | Format | Granularity | Update Frequency | Audience |
|-----------------|---------------------|--------|-------------|------------------|----------|
| At-Risk Client Alerts | Churn probability scores (0-1), alert severity | Boolean + Probability | Per account | Real-time | Account Managers |
| Aspect-Based Sentiment | Granular sentiment per dimension (Product, Support, etc.) | Numeric (-1 to +1) + Ranking | Per response + Aggregate | Real-time + Quarterly | Functional Leaders |
| Customer Health Score | Composite score (0-100) combining NPS, sentiment trajectory, engagement | Numeric | Per account | Weekly | Account Managers |
| Actionable Recommendations | Next-best-actions with owners/due dates | Structured workflow | Per account | Real-time + Quarterly | CS Managers |
| Priority Improvement Areas | Ranked issues weighted by frequency × revenue exposure | Rankings + Text | Portfolio | Quarterly | Executive Leadership |
| Root Cause Analysis | Top drivers with supporting quotes linked to org functions | Text + Entity links | Aggregate | Quarterly | Process Owners |

**Sample Output Record**:

```json
{
  "response_id": "NPS_Q4_2025_789456",
  "client_id": "ENT_12345",
  "client_name": "Acme Manufacturing Inc.",
  "response_timestamp": "2025-12-15T09:23:45Z",
  "nps_score": 6,
  "nps_category": "detractor",
  "previous_nps": 8,
  "nps_trend": "declining",
  "account_tier": "platinum",
  "annual_contract_value": 850000,
  "industry": "manufacturing",
  "tenure_months": 38,
  "renewal_date": "2026-03-31",
  "days_to_renewal": 106,
  
  "overall_sentiment": -0.42,
  "sentiment_label": "negative",
  "confidence": 0.91,
  
  "aspect_sentiments": {
    "account_management": 0.75,
    "product_quality": 0.45,
    "technical_support": -0.88,
    "implementation": -0.23,
    "training": -0.35,
    "billing": 0.12,
    "platform_stability": 0.52
  },
  
  "detected_topics": [
    "support_response_time",
    "technical_expertise_gap",
    "integration_complexity"
  ],
  
  "key_phrases": [
    "support takes days to respond",
    "escalations not handled properly",
    "account manager is excellent",
    "platform itself is solid"
  ],
  
  "entity_mentions": {
    "products": ["Core Platform", "Advanced Analytics Module"],
    "competitors": [],
    "people": ["Sarah (Account Manager)"],
    "functions": ["Technical Support"]
  },
  
  "recommended_actions": [
    {
      "action": "immediate_executive_escalation",
      "priority": "critical",
      "owner": "VP_Customer_Success",
      "due_date": "2025-12-16"
    },
    {
      "action": "assign_dedicated_support_manager",
      "priority": "high",
      "owner": "Support_Director",
      "due_date": "2025-12-18"
    },
    {
      "action": "schedule_technical_review",
      "priority": "high",
      "owner": "Technical_Account_Manager",
      "due_date": "2025-12-20"
    }
  ],
  
  "alert_triggered": true,
  "alert_severity": "critical",
  "alert_reason": "Detractor response from Platinum account with $850K ARR and renewal in 106 days. Sentiment declined from previous survey. High churn risk."
}
```

### Business Outcomes

**Strategic Decisions Enabled:**

1. **Strategic Product Roadmap:** Aligns development priorities with high-impact client needs by quantifying which features drive "Promoter" status.
2. **Account Health Visibility:** Provides Account Managers with specific talking points for QBRs (Quarterly Business Reviews), moving the conversation from generic check-ins to solving specific pain points.
3. **Pricing Strategy:** Identifies if "Price" is a detractor due to lack of value perception or actual budget constraints.
4. **Service Optimization:** Pinpoints specific support teams or regions that are underperforming based on sentiment analysis.

**Operational Improvements:**

1. **Detractor Rescue:** Enables rapid intervention on specific complaints before renewal discussions (closing the loop).
2. **Promoter Activation:** Automatically identifies satisfied clients suitable for case studies or referral programs.
3. **Feedback Routing:** Reduces manual reading time by automatically routing technical feedback to Product and service feedback to Support.

### Business Impact KPIs

* **Primary:** Net Promoter Score (NPS), Gross Revenue Retention (GRR), Net Revenue Retention (NRR).
* **Secondary:** Feedback Response Time, Detractor Resolution Rate.
* **Leading:** Sentiment Trend Slope, Feature Usage Drop-off.
* **Lagging:** Churn Rate, Renewal Rate.

### **B2B-Specific Implementation Considerations**

#### **Data Volume & Quality Challenges**

Unlike B2C systems processing millions of reviews, B2B VoC operates with hundreds to thousands of responses quarterly. This requires:
* **Transfer Learning**: Leverage pre-trained models fine-tuned on domain-specific B2B language rather than training from scratch
* **Zero-Shot Approaches**: Utilize classification models that don't require large labeled datasets
* **Data Augmentation**: Synthesize training examples while maintaining privacy compliance

#### **Privacy & HR Compliance**

B2B feedback frequently mentions specific individuals (account managers, support staff, executives):
* **PII Detection**: Implement NER models to identify person names
* **Anonymization Workflows**: Token-replace names in reports while preserving insights
* **HR Notification**: Structured feedback about employees routed to appropriate HR channels with context
* **Legal Review**: Ensure compliance with employment law regarding employee-related feedback

#### **Revenue-Weighted Analysis**

Not all feedback is equal; a detractor from a $4M account requires different treatment than one from a $200K account:
* **ARR Weighting**: All analyses should include revenue-weighted views alongside count-based views
* **Risk Scoring**: Combine sentiment severity with contract value and renewal proximity
* **Prioritization Algorithms**: Sort interventions by (Churn Probability × ARR × Time-to-Renewal)

#### **Account-Level Intelligence**

B2B requires understanding entire organizations, not individual users:
* **Multi-Stakeholder Analysis**: Track sentiment across different roles within same account (technical users, business owners, executives)
* **Buying Committee Insights**: Identify consensus vs. disagreement within accounts
* **Organizational Context**: Link feedback to org changes (mergers, leadership transitions, restructuring)

> **EXECUTIVE SUMMARY**
> The **Voice of the Client** AI Core is a B2B intelligence engine designed to protect revenue. Unlike B2C tools that analyze volume, this Core analyzes *value*. By applying Aspect-Based Sentiment Analysis and Churn Prediction models to dense B2B feedback, it identifies "silent" risks—such as a Decision Maker's hidden frustration despite a User's happiness. It targets a **10% reduction in churn** and provides explainable insights (via SHAP) to empower Account Managers to act 90 days before renewal, securing millions in at-risk ARR.

---

## Section 2: Problem Definition

### What is the problem?

B2B organizations suffer from the "Silent Exodus" paradox: stable NPS scores mask accelerating enterprise churn because traditional metrics fail to detect strategic relationship deterioration beneath surface-level satisfaction. Account teams reactively process sparse feedback manually, missing early warning signals until renewal discussions—when intervention is too late. A single detractor response from a Platinum account can represent $1M+ in at-risk ARR, yet organizations lack systems to prioritize high-value risks or disentangle mixed feedback addressing multiple service dimensions.

### Why is it a problem?

* **Revenue Concentration:** In B2B, losing one Enterprise client can cost **$100k–$5M+** in ARR.
* **Operational inefficiency**: Account Managers spend 15+ hours weekly manually reading/interpreting feedback instead of strategic engagement
* **Lagging Metrics:** NPS is often collected too late (post-incident).
* **Strategic blindness**: 68% of enterprise churn occurs despite "acceptable" NPS scores (7-8) because tactical satisfaction masks strategic misalignment  

### Whose problem is it?

- **Primary owner**: VP of Customer Success (account health accountability)  
* **Process owners**: Customer Success Managers, Account Managers (frontline retention)  
* **Affected stakeholders**: Product Leadership (roadmap misalignment), Sales Leadership (expansion revenue loss), C-Suite (revenue stability)  

> **PAIN POINTS**
>
> 1. **The "Silent Exodus" (Cost: ~$2M ARR/year)**
>     * *Current Workaround:* Account Managers rely on "gut feel" or generic QBRs.
>     * *Impact:* 85% of churned customers were not flagged as "At Risk" in the CRM prior to cancellation.
>     * *Latency:* Churn is detected only when the cancellation letter arrives (30-day notice).
>
> 2. **Mixed Sentiment Ambiguity**
>     * *Current Workaround:* Account Managers average conflicting feedback ("great product but terrible support") into single sentiment score.
>     * *Impact:* 34% of detractors misclassified as "neutral," delaying intervention until 30 days pre-renewal
>     * *Latency:* 45-60 days (requires manual deep-dive analysis)
>
> 3. **The "False Positive" Promoter (Cost: Missed Upsell)**
>     * *Current Workaround:* Manual reading of comments.
>     * *Impact:* A client gives a "9" (Promoter) for the support team but writes "Product is too expensive" in text. The system flags them as safe, missing the pricing objection.
>     * *Latency:* Risks are identified only during renewal negotiation (Lag: 11 months).

---

## Section 3: Hypothesis & Testing Strategy

### What will be tested?

1. **Hypothesis 1:** Aspect-Based Sentiment Analysis (ABSA) features will improve Churn Prediction accuracy (F1-score) by **>15%** compared to models using only structured data (usage/NPS).
2. **Hypothesis 2:** Zero-shot classification can categorize B2B feedback into 6 business taxonomies with **>80% accuracy** using fewer than 50 labeled examples per class.
3. **Hypothesis 3:** Interventions triggered by "Revenue-Weighted Risk Scores" will yield a **3x higher ROI** than interventions based solely on NPS score drops.

### Strategy

* **Methodology:** **Hybrid NLP + Tabular**. Combine text embeddings (DeBERTa) with structured CRM data (Contract Value, Tenure) in an XGBoost classifier.
* **Leading Indicators:** **Sentiment Slope**. A negative change in sentiment over 3 consecutive interactions is a stronger predictor of churn than absolute NPS.
* **Mathematical Segmentation:** **Hierarchical Aggregation**. Model "Decision Maker" sentiment with 3x weight compared to "End User" sentiment within the same Account ID.
* **Strict Governance:** **PII Redaction**. Use Presidio to mask names/emails before data enters the Feature Store to ensure GDPR compliance.
* **Experimental Validation:** **A/B Testing**. Split "At-Risk" accounts: Group A gets AI-driven "SuccessPlay" (intervention); Group B gets standard QBR. Measure retention delta.

---

## Section 4: Solution Design

### What will be the solution?

The solution is a **Composable AI Core** that ingests multi-channel feedback, processes it through a specialized NLP pipeline (cleaning -> classification -> ABSA -> entity extraction), enriches it with CRM metadata, and feeds it into a predictive engine. The output is pushed back into the CRM as an "Account Health Intelligence" widget.

### Type of Solution

**Hybrid Approach**:

* **Deep Learning (NLP):** For understanding text (Transformers).
* **Classical ML (XGBoost):** For risk scoring and churn prediction.
* **Analytics:** For dashboarding and aggregation.

**Core AI Capabilities**  

1. **Zero-Shot Taxonomy Mapping** (DeBERTa-v3)  
   * *Input*: Raw feedback text + business taxonomy labels  
   * *Output*: Probability distribution across 8 service dimensions  
   * *Necessity*: B2B environments lack labeled data for supervised training during initial deployment; enables immediate value without 6-month labeling cycles  

2. **Aspect-Based Sentiment Disentanglement** (RoBERTa + PyABSA)  
   * *Input*: Feedback text with dependency parse trees  
   * *Output*: Sentiment scores (-1 to +1) per aspect (Account Management, Support, etc.)  
   * *Necessity*: 78% of B2B feedback contains mixed sentiment; unified scores mask critical negatives requiring intervention  

3. **Dynamic Topic Evolution Tracking** (BERTopic + Temporal Embeddings)  
   * *Input*: Quarterly feedback corpus with timestamps  
   * *Output*: Topic clusters with growth/decay rates across time windows  
   * *Necessity*: Detects accelerating concerns (e.g., "strategic alignment" growing 125% QoQ) before they trigger churn  

4. **Revenue-Weighted Risk Scoring** (XGBoost + SHAP)  
   * *Input*: Aspect sentiment scores + ARR + days-to-renewal + tenure  
   * *Output*: Churn probability (0-1) + financial exposure score  
   * *Necessity*: Prioritizes interventions on $2M accounts over $50K accounts despite identical sentiment scores  

5. **PII-Aware Entity Extraction** (spaCy NER + Presidio)  
   * *Input*: Raw feedback text  
   * *Output*: Anonymized entities (products, competitors) + tokenized personnel mentions  
   * *Necessity*: 42% of B2B feedback mentions specific individuals; requires HR-compliant anonymization before analysis  

6. **Multi-Stakeholder Sentiment Aggregation** (Hierarchical Bayesian Modeling)  
   * *Input*: Individual sentiment scores + stakeholder role weights  
   * *Output*: Account-level health score weighted by decision-maker influence  
   * *Necessity*: B2B accounts have 6-10 stakeholders; economic buyer sentiment must dominate end-user sentiment in health scoring  

7. **Causal Driver Attribution** (DoWhy Framework)  
   * *Input*: Sentiment trends + operational metrics (support tickets, usage)  
   * *Output*: Causal impact estimates (e.g., "20% faster resolution → +8 NPS points")  
   * *Necessity*: Distinguishes correlation (high usage ↔ high NPS) from causation to prioritize effective interventions  

8. **Real-Time Alert Prioritization** (Multi-Criteria Decision Analysis)  
   * *Input*: Churn probability + ARR + competitive risk + renewal proximity  
   * *Output*: Ranked alert queue with severity tiers (Critical/High/Medium)  
   * *Necessity*: Prevents alert fatigue by surfacing only top 5% of accounts requiring immediate action  

**Sample Output JSON Schema**  

```json
{
  "response_id": "string",
  "client_id": "string",
  "account_tier": "enum[platinum|gold|silver]",
  "annual_contract_value": "number",
  "nps_score": "integer(0-10)",
  "nps_category": "enum[promoter|passive|detractor]",
  "nps_trend": "enum[improving|stable|declining]",
  "days_to_renewal": "integer",
  "overall_sentiment": "number(-1.0 to 1.0)",
  "sentiment_confidence": "number(0.0 to 1.0)",
  "aspect_sentiments": {
    "account_management": "number(-1.0 to 1.0)",
    "product_quality": "number(-1.0 to 1.0)",
    "technical_support": "number(-1.0 to 1.0)",
    "implementation": "number(-1.0 to 1.0)",
    "training": "number(-1.0 to 1.0)",
    "billing": "number(-1.0 to 1.0)",
    "platform_stability": "number(-1.0 to 1.0)",
    "strategic_partnership": "number(-1.0 to 1.0)"
  },
  "detected_topics": ["string"],
  "topic_growth_rate": "number",
  "key_phrases": ["string"],
  "entity_mentions": {
    "products": ["string"],
    "competitors": ["string"],
    "people_anonymized": ["string"]
  },
  "churn_probability": "number(0.0 to 1.0)",
  "revenue_at_risk": "number",
  "recommended_actions": [
    {
      "action_type": "enum[executive_escalation|support_assignment|roadmap_review]",
      "priority": "enum[critical|high|medium]",
      "owner_role": "string",
      "due_in_days": "integer",
      "talking_points": ["string"]
    }
  ],
  "alert_triggered": "boolean",
  "alert_severity": "enum[critical|high|medium]",
  "privacy_compliant": "boolean"
}
```

---

## Section 5: Data

### Source

* **Internal:** CRM (Salesforce), Survey Platform (Qualtrics), Support Desk (Zendesk).
* **External:** Industry Benchmarks (Optional).
* **Hybrid:** Internal data enriched with external firmographics (e.g., Crunchbase data on client funding status).

### Inputs

**Mandatory Data Skeleton:**

| Variable Name | Description | Data Type | Source System | Mandatory? |
| :--- | :--- | :--- | :--- | :--- |
| `account_id` | Unique Key for Company | String | CRM | **YES** |
| `feedback_text` | Raw open-ended comment | String | Survey/Support | **YES** |
| `nps_score` | 0-10 Rating | Integer | Survey | **YES** |
| `respondent_role` | Job Title/Level | Categorical | CRM | **YES** |
| `contract_value` | ARR/ACV | Float | CRM | **YES** |
| `interaction_date` | Date of feedback | Datetime | Survey | **YES** |
| `support_ticket_vol` | # of tickets last 90d | Integer | Support | Recommended |

*Strategically Necessary (High Accuracy)*  

| Variable Name | Description | Data Type | Source System | Mandatory? |
|---------------|-------------|-----------|---------------|------------|
| stakeholder_role | Influence level (Decision Maker/User/Influencer) | Categorical | CRM | Recommended |
| account_tier | Service level (Platinum/Gold/Silver) | Categorical | CRM | Recommended |
| industry_vertical | Client's primary industry segment | Categorical | CRM | Recommended |
| support_ticket_volume_30d | Rolling 30-day support interactions | Integer | Zendesk | Recommended |
| product_usage_score | Normalized feature adoption metric | Float | Product Analytics | Recommended |
| previous_nps_score | Most recent prior NPS measurement | Integer | Historical Data | Recommended |

### Quality

* **Completeness:** Text fields < 5 words are discarded (noise reduction).
* **Accuracy:** Role metadata must be validated against CRM contact records.
* **Consistency:** "NPS" vs "CSAT" scales normalized to 0-1 range.
* **Timeliness:** Survey data ingested daily; Support data real-time.

### Access vs. Availability

* **Access:** Managed via Kedro DataCatalog with role-based permissions; PII fields require "HR Analyst" role; raw feedback accessible only to anonymized feature store
* **Availability:** Requires min. 12 months historical data for seasonality.
* **PII Gatekeeping:** All text inputs undergo mandatory PII masking pipeline before model ingestion:  

1. Presidio NER detects names, emails, phone numbers  
2. Token replacement with role-based placeholders ("[ACCOUNT_MANAGER]")  
3. HR-sensitive feedback (personnel performance) routed to encrypted HR channel with manager notification  
4. Audit trail of all PII handling operations for GDPR/CCPA compliance  

### Outputs

| Output Category | Specific Predictions | Format | Granularity | Update Frequency |
| :--- | :--- | :--- | :--- | :--- |
| **Risk Intelligence** | Churn Probability, Revenue at Risk | Float | Account | Weekly |
| **Thematic Analysis** | Top 5 Pain Points | List[String] | Segment | Monthly |
| **Sentiment** | Aspect-Based Polarity | JSON | Interaction | Real-time |
| **Action** | Next Best Action | String | Account | Event-driven |

### Test/Train/Validation Split

* **Temporal Split:** Train on Jan-Sept, Validate on Oct, Test on Nov-Dec. (Essential to prevent data leakage from future events).
* **Stratified Sampling:** Ensure "Churned" accounts are equally represented in splits using SMOTE if necessary.

---

## Section 6: Actors & Stakeholders

### Who is your client?

**VP of Customer Success** or **Chief Revenue Officer (CRO)**.

### Who are your stakeholders?

* **Account Managers (AMs):** End-users of the alerts.
* **Product Managers:** Consumers of thematic insights.
* **Data Science Team:** Builders of the Core.
* **Sales Operations:** Integrators into CRM.

### Who is your sponsor?

**Chief Customer Officer (CCO)** (Budget Authority).

### Who will use the solution?

- Account Managers (daily alert triage, QBR preparation)  
* Customer Success Managers (health score monitoring, intervention planning)  
* VP of Customer Success (portfolio risk dashboards, resource allocation)  
* Product Managers (theme analysis for roadmap decisions)  

### Who will be impacted by it?

- Enterprise clients (more proactive relationship management)  
* Support teams (routed feedback requiring technical resolution)  
* Account Management leadership (performance metrics tied to AI-driven interventions)

> **USER PERSONAS**
>
> 1. **"The Blindsided AM" (Account Manager)**
>     * *Goal:* Hit renewal targets.
>     * *Pain:* "I thought the account was healthy because the user loved us, but the VP cancelled."
>     * *Need:* Stakeholder alignment maps and early warning of executive dissatisfaction.
>     * *Quote:* "Don't tell me they are unhappy after they send the cancellation letter."
>
> 2. **"The Firefighting VP" (VP of CS)**
>     * *Goal:* Reduce churn by 5%.
>     * *Pain:* Team spends time on low-value noise while high-value accounts churn silently.
>     * *Need:* Revenue-weighted risk dashboard to prioritize resources.
>     * *Quote:* "We need to save the whales, not just catch the minnows."
>
> 3. **"The Guessing PM" (Product Manager)**
>     * *Goal:* Build features that drive retention.
>     * *Pain:* Relying on anecdotal feedback from sales reps.
>     * *Need:* Quantified evidence of feature-related detractors.
>     * *Quote:* "Show me the data, not the loudest opinion."

---

## Section 7: Actions & Campaigns

### Which actions will be triggered?

* **Automated:**
  * **CRM Flag:** Mark account "At-Risk" in Salesforce.
  * **Slack Alert:** Notify Account Team channel for "Critical" risk (Churn Prob > 75% & ARR > $100k).
  * **Ticket Routing:** Auto-route "Bug" feedback to Engineering Jira, "Pricing" feedback to Sales.
* **Human-in-the-loop:**
  * **SuccessPlay:** Generate a draft "Recovery Plan" for the AM to review and send.

### Which campaigns?

* **"Detractor Rescue":** High-touch intervention for high-value detractors (Executive Sponsor call).
* **"Promoter Activation":** Automated email asking Promoters for a G2 Crowd review or Case Study participation.
* **"Technical Health Check":** Triggered by negative sentiment on "Stability" or "Bugs."

> **INTEGRATION STRATEGY**
>
> 1. **Workflow (Salesforce):**
>     * Write `Churn_Probability__c` and `Top_Pain_Point__c` to Account Object.
>     * Trigger `Task` creation: "Call Client re: [Pain Point]" if Risk > High.
> 2. **Data (Bi-directional):**
>     * **Read:** Nightly batch from Snowflake (Contract Data).
>     * **Write:** Real-time API push to CRM (Analysis results).
> 3. **UX (Embedded):**
>     * **LWC (Lightning Web Component):** "Voice of Client" tab in Salesforce Account View showing historical sentiment trend chart.

---

## Section 8: KPIs & Evaluation

### How to evaluate the model?

*Technical Metrics*  
* ABSA: Aspect-level F1-score (>0.75) validated against human annotations  
* Churn Prediction: AUC-ROC (>0.85), precision@top10% accounts (>0.80)  
* Zero-Shot Classification: Accuracy (>0.75) on 8 business taxonomies  
* Alert System: Precision (>0.70) and recall (>0.65) for critical-risk accounts  

*Business Metrics*  
* ARR saved from interventions / total at-risk ARR  
* Time-to-first-response for detractor alerts  
* % of flagged accounts renewed within 90 days  

### Which metrics should be used?

* **Strategic:** Net Revenue Retention (NRR).
* **Operational:** Time-to-Action (Survey submission to AM contact).

### How much uncertainty can we handle?

* **Churn Probability:** Thresholds tuned for **Recall** over Precision (better to flag a safe account than miss a risky one), but strictly filtered by Contract Value to prevent alert fatigue.
* **Confidence Interval:** 90% CI required for "Revenue at Risk" estimation.

### A/B Testing Strategy

* **Control Group:** Accounts managed via standard reactive processes.
* **Treatment Group:** Accounts managed using "Voice of the Client" alerts and SuccessPlays.
* **Duration:** 6 months (due to B2B renewal cycles).
* **Success Criteria:** Treatment group shows **5% lower churn rate** and **10% higher upsell rate**.

> **KPI TIERS**
>
> * **Strategic (Quarterly):** Net Promoter Score (Target: 45+), Churn Rate (Target: <10%).
> * **Operational (Monthly):** Detractor Contact Rate (Target: 100% within 48h), Feedback Categorization Accuracy (Target: >85%).
> * **Model Performance (Weekly):** Drift Detect (Target: <0.1 drift score), F1-Score Stability.
> * **Business ROI (Annual):** Retained ARR attributed to "Save" interventions (Target: $5M).

---

## Section 9: Value & Risk

### What is the size of the problem?

- $25.2M ARR lost annually from 15% enterprise churn on $168M portfolio
* 68% of churned enterprise accounts cite undetected "strategic misalignment" as primary reason

### What is the baseline?

Currently, churn is predicted by "gut feel" or lagging indicators (cancellation notice). Retention is 90%. NPS score of 42 (slightly above B2B median of 38 but masking silent exodus)

### What is the uplift/savings?

| Scenario | Churn Reduction | ARR Saved Year 1 | Implementation Cost | Net ROI |
|----------|-----------------|------------------|---------------------|---------|
| Conservative | 3% (15% → 12%) | $5.0M | $565K | 786% |
| Expected | 5% (15% → 10%) | $8.4M | $565K | 1,388% |
| Optimistic | 7% (15% → 8%) | $11.8M | $565K | 1,988% |

*Note: ROI calculations include direct ARR saved + CAC avoidance ($4.85M) + referral pipeline value ($1.68M)*

### What are the risks?

* **Technical:** **Concept Drift**. Client language changes (e.g., new product launch changes terminology).
* **Business:** **Alert Fatigue**. If the model cries "wolf" too often, AMs will ignore it.
* **Regulatory:** **GDPR/Privacy**. Accidentally processing PII in the cloud.
* **Reputational:** **Bias**. Model consistently flagging specific regions/industries as "risky" unfairly.

### What might these risks block?

Severe alert fatigue could block **user adoption**, rendering the tool useless.

> **RESPONSIBLE AI CHECKLIST**
>
> * **Transparency:** All predictions include **SHAP plots** to explain *why* an account is at risk.
> * **Privacy:** **Presidio** PII scrubbing is mandatory before storage. Role-based access control (RBAC) for raw text.
> * **Fairness:** Bias testing across Industry Verticals and Regions to ensure error rates are consistent.
> * **Human-in-the-Loop:** No automated emails sent to "Critical" risk accounts; human review required.
> * **Monitoring:** **Evidently AI** dashboard monitors data drift and model performance daily.

---

## Section 10: Theoretical Foundations of the AI Core

### Domain-Specific Frameworks

- **The Signal Triad (Direct, Indirect, Inferred):** Framework for combining explicit feedback (NPS) with implicit signals (Usage) and financial outcomes (Churn).
* **Network Theory:** Modeling B2B accounts as graphs (Decision Makers vs. Influencers) rather than flat lists to understand sentiment contagion.
* **Customer Lifetime Value (CLV) Models:** Probabilistic models (BG/NBD) adapted for contractual B2B settings.
* **Sentiment Analysis Theory**: Valence-Arousal-Dominance (VAD) framework extended for B2B context where professional tone suppresses emotional expression; models calibrated to detect implicit criticism through linguistic markers (hedges, contrastive conjunctions) rather than explicit negative words  
* **Topic Modeling Theory**: Probabilistic Latent Semantic Analysis (pLSA) enhanced with BERT embeddings to overcome vocabulary mismatch between client language ("strategic partnership") and business taxonomies ("account management")  
* **Causal Inference Theory**: Potential outcomes framework (Rubin Causal Model) applied to isolate treatment effect of specific interventions (e.g., executive engagement) from confounding factors (account size, industry) using propensity score matching  
* **Multi-Stakeholder Decision Theory**: B2B buying committees modeled as social choice problem where account health represents aggregation of heterogeneous preferences weighted by stakeholder influence scores derived from org chart analysis  

---

## Section 11: Data Architecture & Engineering

### 11.1 Mandatory Data Variables: The Skeleton

| Variable Category | Description | Examples |
| :--- | :--- | :--- |
| **Identifiers** | Keys to link entities | `account_id`, `user_id`, `survey_id` |
| **Temporal** | Time of feedback/action | `response_date`, `contract_start_date`, `renewal_date` |
| **Behavioral** | Implicit signals | `login_frequency`, `feature_usage_count`, `support_ticket_count` |
| **Contextual** | Metadata | `contract_value`, `industry`, `account_tier`, `respondent_role` |
| **Outcomes** | Targets | `nps_score`, `churn_label` (0/1), `upsell_amount` |

### 11.2 Data Transformation and Engineering

- **Aggregations**:  
  * Daily: Support ticket volume rolling 7/30/90-day windows  
  * Quarterly: NPS trend (current quarter - prior quarter)  
  * Account-level: Sentiment score weighted by stakeholder influence (Decision Maker × 0.6 + User × 0.3 + Influencer × 0.1)  
* **Lag Features**:  
  * NPS_lag1: Previous quarter's NPS score  
  * Sentiment_delta_90d: Current sentiment - sentiment 90 days ago  
  * Usage_decline_60d: (Current MAU - MAU 60d ago) / MAU 60d ago  
* **Ratios**:  
  * Support_friction_index: support_ticket_volume / product_usage_score  
  * Revenue_per_engagement: contract_value_arr / (portal_logins + training_completions)  
* **Embeddings**:  
  * Feedback embeddings: Sentence-BERT (all-mpnet-base-v2) for semantic similarity clustering  
  * Account embeddings: Node2Vec on relationship graph (accounts connected by industry/size)  
* **Categorical Encoding**:  
  * Target encoding for industry_vertical (mean churn rate per industry)  
  * Embedding layers for account_tier in neural network components  
* **Text Preprocessing:** Lowercasing, removing HTML tags, PII tokenization (`<NAME>`).

### 11.3 Online Dataset Search and Analysis

| Dataset Name | Domain | URL/Source | Relevance to AI Core |
| :--- | :--- | :--- | :--- |
| B2B Customer Feedback Corpus | Enterprise SaaS | Kaggle: "B2B NPS Feedback Dataset" | 12K enterprise NPS comments with scores; ideal for pretraining ABSA models on B2B language patterns |
| **CFPB Complaints** | Finance/B2B | [consumerfinance.gov](https://www.consumerfinance.gov/data-research/consumer-complaints/) | Excellent proxy for formal, high-stakes complaint text and categorization. |
| **Olist E-commerce** | Retail/Logistics | [Kaggle (Olist)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) | Contains review text linked to order status (good for "Operational Friction" modeling). |
| **Telco Churn** | Telecom | [Kaggle (IBM)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) | Standard baseline for tabular churn prediction features (Contract, Tenure). |
| **Amazon Reviews** | Retail | [HuggingFace](https://huggingface.co/datasets/amazon_reviews_multi) | Good for pre-training sentiment models (though B2C focused). |

### 11.4 Gap Analysis and Deductive Engineering

*Identified Gaps*  
* Missing "strategic alignment" explicit metric → **Proxy engineered**: (roadmap mention count × negative sentiment) + (competitor consideration mentions)  
* No direct "executive engagement" signal → **Proxy engineered**: Ratio of C-level meeting requests fulfilled / requested + sentiment on "leadership visibility" mentions  
* Incomplete stakeholder influence mapping → **Proxy engineered**: Org chart position inferred from email domain analysis + meeting participation patterns  

*Deductive Engineering Process*  

1. Gap identification via feature importance analysis (SHAP values showing high-impact missing variables)  
2. Proxy design using causal diagrams to ensure no introduction of confounding  
3. Validation against human-labeled subset (n=100 accounts) to confirm proxy accuracy >70%  
4. Gradual replacement with direct measurement as data collection matures (e.g., explicit strategic alignment survey question added in Q3)  

---

## Section 12: Impact & Measurement

### What is the impact?

The AI Core shifts the organization from **Reactive** (fighting fires) to **Proactive** (fire prevention). It quantifies the "Voice of the Client" into dollar terms.

### Where can you see the improvement?

* **Dashboards:** "Strategic Account Intelligence" Dashboard in Tableau/PowerBI.
* **Operational Metrics:** Reduction in "Uncategorized" feedback (Automated tagging).
* **CRM:** "Health Score" field in Salesforce is accurate and predictive.

### Success Criteria

| Metric | Baseline | Target | Measurement Frequency |
| :--- | :--- | :--- | :--- |
| **Gross Revenue Retention** | 88% | 92% | Quarterly |
| **Churn Prediction Accuracy** | 60% (Human) | 80% (AI) | Monthly |
| **Feedback Processing Time** | 5 days | < 1 hour | Real-time |
| **Detractor Contact Rate** | 40% | 95% | Weekly |
