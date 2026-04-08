# User Guide: Predictive CLTV Insights Cheat Sheet

**Quick Reference for Business Users**  
**Version:** 1.0.2 | **Last Updated:** 2026-01-22

---

## 🎯 What Does This AI Core Do?

**Predictive CLTV Insights** helps you **forecast the long-term monetary value of your subscriber base**.
While traditional churn models predict *duration* (how long a user stays), this core predicts *magnitude* (how much they spend). This distinction allows you to:

- **Isolate Monetary Value:** Identify "Whales" (High-ARPU users) independent of their churn risk.
- **Prevent "Binge-and-Burnout":** Detect users who consume content rapidly and are at risk of immediate cancellation (The "Content Cliff").
- **Optimize CAC:** Set precise break-even points for customer acquisition costs based on predicted LTV.

**Key Capabilities:**
- ✅ **Forecast CLTV:** Predict 12, 24, and 36-month value using **BG/NBD** (Frequency) and **Gamma-Gamma** (Monetary).
- ✅ **Hybrid Churn Modeling:** Combine **Survival Analysis** (Contractual) with **Probabilistic Models** (Non-Contractual).
- ✅ **Behavioral Refinement:** Use **XGBoost** to refine predictions based on engagement signals (e.g., binge velocity, QoE).
- ✅ **AI-Powered Explainability:** Generate natural language strategic insights using Small Language Models (SLMs).

---

## 📊 Accessing the Dashboard

### 1. **Launch the Dashboard**
```bash
# Navigate to the AI Core directory
cd media_entertainment/predictive-CLTV-insights

# Start the Streamlit dashboard
streamlit run app/app.py
```

The dashboard will open in your browser at: `http://localhost:8501`

### 2. **Dashboard Overview**

| Tab | Purpose | Key Metrics |
|-----|---------|-------------|
| **📈 Executive Summary** | High-level KPIs and trends | `Revenue Projection`, `CAC Efficiency`, `Whale Concentration` |
| **🎯 Risk Radar** | Identify at-risk entities | `P(Churn 30d)`, `P(Alive)`, `Intervention Priority` |
| **🔍 Customer Search** | Deep-dive into individual profiles | `CLTV Prediction`, `AI Strategic Insight`, `Binge Velocity` |
| **📊 Cohort Analysis** | Compare performance by segment | `Acquisition Month`, `CLTV Quintile` |
| **💡 Explainability** | Understand model predictions | `Model Parameters (SLM)`, `Feature Importance` |

---

## 🔑 Key Metrics Explained

### **1. Churn Probability (`p_churn_30day`)**
- **What it is:** The likelihood a customer will stop transacting or cancel in the next 30 days.
- **Derivation:** Combines probabilistic "dropout" risk (BG/NBD) with behavioral burnout signals (XGBoost).
- **Interpretation:**
  - 🟢 **Low (0-20%):** Healthy, engaged.
  - 🟡 **Medium (20-60%):** "Silent Attrition" risk. Monitor.
  - 🔴 **High (60-100%):** Immediate burnout or cancellation risk (e.g., post-binge).

### **2. Customer Lifetime Value (`clv_12mo`)**
- **What it is:** The Net Present Value (NPV) of future profits from a customer over the next 12 months.
- **Models:** Uses **Gamma-Gamma** for monetary value and **BG/NBD** for frequency.
- **Action:**
  - 💎 **Whales (Q5_Elite):** Protect at all costs. (Top 20% often drive ~70% revenue).
  - 🐟 **High Potential (Q4):** Upsell target.
  - 🦐 **Low Value (Q1-Q2):** Automate support.

### **3. Probability Alive (`p_alive`)**
- **What it is:** For non-contractual settings (TVOD/Gaming), the probability a user is still "active" despite a pause in transactions.
- **Insight:** A user with high `recency` but low `p_alive` has likely defected to a competitor.

---

## 🛠️ Common Tasks

### **Task 1: Find "At-Risk Whales"**
1. Navigate to the **🎯 Risk Radar** tab.
2. Set filters:
   - **CLTV Segment:** `Q5_Elite`
   - **Churn Risk (30d):** `> 0.50`
3. This isolates your most valuable customers who are about to leave.
4. **Action:** Export list and trigger "VIP Concierge" outreach.

### **Task 2: Detect "Binge-and-Burnout" Candidates**
*Scenario: Users who binge a hit show and cancel immediately.*
1. Navigate to **🔍 Customer Search**.
2. Look for customers with **High Intensity** (e.g., completed a season in <3 days) and **High Churn Risk**.
3. **Action:** Offer "Bridge Content" recommendations or an Annual Plan upgrade discount before their subscription expires.

### **Task 3: Analyze AI Interpretations**
1. Navigate to **🔍 Customer Search**.
2. Select a customer.
3. Look for the **AI Strategic Summary** card.
4. **Example Insight:** *"Customer 452 is a high-value 'Whale' with declining frequency. While monetary value remains high, the gap in recent logins suggests potential burnout. Recommend immediate engagement."*

---

## 🚨 Alerts & Notifications

### **When to Act:**

| Alert Type | Trigger | Recommended Action | Timeline |
|------------|---------|-------------------|----------|
| 🚨 **Critical Whale** | Segment=Elite AND Risk > 60% | Personal phone call / VIP Offer | **24 hours** |
| ⚠️ **High Potential** | Segment=Q4 AND Risk > 50% | Targeted Email Campaign | **3 days** |
| ℹ️ **Silent Churn** | P(Alive) < 30% | Win-back Campaign (Aggressive Discount) | **7 days** |

---

## 📖 Glossary

| Term | Definition |
|------|------------|
| **Entity** | The unit of analysis (Customer/Subscriber/Player). |
| **BG/NBD** | Model predicting *frequency* and *dropout* (Recency/Frequency). |
| **Gamma-Gamma** | Model predicting *average monetary value*. |
| **CAC** | Customer Acquisition Cost. |
| **SLM** | Small Language Model (e.g., Phi-4) used for text generation. |
| **Behavioral Refinement** | Using XGBoost to adjust predictions based on usage logs (e.g., binge velocity). |
| **QoE** | Quality of Experience (e.g., buffering, lag) - a leading indicator of churn. |

---

## ❓ Frequently Asked Questions

### **Q1: Why do we use two models (BG/NBD + XGBoost)?**
**A:** BG/NBD is excellent for long-term trends based on purchase history. XGBoost is better at detecting immediate "shocks" from behavioral data (e.g., a bad customer support experience yesterday). We combine them for maximum accuracy.

### **Q2: What is the source of the "AI Insights"?**
**A:** The system runs a local **Small Language Model (SLM)** to analyze the numerical predictions and generate human-readable text. This ensures data privacy (no data sent to external APIs) while providing high-quality explanations.

### **Q3: What data is required for this to work?**
**A:** At minimum:
1. **Transaction History:** (Date, Amount, ID) - 6-12 months.
2. **Subscription Data:** (Start/End dates, Plan Tier) for contractual models.
3. **Churn Ground Truth:** (Cancellation dates).
4. **Engagement Metrics:** (Optional but recommended for refinement) e.g., watch hours, session count.

### **Q4: Can I trust the "Predicted CLTV"?**
**A:** Check the **Confidence Intervals** (`clv_95_ci_lower/upper`). A narrow range means high confidence. New customers often have wider ranges due to limited data. Targets are typically MAPE < 18%.

---

## 🎓 Best Practices

### ✅ **DO:**
- Focus retention budget on **High CLV** customers (Whales).
- Use **AI Insights** to personalize your communication.
- Check **Cohort Analysis** to see if newer customers are worse quality than older ones.
- Monitor **Acquisition Efficiency** (Target CLV:CAC > 3:1).

### ❌ **DON'T:**
- Spam "Low Value" customers with expensive retention offers.
- Ignore "Silent Churn" (customers who just stop buying without cancelling).
- Discount high-value stable customers (cannibalization).

---

## 📞 Support & Resources

| Resource | Link/Contact |
|----------|--------------|
| **Technical Documentation** | `technical_design.md` |
| **Operational Guide** | `runbook.md` |
| **Support Email** | `support@aicore.example.com` |
