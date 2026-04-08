# User Guide: Customer Survival Analyzer

**Quick Reference for Business Users**  
**Version:** 1.0.0 | **Last Updated:** 2026-01-22

---

## 🎯 What Does This AI Core Do?

**Customer Survival Analyzer** shifts from simple "churn prediction" to **Time-to-Event Modeling**. Instead of just flagging *who* will leave, it predicts *when* they will leave and *why*, enabling you to maximize **Total Subscriber Days**.

**Key Capabilities:**
- ✅ **Survival Probability S(t):** Forecasts the probability of retention for the next 30/60/90/180 days.
- ✅ **Hazard Profiling:** Quantifies instantaneous risk using Hazard Ratios (e.g., "User is 2.3x more likely to leave than average").
- ✅ **Intervention Timing:** Pinpoints the exact 14–21 day window to contact a user for maximum impact.
- ✅ **Tenure Expansion:** Identifying specific content and behaviors that extend median subscriber lifetime.

---

## 📊 Accessing the Dashboard

### 1. **Launch the Dashboard**

```bash
# Navigate to the AI Core directory
cd cores/media_entertainment/customer-survival-analyzer

# Start the Streamlit dashboard
streamlit run app/app.py
```

The dashboard will open in your browser at: `http://localhost:8501`

### 2. **Dashboard Overview**

| Tab | Purpose | Key Metrics |
|-----|---------|-------------|
| **📈 Executive Summary** | High-level portfolio health | `Median Subscriber Lifetime`, `Net ROI`, `Active Subscriber Cohorts` |
| **🎯 Risk Radar** | Identify immediate threats | `Hazard Ratio`, `Risk Segment`, `Probability S(30d)` |
| **📊 Cohort Analysis** | Compare survival curves | `Survival by Acquisition Channel`, `Plan Tier Longevity` |
| **🔍 Customer Search** | Deep-dive into individual timelines | `Individual Hazard Function h(t)`, `Intervention Window` |
| **💡 Explainability** | Understand risk drivers | `SHAP Values`, `Top Risk Factors` (e.g. Buffering, Content Gap) |

---

## 🔑 Key Metrics Explained

### **1. Hazard Ratio (Relative Risk)**
- **What it is:** A multiplier indicating a subscriber's risk level compared to the baseline average.
- **Interpretation:**
  - 🟢 **< 0.8:** Low Risk (Likely to stay longer than average).
  - 🟡 **0.8 - 1.2:** Average Risk.
  - 🔴 **> 1.5:** High Risk (50% more likely to terminate).
  - 🚨 **> 2.5:** Critical Risk (Immediate attention required).

### **2. Predicted Median Tenure**
- **What it is:** The predicted total number of days a subscriber will stay active (the point where their Survival Probability S(t) drops to 50%).
- **How to use it:** Identify "Whales" (Long Tenure) vs. "Tourists" (Short Tenure) early in their lifecycle to adjust acquisition spend (CAC) targets.

### **3. Survival Probability S(30d / 90d)**
- **What it is:** The probability (0% to 100%) that the subscriber will still be active 30 or 90 days from now.
- **Action:** If S(30d) drops below 70%, trigger a **"Pre-Emptive Intervention"**.

---

## 🛠️ Common Tasks

### **Task 1: Identify "Silent Churn" Risk**
*Focus on subscribers who haven't cancelled yet but show high "Hazard" signals.*

1. Navigate to **🎯 Risk Radar**.
2. Filter by:
   - **Hazard Ratio:** `> 1.5`
   - **S(30d):** `< 0.70`
   - **Status:** `Active`
3. Sort by **CLTV (Customer Lifetime Value)** descending.
4. **Action:** Export list for the "High-Value Rescue" email campaign.

### **Task 2: Diagnose Why a VIP is Leaving**
1. Go to **🔍 Customer Search**.
2. Enter the Subscriber ID.
3. Look at **Top Risk Factors**. Common drivers:
   - `avg_buffering_ratio_7d` (Technical Friction)
   - `content_catalog_overlap` (Nothing left to watch)
   - `payment_failed_last_cycle` (Transactional risk)
4. **Action:** If technical, alert Support. If content-related, send a "Hidden Gems" recommendation.

### **Task 3: Optimize Retention Spend**
1. Go to **📈 Executive Summary**.
2. Check **Optimal Intervention Window**.
3. **Insight:** The model might recommend contacting users "18 days before predicted termination."
4. **Action:** Schedule marketing automation to fire at exactly `Predicted_End_Date - 18 days`.

---

## 🚨 Alerts & Risk Segmentation

The system automatically categorizes users into daily risk segments:

| Segment | Definition | Recommended Action |
|---------|------------|--------------------|
| **Very High Risk** | Top 10% Hazard Ratio (> 2.5x) | **Personal Outreach:** Dedicated account manager call or 50% off offer. |
| **High Risk** | Top 10-25% Hazard Ratio | **Automated Winback:** "We miss you" campaign with content highlight. |
| **Medium Risk** | 25-50% Percentile | **Nurture:** Standard newsletter and new release alerts. |
| **Safe / Loyal** | Bottom 50% Hazard Ratio | **Upsell:** Recommend family plan or annual billing upgrade. |

---

## 📖 Glossary

| Term | Definition |
|------|------------|
| **Censored Data** | Data points where the event (cancellation) has *not* happened yet. These are active subscribers. |
| **Hazard Function h(t)** | The instantaneous rate of churn at a specific point in time t. |
| **Survival Function S(t)** | The probability of a subscriber "surviving" (staying active) past time t. |
| **Intervention Window** | The specific date range where a retention offer has the highest ROI (typically 14-21 days before risk peaks). |
| **C-Index** | Concordance Index. A measure of how well the model predicts the *order* of churn events (Target > 0.78). |

---

## ❓ Frequently Asked Questions

### **Q1: Why do we use "Survival Analysis" instead of standard Churn Classification?**
**A:** Standard classification only says *if* someone will churn (Binary). Survival analysis tells us *when* (Temporal), allowing us to prioritize interventions for next week vs. next month.

### **Q2: What is the "Observation Window"?**
**A:** The specific timeframe used for training the model. We typically use a 180-day lookback for feature engineering and project risk out 30/60/90 days.

### **Q3: How often are the predictions updated?**
**A:**
- **Individual Scores:** Updated **Daily** based on yesterday's viewing and payment behavior.
- **Model Training:** Retrained **Weekly** (Random Survival Forests) or **Quarterly** (DeepSurv) to capture new content trends.

### **Q4: A user has a high Risk Score but just renewed. Why?**
**A:** The model may detect "silent churn" behaviors (e.g., stopped watching midway through a series, high buffering). They might be planning to cancel *before* the next billing cycle. Trust the **Hazard Ratio**.

---

## 📞 Support & Resources

| Resource | Link/Contact |
|----------|--------------|
| **Technical Design** | `docs/technical_design.md` |
| **Model Specs** | `docs/technical_walkthrough.md` |
| **API Docs** | `docs/api_specification.md` |
| **Support Email** | `support@aicore.media` |

---
**Need Help?** Contact the MLOps team at `mlops@aicore.media`.