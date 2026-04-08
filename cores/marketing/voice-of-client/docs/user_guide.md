---
title: "User Guide: voice-of-client Cheat Sheet"
description: "Quick reference guide for business users to navigate the dashboard and interpret AI insights."
audience: end-user
doc-type: how-to
last-updated: 2026-02-13
---

# User Guide: voice-of-client Cheat Sheet

**Quick Reference for Business Users**  
**Version:** 1.0.0 | **Last Updated:** 2026-02-13

---

## 🎯 What Does This AI Core Do?

**voice-of-client** helps you transform unstructured B2B feedback into predictive churn risk intelligence.

**Key Capabilities:**
- ✅ Predict customer churn 30-90 days in advance
- ✅ Identify high-value accounts for retention
- ✅ Explain specific risk drivers for each account
- ✅ Recommend personalized intervention strategies

---

## 📊 Accessing the Dashboard

### 1. **Launch the Dashboard**
```bash
# Navigate to the AI Core directory
cd [industry]/[ai_core_name]

# Start the Streamlit dashboard
streamlit run app/app.py --server.port=8501
```

The dashboard will open in your browser at: `http://localhost:8501`

### 2. **Dashboard Overview**

| Tab | Purpose | Key Metrics |
|-----|---------|-------------|
| **📈 Executive Summary** | High-level KPIs and trends | `[e.g., Churn Rate, NRR, ROI]` |
| **🎯 Risk Radar** | Identify at-risk entities | `[e.g., Risk Score, CLTV, Segment]` |
| **📊 Cohort Analysis** | Compare performance by segment | `[e.g., Acquisition Month, Product]` |
| **🔍 Customer Search** | Deep-dive into individual profiles | `[e.g., Top Risk Drivers, Timeline]` |
| **💡 Explainability** | Understand model predictions | `[e.g., SHAP values, Feature Impact]` |

---

## 🔑 Key Metrics Explained

### **[METRIC 1 - e.g., Churn Risk Score]**
- **What it is:** `[Simple explanation]`
- **Range:** `[e.g., 0-100, where 100 = highest risk]`
- **How to use it:** `[Action guidance - e.g., "Prioritize customers with scores >70 for retention campaigns"]`
- **Interpretation:**
  - 🟢 **Low (0-30):** Healthy, engaged customers
  - 🟡 **Medium (31-70):** Monitor for early warning signs
  - 🔴 **High (71-100):** Immediate intervention required

### **[METRIC 2 - e.g., Customer Lifetime Value (CLTV)]**
- **What it is:** `[Simple explanation]`
- **Range:** `[e.g., $0 - $10,000+]`
- **How to use it:** `[Action guidance]`
- **Interpretation:**
  - 💎 **Whales (Top 5%):** VIP treatment, dedicated support
  - 🐟 **High-Value (Top 20%):** Premium retention offers
  - 🦐 **Standard (Bottom 80%):** Automated engagement

### **[METRIC 3 - e.g., Intervention Priority]**
- **What it is:** `[Simple explanation]`
- **Categories:**
  - 🚨 **Critical:** High risk + High CLTV → Act within 48 hours
  - ⚠️ **High:** High risk OR High CLTV → Act within 7 days
  - ℹ️ **Medium:** Moderate risk → Monitor weekly
  - ✅ **Low:** Healthy → Standard engagement

---

## 🛠️ Common Tasks

### **Task 1: Find At-Risk Customers**
1. Navigate to the **🎯 Risk Radar** tab
2. Set filters:
   - Risk Score: `> 70`
   - CLTV: `> $500` (focus on high-value)
3. Sort by **Intervention Priority** (Critical first)
4. Export the list using the **Download CSV** button

**Expected Output:** List of customers requiring immediate retention action

---

### **Task 2: Understand Why a Customer is At-Risk**
1. Navigate to the **🔍 Customer Search** tab
2. Enter the `Customer ID` or `Email`
3. Review the **Top Risk Drivers** section:
   - **Feature:** What behavior is driving the risk
   - **Impact:** How much it contributes to the risk score
   - **Direction:** Is it increasing or decreasing risk?

**Example:**
| Feature | Impact | Direction | Interpretation |
|---------|--------|-----------|----------------|
| `engagement_score` | 42% | ⬇️ Negative | Low engagement is the #1 risk driver |
| `tenure_days` | 28% | ⬆️ Positive | Long tenure is reducing risk |
| `payment_failures` | 18% | ⬇️ Negative | Recent payment issues |

**Action:** Focus retention efforts on re-engagement (e.g., personalized content recommendations)

---

### **Task 3: Compare Cohort Performance**
1. Navigate to the **📊 Cohort Analysis** tab
2. Select grouping dimension:
   - **Acquisition Month:** Compare new vs. old customers
   - **Product Category:** Compare different product lines
   - **Region:** Compare geographic performance
3. Review the heatmap for patterns

**Insight Example:** "Customers acquired in Q4 2025 have 2x higher churn than Q1 2026 cohort"

---

### **Task 4: Export Data for Campaigns**
1. Apply filters in any tab (Risk Score, Segment, etc.)
2. Click **Download Filtered Data** button
3. Open the CSV in Excel or your CRM tool
4. Use the `entity_id` column to target campaigns

**CSV Columns:**
- `entity_id`: Unique customer identifier
- `risk_score`: Churn probability (0-100)
- `cltv_estimate`: Predicted lifetime value
- `intervention_priority`: Critical/High/Medium/Low
- `top_risk_driver`: Primary reason for risk
- `recommended_action`: Suggested intervention

---

## 🚨 Alerts & Notifications

### **When to Act:**

| Alert Type | Trigger | Recommended Action | Timeline |
|------------|---------|-------------------|----------|
| 🚨 **Critical** | Risk >80 + CLTV >$1000 | Personal outreach from account manager | **24-48 hours** |
| ⚠️ **High** | Risk >70 OR CLTV >$500 | Automated retention offer (discount, upgrade) | **7 days** |
| ℹ️ **Medium** | Risk 40-70 | Re-engagement campaign (email, push) | **14 days** |
| ✅ **Low** | Risk <40 | Standard nurture flow | **Monthly** |

---

## 📖 Glossary

| Term | Definition |
|------|------------|
| **Entity** | The unit of analysis (e.g., customer, subscriber, account) |
| **Churn** | When a customer cancels or stops engaging |
| **CLTV** | Customer Lifetime Value - total revenue expected from a customer |
| **Cohort** | Group of customers with a shared characteristic (e.g., signup month) |
| **Feature** | Data point used by the model (e.g., tenure, engagement score) |
| **SHAP Value** | Measure of how much a feature contributes to the prediction |
| **Intervention** | Action taken to prevent churn (e.g., discount, outreach) |
| **NRR** | Net Revenue Retention - revenue retained from existing customers |
| **QoE** | Quality of Experience - technical performance metrics |

---

## ❓ Frequently Asked Questions

### **Q1: How often is the data updated?**
**A:** The dashboard refreshes `[FREQUENCY - e.g., daily at 6 AM UTC]`. Predictions are based on the most recent `[LOOKBACK WINDOW - e.g., 90 days]` of activity.

### **Q2: What does "Confidence Interval" mean?**
**A:** The range where we're 95% confident the true value lies. Example: CLTV = $500 (CI: $450-$550) means we're very confident the actual CLTV is between $450 and $550.

### **Q3: Can I trust the predictions?**
**A:** The model is validated on historical data with `[ACCURACY METRIC - e.g., 94% AUC-ROC]`. However, predictions are probabilistic—use them to prioritize, not as absolute truth.

### **Q4: What if a customer's risk score changes dramatically?**
**A:** This is normal! Risk scores update based on recent behavior. A sudden spike may indicate:
- Recent payment failure
- Drop in engagement
- Technical issues (buffering, crashes)

Check the **Top Risk Drivers** to understand the cause.

### **Q5: How do I integrate this with our CRM?**
**A:** Use the **Download CSV** feature to export customer lists. Your IT team can also connect via the API (see `api_specification.md`).

### **Q6: Who do I contact for support?**
**A:** 
- **Technical Issues:** `[SUPPORT_EMAIL]`
- **Business Questions:** `[PRODUCT_OWNER_EMAIL]`
- **Feature Requests:** `[PRODUCT_TEAM_EMAIL]`

---

## 🎓 Best Practices

### ✅ **DO:**
- Review the dashboard `[FREQUENCY - e.g., weekly]` to identify trends
- Focus on **Critical** and **High** priority customers first
- Use **Explainability** to tailor interventions (don't send generic offers)
- Export data regularly for campaign tracking
- Share insights with your team (use the **Share** button)

### ❌ **DON'T:**
- Ignore **Medium** risk customers—they can escalate quickly
- Rely solely on risk scores—combine with business judgment
- Overwhelm customers with too many interventions
- Forget to track intervention outcomes (measure ROI)

---

## 📞 Support & Resources

| Resource | Link/Contact |
|----------|--------------|
| **Technical Documentation** | `technical_design.md`, `technical_walkthrough.md` |
| **API Documentation** | `api_specification.md` |
| **Operational Guide** | `runbook.md` |
| **Support Email** | `[SUPPORT_EMAIL]` |
| **Slack Channel** | `#ai-core-[name]` |
| **Training Videos** | `[TRAINING_PORTAL_URL]` |

---

## 🚀 Quick Start Checklist

- [ ] Launch the dashboard (`streamlit run app/app.py`)
- [ ] Explore the **Executive Summary** tab
- [ ] Filter for **Critical** priority customers
- [ ] Deep-dive into 2-3 customer profiles
- [ ] Export a CSV for your next retention campaign
- [ ] Share insights with your team
- [ ] Schedule a weekly dashboard review

---

**Need Help?** Contact `[SUPPORT_EMAIL]` or visit our [Training Portal]([URL])

**Feedback?** We're always improving! Share suggestions at `[FEEDBACK_EMAIL]`
