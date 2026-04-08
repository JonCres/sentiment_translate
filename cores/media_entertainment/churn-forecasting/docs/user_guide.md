# User Guide: Churn Forecasting Cheat Sheet

**Quick Reference for Business Users**  
**Version:** 1.0.0 | **Last Updated:** 2026-01-21

---

## 🎯 What Does This AI Core Do?

**Churn Forecasting** helps you **optimize viewer retention by quantifying the probability of churn for every subscriber**, moving beyond binary labels to predict a potential exit 30–90 days before attrition occurs.

**Key Capabilities:**

- ✅ **Temporal Engagement Analysis:** Detect early-warning signs of churn through watch-time velocity and session density decay.
- ✅ **Risk Tiering & Prioritization:** Automatically segment users into Imminent, At-Risk, and Passive categories for targeted action.
- ✅ **XAI Root Cause Analysis:** Leverage SHAP and LIME to understand why a subscriber is disengaging (e.g., content fatigue vs. technical friction).
- ✅ **Early Warning System:** Estimate the likelihood of exit across multiple time horizons (30, 60, 90 days).

---

## 📊 Accessing the Dashboard

### 1. **Launch the Dashboard**

```bash
# Navigate to the AI Core directory
cd media_entertainment/churn-forecasting

# Start the Streamlit dashboard
streamlit run app/app.py --server.port=8501
```

The dashboard will open in your browser at: `http://localhost:8501`

### 2. **Dashboard Overview**

| Tab | Purpose | Key Metrics |
|-----|---------|-------------|
| **📈 Executive Summary** | High-level KPIs and Subscription Health | Churn rate vs. 3.1% Target, Retention Rate, Saved Revenue |
| **🎯 Risk Radar** | Identify and prioritize at-risk subscribers | Churn Score, Engagement Velocity, Intervention Priority |
| **📊 Cohort Analysis** | Monitor subscriber retention trends | Retention Rate, Churn Probability (30/60/90d) |
| **🔍 Subscriber Deep-Dive** | Root cause analysis for individual users | Engagement Decay, QoE Friction, SHAP Indicators |
| **💡 Explainability** | Global drivers of brand-wide attrition | Top Churn Triggers, Model Confidence Intervals |

---

## 🔑 Key Metrics Explained

### **Churn Risk Score**

- **What it is:** The estimated probability that a subscriber will cancel or stop engaging within the next 30 days.
- **Range:** **0-100**, where 100 represents an imminent exit.
- **How to use it:** Focus retention budget on high-engagement subscribers with scores above 75.
- **Interpretation:**
  - 🟢 **Passive (0-33):** Healthy engagement; focus on automated content discovery.
  - 🟡 **At-Risk (34-75):** Significant decay in watch-time; requires re-engagement.
  - 🔴 **Imminent (76-100):** High probability of churn; requires immediate retention offer.

### **Engagement velocity**

- **What it is:** The rate of change in watch-time or session frequency compared to the subscriber's historical average.
- **Range:** **-100% to +100%**
- **How to use it:** Identify "fading" subscribers whose activity is trending downward.
- **Interpretation:**
  - 💎 **Power Users (+20% or more):** Increasing engagement; target with premium features.
  - 🐟 **Stable (-10% to +10%):** Standard subscribers; maintain regular content updates.
  - 🦐 **Fading (-20% or less):** High risk of churn due to declining interest.

### **Intervention Priority**

- **What it is:** A composite ranking based on churn urgency and subscriber engagement history.
- **Categories:**
  - 🚨 **Critical:** Imminent risk + High historical engagement → Personal outreach within 24–48 hours.
  - ⚠️ **High:** At-risk status OR declining QoE signals → Action within 7 days.
  - ℹ️ **Medium:** Early behavioral decay → Automated nurture campaign.
  - ✅ **Low:** High engagement probability → Baseline engagement.

---

## 🛠️ Common Tasks

### **Task 1: Find At-Risk Customers**

1. Navigate to the **🎯 Risk Radar** tab
2. Set filters:
   - **Churn Risk Score:** `> 75`
   - **Subscriber Tier:** `Whales` or `Top 20%`
3. Sort by **Intervention Priority** (Critical first)
4. Export the list for CRM synchronization (Braze/Salesforce)

**Expected Output:** A prioritized list of high-value subscribers requiring immediate intervention to prevent revenue leakage.

---

### **Task 2: Understand Why a Customer is At-Risk**

1. Navigate to the **🔍 Subscriber Deep-Dive** tab
2. Enter the `Subscriber ID` or `Account Email`
3. Review the **Top Risk Drivers** (SHAP Attribution):
   - **Feature:** Behavior identified (e.g., "7d Watch-Time Velocity")
   - **Impact:** % contribution to the risk score
   - **Context:** Comparison to the cohort average

**Example:**

| Feature | Impact | Direction | Interpretation |
|---------|--------|-----------|----------------|
| `engagement_velocity` | 45% | ⬇️ Negative | Watch-time has dropped 60% week-over-week |
| `buffering_events` | 30% | ⬆️ Positive | Recent technical frustration is driving risk |
| `genre_variety` | 25% | ⬇️ Negative | Content fatigue (watching only one series) |

**Action:** If QoE (buffering) is the driver, route to Engineering. If engagement velocity is the driver, trigger a "Content Recommendation" campaign.

---

### **Task 3: Compare Cohort Performance**

1. Navigate to the **📊 Cohort Analysis** tab
2. Select grouping dimension:
   - **Acquisition Month:** Track the retention trends of specific monthly cohorts.
   - **Subscription Plan:** Compare churn rate between SVOD and AVOD segments.
   - **Acquisition Channel:** Review if Telco/Wholesale bundles have higher churn rates than Direct-to-Consumer (DTC).
3. Review the **Retention Curve** to see where the 50% drop-off occurs.

**Insight Example:** "Subscribers acquired via Chrome/Web have a 40% higher churn probability in the first 30 days compared to CTV users."

---

### **Task 4: Export Data for Campaigns**

1. Apply filters in any tab (Risk Score, Segment, etc.)
2. Click **Download Filtered Data** button
3. Open the CSV in Excel or your CRM tool
4. Use the `entity_id` column to target campaigns

**CSV Columns:**

- `subscriber_id`: Unique platform identifier
- `churn_prob_30d`: Predicted likelihood of exit (0-1.0)
- `engagement_velocity`: Current trend in watch-time
- `intervention_priority`: Critical/High/Medium/Low
- `primary_risk_driver`: Behavior-based reason for score (e.g., "QoE Failure")
- `recommended_action`: Suggested channel and offer type

---

## 🚨 Alerts & Notifications

### **When to Act:**

| Alert Type | Trigger | Recommended Action | Timeline |
|------------|---------|-------------------|----------|
| 🚨 **Critical** | Risk >85% + High Engagement | Manual VIP outreach + Premium content credit | **24-48 hours** |
| ⚠️ **High** | Risk >70% + Declining Engagement | Targeted re-engagement offer (30 days free) | **7 days** |
| ℹ️ **Medium** | Risk 40-70% | Automated "Watch Next" personalized email | **14 days** |
| ✅ **Low** | Risk <40% | Maintain baseline nurture and feature updates | **Monthly** |

---

## 📖 Glossary

| Term | Definition |
|------|------------|
| **Subscriber** | The individual user/account being analyzed |
| **Churn Probability** | The estimated risk of a subscriber canceling within a specific window (e.g., 30 days) |
| **Retention Rate** | The percentage of subscribers who remained active over a period |
| **Engagement Decay** | A statistical drop in watch-time velocity or session frequency week-over-week |
| **QoE Friction** | Technical playback failures (buffering, bitrate drops) causing user frustration |
| **SHAP Value** | A mathematical weight showing how much a behavior contributed to the risk score |
| **NRR** | Net Revenue Retention – revenue retained from existing users after churn/downgrades |

---

## ❓ Frequently Asked Questions

### **Q1: How often is the data updated?**

**A:** The intelligence engine refreshes **daily at 06:00 AM UTC**. Predictions incorporate the last 90-180 days of telemetry to calculate velocity and decay signals.

### **Q2: Why does the dashboard mention "Churn Probability"?**

**A:** Churn probability represents the likelihood of an exit within a month. Unlike a simple flag, it tells you the intensity of the risk right now, allowing you to prioritize the most critical cases.

### **Q3: How accurate are the predictions?**

**A:** The current ensemble (XGBoost + CNN-BiLSTM) achieves a **95%+ AUC-ROC** on historical hold-out sets. We prioritize **Recall** (>90%) to ensure we don't miss potential leavers.

### **Q4: What if a subscriber's risk score changes dramatically?**

**A:** Dramatic shifts are usually driven by "Trigger Events":

- Multiple technical failures (Buffering >5% of session)
- Rapid drop in engagement velocity (No sessions in 7 days)
- Finishing a primary "Genre Anchor" (e.g., finished a hit series)

Check the **Subscriber Deep-Dive** to identify the specific trigger.

### **Q5: How do I integrate this with our CRM?**

**A:** Use the **Download CSV** feature to export customer lists. Your IT team can also connect via the API (see `api_specification.md`).

### **Q6: Who do I contact for support?**

**A:**

- **Technical Issues:** `support@aicore.example.com`
- **Business Questions:** `product@aicore.example.com`
- **Feature Requests:** `product-team@aicore.example.com`

---

## 🎓 Best Practices

### ✅ **DO:**

- Review the **Critical** priority dashboard daily to prevent high-value revenue leakage.
- Use **Explainability** to tailor interventions (e.g., offer a free month for technical issues).
- Track **Intervention ROI** to see if your re-engagement offers are effectively extending tenure.
- Monitor **Engagement Velocity** trends to identify broad catalog-wide content fatigue.

### ❌ **DON'T:**

- Perform generic mass-outreach; use risk drivers to hyper-personalize the message.
- Ignore **At-Risk** subscribers—they are in the optimal 14-21 day window for a successful save.
- Treat Telco/Bundle subscribers the same as Direct (DTC) users; check their specific bundle terms.
- Forget that churn can be seasonal (e.g., increases after major series finales).

---

## 📞 Support & Resources

| Resource | Link/Contact |
|----------|--------------|
| **Technical Documentation** | `technical_design.md`, `technical_walkthrough.md` |
| **API Documentation** | `api_specification.md` |
| **Operational Guide** | `runbook.md` |
| **Support Email** | `support@aicore.example.com` |
| **Slack Channel** | `#ai-core-[name]` |
| **Training Videos** | `https://training.aicore.example.com` |

---

## 🚀 Quick Start Checklist

- [ ] Launch the Retention Command Center (`streamlit run app/app.py`)
- [ ] Review the **Executive Summary** for the latest NRR (Net Revenue Retention)
- [ ] Filter the **Risk Radar** for "Whales" with Critical priority
- [ ] Analyze the **Risk Drivers** for the top 5 at-risk accounts
- [ ] Export the "Daily Save List" CSV for marketing execution
- [ ] Schedule a weekly review of the **Retention Trends** with the Content team

---

**Need Help?** Contact `support@aicore.example.com` or visit our [Training Portal](https://training.aicore.example.com)

**Feedback?** We're always improving! Share suggestions at `feedback@aicore.example.com`
