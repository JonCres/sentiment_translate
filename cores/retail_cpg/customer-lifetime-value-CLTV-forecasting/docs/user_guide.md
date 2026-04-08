# User Guide: Customer Lifetime Value CLTV Forecasting Cheat Sheet

**Quick Reference for Business Users**  
**Version:** 1.0.0 | **Last Updated:** 2026-01-21

---

## 🎯 What Does This AI Core Do?

**Customer Lifetime Value CLTV Forecasting** helps you **shift from retrospective analysis to predictive insights** by forecasting exactly how much revenue a customer will generate in the next 12, 24, and 36 months.

**Key Capabilities:**
- ✅ **Hybrid CLV Prediction:** Combines discretionary spend (Non-Contractual) with subscription stability (Contractual) for a total value view.
- ✅ **Churn Probability:** Estimates the likelihood a customer is "Alive" or will churn in the next 30 to 90 days.
- ✅ **Strategic Segmentation:** Categorizes customers into quintiles (Q1-Q5) to optimize marketing spend.
- ✅ **Behavioral Refinement:** Adjusts predictions based on engagement signals (e.g., support tickets, descaling kit purchases).

---

## 📊 Accessing the Dashboard

### 1. **Launch the Dashboard**
```bash
# Navigate to the AI Core directory
cd retail_cpg/customer-lifetime-value-CLTV-forecasting

# Start the Streamlit dashboard
streamlit run app/app.py
```

The dashboard will open in your browser at: `http://localhost:8501`

### 2. **Dashboard Overview**

| Tab | Purpose | Key Metrics |
|-----|---------|-------------|
| **📈 Executive Summary** | High-level KPIs and trends | Sum of CLV Predictions, Avg CLV vs. Baseline, ROI on Retention |
| **🎯 Risk Radar** | Identify at-risk high-value customers | P(Alive), P(Churn 90-day), CLV Quintile |
| **📊 Cohort Analysis** | Compare performance by acquisition channel | CLV by Channel, Month, and Geography |
| **🔍 Customer Search** | Deep-dive into individual profiles | Expected Transactions, Expected Value per Transaction |
| **💡 Explainability** | Understand behavior drivers | Covariate Impact (e.g., Machine Plan type, Billing Interval) |

---

## 🔑 Key Metrics Explained

### **P(Alive)**
- **What it is:** The probability that a customer is still active and likely to purchase again.
- **Range:** 0.0 - 1.0 (where 1.0 is 100% certainty they are active).
- **How to use it:** Identify customers with falling scores (e.g., dropping from 0.9 to 0.6) for immediate re-engagement.
- **Interpretation:**
  - 🟢 **High (0.8-1.0):** Stable, loyal customers.
  - 🟡 **Warning (0.4-0.7):** Entering the "danger zone"—at risk of becoming dormant.
  - 🔴 **Low (<0.4):** Likely churned or already dormant.

### **CLTV (12/24/36 Month)**
- **What it is:** The predicted net revenue expected from the customer over the specified horizon.
- **Range:** Varies by customer (Target Avg: $425).
- **How to use it:** Determine eligibility for high-cost retention offers (e.g., free machine upgrades).
- **Interpretation:** Use the **95% Confidence Interval** to understand the range of potential value ($312 - $1,678 range).

### **CLTV Segment (Q1-Q5)**
- **What it is:** A categorization of your customer base into five equal groups based on predicted value.
- **Categories:**
  - 💎 **Q5 (High):** The top 20% who often generate ~75% of total revenue.
  - 🐟 **Q3-Q4 (Medium):** Stable growth opportunities.
  - 🦐 **Q1-Q2 (Low):** Discretionary shoppers with low predicted upside.

---

## 🛠️ Common Tasks

### **Task 1: Optimize Marketing Budget Allocation**
1. Navigate to the **📊 Cohort Analysis** tab.
2. Select **Acquisition Channel** as the primary dimension.
3. Compare the **CAC:CLV Ratio** across channels (e.g., Paid Search vs. Organic).
4. **Action:** Shift budget from channels with 1:2.5 ratios to those demonstrating 1:3.5+ predicted efficiency.

---

### **Task 2: Decide on "Free Upgrade" Eligibility (The Alice Scenario)**
1. Navigate to the **🔍 Customer Search** tab and enter the Customer ID.
2. Review the **Total Hybrid CLV** (Stream A + Stream B).
3. Check the **Refinement Insights**:
   - Does the customer have a descaling purchase? (Boosts P(Alive)).
   - Is there a pending support ticket for "Leaking"? (Increases Churn Risk).
4. **Action:** If Predicted Hybrid CLV > [Hardware Cost + Margin], flag as "Eligible for Free Upgrade."

---

### **Task 3: Prioritize At-Risk High-Value Customers**
1. Navigate to the **🎯 Risk Radar** tab.
2. Set filters:
   - CLTV Segment: `Q5_High`
   - P(Churn 30-day): `> 0.20`
3. Review the **Expected Transactions 12mo** column.
4. **Action:** Hand off the "Top 50" list to the Customer Success/VIP team for personal outreach.

---

### **Task 4: Export Data for Retention Campaigns**
1. Apply filters for **Q4 (Medium-High)** customers with **P(Alive) < 0.5**.
2. Click the **Download Filtered Data** button.
3. Use the `clv_12mo` and `acquisition_channel` columns to tailor your email/push notification discount thresholds.

---

## 🚨 Alerts & Notifications

### **When to Act:**

| Alert Type | Trigger | Recommended Action | Timeline |
|------------|---------|-------------------|----------|
| 🚨 **Critical** | Q5 Customer + P(Churn) > 0.30 | Personal outreach from account manager | **24-48 hours** |
| ⚠️ **High** | Q4/Q5 Customer + Support Ticket "Leak" | Expedited support + retention discount | **7 days** |
| ℹ️ **Medium** | Q3 Customer + Drop in App Sessions | Re-engagement campaign (personalized content) | **14 days** |
| ✅ **Low** | Q1/Q2 Customer | Standard automated nurture flow | **Monthly** |

---

## 📖 Glossary

| Term | Definition |
|------|------------|
| **BG/NBD** | Beta-Geometric/Negative Binomial Distribution - used to predict transaction frequency. |
| **Gamma-Gamma** | Model used to estimate the average monetary value of future transactions. |
| **sBG / Weibull AFT** | Survival models used for contractual/subscription churn forecasting. |
| **Hybrid CLV** | The sum of discretionary spend predictions and subscription contract value. |
| **MAPE** | Mean Absolute Percentage Error - a measure of how accurate our CLV predictions are. |
| **Covariates** | Additional factors (like device type or plan tier) that "shift" the prediction. |

---

## ❓ Frequently Asked Questions

### **Q1: How often is the data updated?**
**A:** Customer-level CLV and risk scores are updated **Weekly**. Cohort distributions and distribution curves are updated **Monthly**.

### **Q2: Why do we separate Contractual and Non-Contractual data?**
**A:** Treating them as one lump sum underestimates the stability of subscription cash flow or overestimates the loyalty of discretionary boutique shoppers.

### **Q3: Can I trust the predictions for a new customer?**
**A:** The model requires a **minimum of 6-12 months** of history for high accuracy. New customers will have wider confidence intervals until more data is collected.

### **Q4: What is "Pareto Efficiency" in the dashboard?**
**A:** It tracks if the Top 20% of your customers are generating the target 75% of revenue. If this concentration drops, it indicates your high-value segment is weakening.

---

## 🎓 Best Practices

### ✅ **DO:**
- Use **Confidence Intervals** when setting break-even points for acquisition costs.
- Combine **Churn Probability** with **Monetary Value** to prioritize retention (don't save everyone).
- Look for behavioral sequences (like buying a descaling kit) signal hardware commitment.

### ❌ **DON'T:**
- Use predictions as the *only* factor for discounting high-value, stable customers.
- Ignore the "Stream A" (Non-Contractual) upside just because someone has a "Stream B" subscription.
- Rely on CLV for customers with less than 6 months of transaction history.

---

## 🚀 Quick Start Checklist

- [ ] Launch the dashboard (`streamlit run app/app.py`)
- [ ] Check the **Segment Concentration** metric in the Executive Summary.
- [ ] Identify the top 3 acquisition channels by **Predicted CLV ROI**.
- [ ] Filter the **Risk Radar** for at-risk Q5 customers.
- [ ] Export your first campaign list for the marketing team.

---

**Need Help?** Contact `support@aicore.example.com` or visit our [Training Portal](https://training.aicore.example.com)

**Feedback?** We're always improving! Share suggestions at `feedback@aicore.example.com`
