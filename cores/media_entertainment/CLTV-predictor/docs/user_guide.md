# User Guide: CLTV Predictor Cheat Sheet

**Quick Reference for Business Users**  
**Version:** 1.2.0 | **Last Updated:** 2026-01-22

---

## 🎯 What Does This AI Core Do?

**CLTV Predictor** transitions your strategy from "subscriber growth" to **profitability**. It estimates each user's net monetary contribution across their entire relationship with the brand, quantifying long-term revenue potential to drive content, marketing, and pricing decisions.

**Key Capabilities:**

- ✅ **Forecast Multi-Horizon Revenue:** Predict exact dollar values for the next 12, 24, and 36 months (MAPE <19%).
- ✅ **Revenue Mix Decomposition:** Breakdown of value into Subscription fees (SVOD), Advertising impressions (AVOD), and Transactions (TVOD).
- ✅ **Content Value Correlation:** Identify which genres and titles drive the highest predicted lifetime value.
- ✅ **Whale & Tier Identification:** Automatically segment users into value quintiles (Q1-Q5) to prioritize high-value retention.
- ✅ **Monetization Potential:** Identify "Value Multipliers" like CTV usage or social engagement that signal upsell opportunities.

---

## 📊 Accessing the Dashboard

### 1. **Launch the Dashboard**

```bash
# Navigate to the AI Core directory
cd media_entertainment/CLTV-predictor

# Start the Streamlit dashboard
streamlit run app/app.py
```

The dashboard will open in your browser at: `http://localhost:8501`

### 2. **Dashboard Overview**

| Tab | Purpose | Key Metrics |
|-----|---------|-------------|
| **📈 Executive Summary** | High-level portfolio profitability | `Avg CLTV (12/24/36m), Revenue Mix, Whale %` |
| **💎 Value Navigator** | Identify and segment high-value users | `Predicted CLTV, Revenue Tier, Growth Potential` |
| **📊 Cohort Analysis** | Compare ROI by acquisition channel | `LTV:CAC, Channel Variance, Tenure Decay` |
| **🔍 Individual Profiles** | Deep-dive into a subscriber's value data | `Revenue Trajectory, Genre Affinity, Ad Value` |
| **💡 Explainability** | Understand value drivers (Why is this user a Whale?) | `SHAP Value Drivers, Content Impact` |

---

## 🔑 Key Metrics Explained

### **Predicted CLTV (12/24/36m)**

- **What it is:** The total net revenue (SVOD + AVOD + TVOD) expected from a subscriber over the specified timeframe.
- **Range:** $0 to $10,000+ (highly dependent on geography and tier).
- **How to use it:** Use 36-month CLTV for long-term content planning and 12-month CLTV for immediate marketing budget allocation.

### **Revenue Mix (Component Breakdown)**

- **What it is:** The split of a user's total value into SVOD (Subscription), AVOD (Ad-supported), and TVOD (Purchases).
- **Why it matters:** An AVOD user with high engagement may have a higher total CLTV than a low-engagement SVOD "ghost" subscriber.
- **Strategic Use:** Target users with high **Ad-Supported Potential** for specific "Ad-Free" premium upgrades if the predicted margin is higher.

### **Content Affinity Value Score**

- **What it is:** A correlation index showing how much a user's content preference contributes to their total value.
- **Interpretation:**
  - � **High Impact:** Genre preferences (e.g., Live Sports, Exclusive Drama) that correlate with 2x+ higher CLTV.
  - 📉 **Low Impact:** Generic content that drives low retention or monetization.

---

## 🛠️ Common Tasks

### **Task 1: Identify High-Value Targets for Retention/Upsell**

1. Navigate to the **💎 Value Navigator** tab.
2. Set filters:
    - CLTV Quintile: `Q5 (Top 20%)`
    - Growth Rate: `> 2%` (identifying users on an upward trajectory)
3. Sort by **Premium Upgrade Potential**.
4. **Action:** Export this list to trigger a VIP loyalty offer or a premium tier discount.

---

### **Task 2: Optimize Content Production Strategy**

1. Navigate to the **� Cohort Analysis** tab.
2. Group by: **Primary Content Genre**.
3. Review the **CLTV Lift by Genre** chart.
4. **Insight:** Users primarily watching "Sci-Fi Originals" may show 40% higher CLTV than "Reality TV" fans.
5. **Action:** Justify higher production budgets for Sci-Fi based on the long-term revenue multiplier.

---

### **Task 3: Evaluate Acquisition Channel ROI**

1. Navigate to the **📊 Cohort Analysis** tab.
2. Compare: **Wholesale Bundle vs. Direct DTC**.
3. Review: **Average CLTV**.
4. **Insight:** "Wholesale" subscribers may have lower per-month revenue but significantly longer tenure, resulting in higher 36-month CLTV.
5. **Action:** Benchmark against Customer Acquisition Cost (CAC) to adjust channel-specific marketing spend.

---

### **Task 4: Individual Value Deep-Dive**

1. Navigate to the **🔍 Individual Profiles** tab.
2. Enter a `Subscriber ID`.
3. Review the **Future Revenue Trajectory** (12-month forecasted curve).
4. **Action:** If the trajectory is flat but the **Watch-time velocity** is high, offer high-value users a TVOD (Purchase) credit to stimulate transactional revenue.

---

## 🚨 Value Alerts

| Alert Type | Trigger | Business Action | Timeline |
|------------|---------|-----------------|----------|
| � **Whale Alert** | User enters Q5 (Top 5%) Value Rank | Enroll in VIP automated support & early content access | **Immediate** |
| 📈 **Upsell Signal** | AVOD value exceeds SVOD margin | Trigger "Upgrade to Ad-Free" personalized offer | **7 days** |
| 📉 **Decay Warning** | Monthly revenue trajectory drops >15% | Send content re-engagement or "New in Category" alert | **14 days** |
| 🚨 **ROI Risk** | CAC for cohort exceeds predicted 12m CLTV | Pause/Re-evaluate acquisition channel bidding | **Weekly** |

---

## 📖 Glossary

| Term | Definition |
|------|------------|
| **Monetization Profile** | Categorization based on revenue source (e.g., "Ad-Supported High Engagement"). |
| **Revenue growth rate** | Predicted month-over-month increase/decrease in value. |
| **CTV Multiplier** | The value uplift observed in users who consume via Connected TV (avg 1.45x). |
| **Wholesale Awareness** | Analysis of value for users acquired through telco/aggregator bundles. |
| **NPV (Net Present Value)** | The total value of the audience after accounting for the time-value of money. |
| **MAPE** | Mean Absolute Percentage Error—the primary accuracy metric for revenue forecasts. |

---

## ❓ Frequently Asked Questions

### **Q1: Does this predict if a user will leave?**

**A:** No. While retention is a factor, this AI Core focuses on **Monetary Value**. It answers: "How much is this user worth to us over time?" rather than just "Will they stay?"

### **Q2: Why is a customer's CLTV different for two users on the same plan?**

**A:** Because CLTV accounts for **engagement**. A user watching 40 hours of AVOD (Ad-supported) content generates more advertising revenue than a user watching only 5 hours on the same tier.

### **Q3: How do I use this for content greenlighting?**

**A:** Refer to the **Content Value Impact** metrics. This shows the correlation between specific genres/titles and actual revenue lifecycles. Greenlight content that attracts users who stay longer and spend more.

### **Q4: Is ad revenue really predictable?**

**A:** Yes. By analyzing "Watch-time velocity" and ad completion rates (95%+ on CTV), the model forecasts the volume of ad inventory a user will consume.

### **Q5: Can I use this for ad inventory pricing?**

**A:** Absolutely. By identifying high-CLTV segments, you can justify premium CPMs for advertisers wanting to reach high-value, high-engagement audiences.

---

## 🎓 Best Practices

### ✅ **DO:**

- Use **36-month CLTV** for strategic investment and **12-month CLTV** for tactical marketing.
- Factor in the **CTV consumption percentage** when valuing a segment; it's a primary value driver.
- Compare **Organic vs. Paid** channel value to optimize your CAC budgets.
- Regularly check the **Revenue Component Breakdown** to see if shift toward AVOD-heavy tiers is cannibalizing SVOD value.

### ❌ **DON'T:**

- Focus only on the top quintile; Q3-Q4 users often have the highest **Upsell Potential**.
- Ignore **Wholesale** cohorts; while their ARPU (Average Revenue Per User) may be lower, their stability often leads to higher 36-month CLTV.
- Use CLTV for discriminatory pricing (stay compliant with ethical AI guidelines).

---

## 📞 Support & Resources

| Resource | Link/Contact |
|----------|--------------|
| **Core Documentation** | `docs/overview.md`, `docs/technical_design.md` |
| **Business Impact KPI Guide**| `docs/overview.md#business-impact-kpis` |
| **Feature Store Documentation**| `feature_repo/README.md` |
| **Monetization Slack** | `#revenue-science-core` |

---

## 🚀 Quick Start Checklist

- [ ] Check the **Executive Summary** for the current Revenue Mix (SVOD vs. AVOD).
- [ ] Identify which acquisition channel has the highest **36-month CLTV**.
- [ ] List the Top 20% of users with high **Premium Upgrade Value Potential**.
- [ ] Export a value-segmented list for your next Content Marketing campaign.
- [ ] Schedule a monthly "Value Review" to align content investment with revenue trends.

---

**Need Help?** Contact `support@aicore.example.com` or join the `#revenue-science` channel.
