# User Guide: Voice of Customer AI Cheat Sheet

**Quick Reference for Business Users**  
**Version:** 1.0.0 | **Last Updated:** 2026-01-22

---

## 🎯 What Does This AI Core Do?

**Voice of Customer AI** helps you **understand what your customers are really saying across all channels (Reviews, Calls, Surveys) to prioritize product improvements and prevent churn.**

**Key Capabilities:**
- ✅ **Analyze Sentiment at Scale:** Instantly categorize feedback as Positive, Neutral, or Negative.
- ✅ **Extract Specific Aspects:** Know exactly *what* customers like or dislike (e.g., "Price" vs. "Quality").
- ✅ **Detect Emotions:** Identify high-emotion interactions (e.g., Anger, Frustration) for immediate action.
- ✅ **Discover Emerging Topics:** Automatically cluster thousands of reviews into themes like "Shipping Delays" or "Product Defects".

---

## 📊 Accessing the Dashboard

### 1. **Launch the Dashboard**
```bash
# Navigate to the AI Core directory
cd cores/retail_cpg/voice-of-customer

# Start the Streamlit dashboard
streamlit run app/app.py
```

The dashboard will open in your browser at: `http://localhost:8501`

### 2. **Dashboard Overview**

| Tab | Purpose | Key Metrics |
|-----|---------|-------------|
| **📈 Executive Summary** | High-level sentiment and volume trends | `NPS Estimate`, `CSAT Proxy`, `Volume` |
| **🎭 Sentiment & Emotions** | Deep dive into customer feelings | `Sentiment Distribution`, `Emotion Heatmap` |
| **🏷️ Topics & Aspects** | Understand *what* is being discussed | `Top Topics`, `Aspect Sentiment`, `Word Clouds` |
| **🔍 Customer Search** | Deep-dive into individual customer history | `Customer Sentiment Score`, `Key Topics` |
| **🚨 High Urgency** | Triage critical feedback requiring action | `Urgency Score`, `Recommended Actions` |

---

## 🔑 Key Metrics Explained

### **Sentiment Score**
- **What it is:** A measure of customer positivity/negativity.
- **Range:** `-1.0` (Very Negative) to `+1.0` (Very Positive).
- **How to use it:** Monitor the trend line. A sudden drop indicates a potential issue with a new product launch or policy change.
- **Interpretation:**
  - 🟢 **Positive (> 0.3):** Satisfied customers.
  - 🟡 **Neutral (-0.3 to 0.3):** Indifferent or mixed feedback.
  - 🔴 **Negative (< -0.3):** Detractors/Unhappy customers.

### **Urgency Level**
- **What it is:** An AI-derived flag indicating how quickly you need to respond.
- **Derived from:** Negative Sentiment + High Confidence + High-Arousal Emotions (Anger, Fear).
- **Categories:**
  - 🚨 **High:** Critical issues (Safety, Legal, Rage) → Act immediately.
  - ⚠️ **Medium:** Service failures (Shipping delay, Defect) → Act within 24h.
  - ℹ️ **Low:** General feedback or suggestions → Review in weekly batch.

### **Aspect Sentiment**
- **What it is:** Sentiment specific to a feature (e.g., "The **screen** is great, but **battery** sucks").
- **How to use it:** Identify product flaws even in positive reviews.
- **Example:** A product might have 4.5 stars but a negative score for the "Battery" aspect.

---

## 🛠️ Common Tasks

### **Task 1: Find Critical Issues (High Urgency)**
1. Navigate to the **🚨 High Urgency** tab.
2. Review the list of interactions flagged as **High**.
3. Check the **Recommended Action** column (e.g., "Check Logistics Status", "Escalate to Senior Support").
4. Click **Export Alerts** to share with the CS team.

**Expected Output:** A prioritized list of angry/frustrated customers for immediate recovery.

---

### **Task 2: Identify Product Quality Issues**
1. Navigate to the **🏷️ Topics & Aspects** tab.
2. Look at the **Aspect Sentiment** chart.
3. Find aspects with high **Negative Sentiment** (Red bars).
4. Click on the "Quality" or "Defects" aspect to see related review snippets.

**Insight Example:** "We see a spike in 'Zipper' complaints for the new Jacket SKU."

---

### **Task 3: Analyze a Specific Customer**
1. Navigate to the **🔍 Customer Search** tab.
2. Enter the `Customer ID` or `Email`.
3. Review their **Sentiment Profile**:
   - **History:** Have they always been unhappy, or is this new?
   - **Topics:** What do they usually complain/praise about?
   - **Emotions:** Are they frustrated or just disappointed?

**Action:** Tailor your support response based on their history.

---

### **Task 4: Compare Performance by Channel**
1. Navigate to the **📈 Executive Summary** tab.
2. Use the filters to select `Channel: Call Center` vs `Channel: Web Reviews`.
3. Compare the **Sentiment Score**.

**Insight Example:** "Call center sentiment is 20% lower than web reviews, indicating complex unresolved issues are driving calls."

---

## 🚨 Alerts & Notifications

### **When to Act:**

| Alert Type | Trigger | Recommended Action | Timeline |
|------------|---------|-------------------|----------|
| 🚨 **Critical Alert** | Urgency = High | Personal outreach / Crisis Mgmt | **Immediate** |
| 📉 **Sentiment Drift** | Avg Sentiment drops > 10% | Investigate recent product changes | **24 hours** |
| 🆕 **New Topic** | New topic cluster > 50 reviews | Review for emerging trends/viral issues | **Weekly** |
| 🔴 **Negative Aspect** | Aspect sentiment < -0.5 | Notify Product/R&D team | **Weekly** |

---

## 📖 Glossary

| Term | Definition |
|------|------------|
| **ABSA** | **Aspect-Based Sentiment Analysis**. Breaking down feedback into features (e.g., Price, Quality). |
| **MER** | **Multimodal Emotion Recognition**. Detecting emotion from text (and potentially audio/video). |
| **NPS** | **Net Promoter Score**. A standard loyalty metric (-100 to +100). |
| **Sentiment** | The polarity of the feedback (Positive vs. Negative). |
| **Topic** | A theme discovered by the AI (e.g., "Shipping Delays", "Sizing Issues"). |
| **Urgency** | Priority level assigned to an interaction based on risk. |

---

## ❓ Frequently Asked Questions

### **Q1: How accurate is the sentiment analysis?**
**A:** The model uses advanced Transformers (DeBERTa) and typically achieves 85-90% accuracy. However, sarcasm and very subtle nuance can sometimes be missed.

### **Q2: Why is a review marked "High Urgency"?**
**A:** The system detected strong negative emotions (like Anger) combined with high confidence in the negative sentiment. Check the "Emotions" column to see details.

### **Q3: Can I define my own Topics?**
**A:** The system discovers topics automatically (Unsupervised Learning). However, the Data Science team can seed specific topics if needed. Contact support.

### **Q4: How often does the dashboard update?**
**A:** Real-time for API calls; Batch data updates `[FREQUENCY - e.g., Daily at 6 AM UTC]`.

### **Q5: Can I download the data?**
**A:** Yes, look for the **Download CSV** button on the top right of any data table in the dashboard.

### **Q6: Who do I contact for support?**
**A:** 
- **Technical Issues:** `support@aicore.example.com`
- **Data Questions:** `data-science@aicore.example.com`

---

## 🎓 Best Practices

### ✅ **DO:**
- **Triangulate:** Look at Sentiment AND Topics together. A drop in sentiment is more useful when you know it's caused by "Shipping".
- **Act on High Urgency:** These represent your highest churn risk.
- **Share Insights:** Use the charts to back up product decisions with data, not just anecdotes.

### ❌ **DON'T:**
- **Ignore "Neutral":** Neutral reviews often contain valuable feature requests or mild friction points.
- **Treat all Negatives the same:** A "Shipping Delay" (Operational) is different from "Bad Quality" (Product). Use the **Aspects** view to differentiate.

---

## 📞 Support & Resources

| Resource | Link/Contact |
|----------|--------------|
| **Technical Documentation** | `technical_design.md` |
| **API Documentation** | `api_specification.md` |
| **Operational Guide** | `runbook.md` |
| **Support Email** | `support@aicore.example.com` |
| **Slack Channel** | `#ai-core-voc` |

---

## 🚀 Quick Start Checklist

- [ ] Launch the dashboard (`streamlit run app/app.py`)
- [ ] Check **Executive Summary** for overall health.
- [ ] Go to **High Urgency** and clear any critical alerts.
- [ ] Review **Topics** to see what people are talking about today.
- [ ] Export findings to share with your team.

---

**Need Help?** Contact `support@aicore.example.com`