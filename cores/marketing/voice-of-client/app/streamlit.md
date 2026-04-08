# APPLICATION CONTEXT: STRATEGIC ACCOUNT INTELLIGENCE DASHBOARD

## 1. Purpose
This Streamlit application serves as the "Cockpit" for Account Managers (AMs) and VPs of Customer Success. It visualizes the outputs of the `Voice of the Client` AI Core.

**Target Audience:** Non-technical business users.
**Design Philosophy:** "Don't just show the score; show the *why* and the *what now*."

## 2. Mandatory Views & Components

### A. Account Deep Dive (Detail Page)
*   **Header:** Client Name, Tier, Health Score.
*   **Stakeholder Alignment Map:** A visual comparison of "Decision Maker" sentiment vs. "End User" sentiment.
*   **Aspect Sentiment Radar:** Radar chart showing sentiment across 8 dimensions (e.g., Billing, Support, Product, Roadmap).
*   **Trend Analysis:** Line chart overlaying NPS Score vs. Support Ticket Volume over time.

### B. The "Why" (Explainability)
*   **SHAP Force Plot:** Visualizing the top positive/negative contributors to the current Churn Score.
*   **Narrative Insight:** Display the SLM-generated text summary (e.g., *"Risk is driven by negative sentiment on Pricing (-0.8) despite high Product Usage."*).

### C. Action Center
*   **Recommendations:** List of "Next Best Actions" (e.g., "Schedule Executive QBR").
*   **Feedback Loop:** Buttons for the AM to Validate/Reject the AI prediction (feeds back into training).

## 3. Data Integration
*   The app reads from the **Feature Store** (curated outputs) and **Model Registry** (metadata).
*   **Latency:** Dashboard must reflect data processed within the last 24 hours (or real-time for single-account inference).

## 4. UX Constraints
*   **Terminology:** Use business terms ("At-Risk Revenue"), not ML terms ("Probability Class 1").
*   **Privacy:** Display `<PERSON>` tokens instead of real names in raw text feedback unless the user has specific RBAC clearance.