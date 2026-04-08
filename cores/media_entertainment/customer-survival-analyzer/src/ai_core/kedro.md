# Pipeline Layer (Kedro)

**Role:** Senior Data Scientist & Data Engineer.

## 1. Modular Architecture
- `data_processing`: Transforms raw snapshots into **Counting Process** format. Handles non-informative censoring.
- `feature_engineering`: 
    - **Temporal Path:** Rolling watch-hours, days-since-last-session.
    - **Diversity Path:** Genre entropy and content repetition indices.
    - **Friction Path:** QoE scores (buffering/crash rates).
- `data_science`: Trains DeepSurv and NMTLR. Evaluates using C-index and Brier scores.
- `explainability_insights`: (See Section 2).

## 2. Explainability & SLM Integration
- **Hazard Attribution:** Use **Hazard Ratios** for Cox PH and **SHAP for Survival** (DeepSurv) to identify risk drivers (e.g., "Declining Watch Hours" vs "Competitive Launch").
- **SLM Narrative Engine:** Integrate a **Small Language Model (SLM)** (e.g., Phi-3 or Mistral) to interpret survival curve shifts.
    - **Insight Generation:** Convert SHAP vectors and hazard spikes into natural language: *"The 'Prestige Drama' cohort shows a hazard spike at month 4. The SLM identifies 'Content Exhaustion' as the driver, recommending a transition to 'Reality Competition' genres to extend survival by an estimated 12%."*

## 3. Coding Standards
- **Validation:** Use **Pandera** to enforce that `t_stop > t_start` and `event_indicator` is binary.
- **Purity:** Nodes must be pure functions. Strictly enforce temporal splits to prevent future leakage (Training: T-24 to T-12 months).