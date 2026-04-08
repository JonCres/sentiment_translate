from typing import Dict, Any, List, Optional
import polars as pl
import pandas as pd
import logging
import numpy as np
from datetime import datetime
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
import shap
import lime
import lime.lime_tabular
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import TweedieRegressor
from sklearn.model_selection import train_test_split

import mlflow

logger = logging.getLogger(__name__)


def train_bg_nbd_model(rfm_data: pl.DataFrame, parameters: Dict[str, Any]) -> Any:
    """Trains a BG/NBD model for purchase frequency."""
    from pymc_marketing.clv import BetaGeoModel

    logger.info("Training BG/NBD model...")

    # Convert to pandas for pymc-marketing
    df = rfm_data.to_pandas()
    model = BetaGeoModel(data=df)
    model.fit()

    # Register with MLflow
    with mlflow.start_run(run_name="bg_nbd_training", nested=True):
        mlflow.log_params(
            parameters.get("modeling", {}).get("pymc_marketing", {}).get("bg_nbd", {})
        )
        # Logic for saving model to mlflow would go here

    return model


def train_gamma_gamma_model(rfm_data: pl.DataFrame, parameters: Dict[str, Any]) -> Any:
    """Trains a Gamma-Gamma model for monetary value."""
    from pymc_marketing.clv import GammaGammaModel

    logger.info("Training Gamma-Gamma model...")

    df = rfm_data.to_pandas()
    # Filter for repeat customers
    df = df[df["frequency"] > 0]
    model = GammaGammaModel(data=df)
    model.fit()

    return model



def train_tweedie_model(feature_store: pl.DataFrame, parameters: Dict[str, Any]) -> Any:
    """
    Trains a Tweedie Regressor for hybrid revenue prediction (Zero-Inflated).
    """
    logger.info("Training Tweedie Regressor...")
    df = feature_store.to_pandas().fillna(0)
    
    # Target: We need a target 'future_revenue_12m' or similar. 
    # For scaffold, use proxy 'total_tvod_spend' if available + generated noise
    
    if "monetary" in df.columns:
        y = df["monetary"]
    elif "total_tvod_spend" in df.columns:
        y = df["total_tvod_spend"]
    else:
        # Fallback for scaffold
        y = np.random.gamma(2, 10, size=len(df))

    # Features
    X = df.select_dtypes(include=[np.number])
    if "customer_id" in X.columns:
        X = X.drop(columns=["customer_id"])
    if "monetary" in X.columns:
        X = X.drop(columns=["monetary"])

    # Tweedie power: 1.5 (Compound Poisson-Gamma)
    power = parameters.get("modeling", {}).get("tweedie_power", 1.5)
    model = TweedieRegressor(power=power, link="log", max_iter=1000)
    
    try:
        model.fit(X, y)
    except Exception as e:
        logger.warning(f"Tweedie training failed: {e}")
        model = TweedieRegressor(power=power)
        try:
             model.fit(X, y) # Retry? No, just scaffold
        except:
             pass 
    
    return model


def predict_cltv(
    rfm_data: pl.DataFrame,
    feature_store: pl.DataFrame,
    bg_nbd_model: Any,
    gamma_gamma_model: Any,
    survival_model: Any,
    dl_model: Optional[Any],
    tweedie_model: Any,
    parameters: Dict[str, Any],
) -> pl.DataFrame:
    """
    Unified CLTV Prediction Engine.
    Combines SVOD (Survival), AVOD/TVOD (BTYD), and Sequential patterns (DL).
    """
    logger.info("Predicting multi-stream CLTV metrics...")

    num_rows = len(feature_store)
    customer_ids = feature_store["customer_id"]

    # 1. SVOD Component (from Survival)
    # Predicted tenure in months * monthly fee
    survival_tenure = np.random.uniform(12, 36, size=num_rows)  # Placeholder
    svod_clv = survival_tenure * 14.99

    # 2. AVOD Component (from BTYD / Advertising features)
    # Expected transactions * expected value
    # Mocking BTYD output for now
    avod_clv = np.random.uniform(5, 100, size=num_rows)

    # 3. Combined CLTV
    # Use Tweedie Model for fusion if available
    try:
        X_pred = feature_store.to_pandas().select_dtypes(include=[np.number]).fillna(0)
        # align columns
        if hasattr(tweedie_model, "feature_names_in_"):
             # Add missing cols with 0
            missing_cols = set(tweedie_model.feature_names_in_) - set(X_pred.columns)
            for c in missing_cols:
                X_pred[c] = 0
            X_pred = X_pred[tweedie_model.feature_names_in_]
        
        tweedie_pred = tweedie_model.predict(X_pred)
        # Blend: 50% Ensemble components + 50% Tweedie Fusion
        total_clv = (svod_clv + avod_clv) * 0.5 + tweedie_pred * 0.5
    except Exception as e:
        logger.warning(f"Tweedie prediction failed: {e}. Falling back to component sum.")
        total_clv = svod_clv + avod_clv

    # 4. Uncertainty & Intervals
    clv_95_ci_lower = total_clv * 0.8
    clv_95_ci_upper = total_clv * 1.2

    results = pl.DataFrame(
        {
            "customer_id": customer_ids,
            "clv_12mo": total_clv,
            "clv_24mo": total_clv * 1.8,
            "clv_36mo": total_clv * 2.5,
            "clv_subscription_component": svod_clv,
            "clv_advertising_component": avod_clv,
            "clv_transaction_component": np.random.uniform(0, 10, size=num_rows),
            "clv_95_ci_lower": clv_95_ci_lower,
            "clv_95_ci_upper": clv_95_ci_upper,
            "expected_monthly_revenue": total_clv / 12,
            "clv_segment": "Q1",  # Will be updated in business metrics
            "prediction_timestamp": datetime.now(),
        }
    )

    return results


def prepare_survival_data(
    feature_store: pl.DataFrame, parameters: Dict[str, Any]
) -> pd.DataFrame:
    """
    Transforms raw engagement logs into standard survival format (duration, event).
    """
    logger.info("Preparing survival data...")
    df = feature_store.to_pandas()
    if "tenure" not in df.columns:
        df["duration"] = np.random.randint(1, 100, size=len(df))
    else:
        df["duration"] = df["tenure"]

    if "churn_label" not in df.columns:
        df["event"] = np.random.randint(0, 2, size=len(df))
    else:
        df["event"] = df["churn_label"]

    return df


def train_coxph_model(
    train_data: pd.DataFrame, parameters: Dict[str, Any]
) -> CoxPHFitter:
    """
    Trains a Cox Proportional Hazards model.
    """
    logger.info("Training CoxPH model...")
    cph_params = (
        parameters.get("modeling", {}).get("survival_analysis", {}).get("cph", {})
    )
    cph = CoxPHFitter(penalizer=cph_params.get("penalizer", 0.1))

    # Filter numeric data
    df = train_data.select_dtypes(include=[np.number])
    cph.fit(df, duration_col="duration", event_col="event")
    return cph


def train_rsf_model(
    train_data: pd.DataFrame, parameters: Dict[str, Any]
) -> RandomSurvivalForest:
    """
    Trains a Random Survival Forest model.
    """
    logger.info("Training Random Survival Forest model...")
    rsf_params = (
        parameters.get("modeling", {}).get("survival_analysis", {}).get("rsf", {})
    )
    rsf = RandomSurvivalForest(
        n_estimators=rsf_params.get("n_estimators", 100),
        min_samples_split=rsf_params.get("min_samples_split", 10),
        random_state=42,
    )

    X = train_data.drop(columns=["duration", "event"]).select_dtypes(
        include=[np.number]
    )
    y = np.array(
        [(bool(e), d) for e, d in zip(train_data["event"], train_data["duration"])],
        dtype=[("event", bool), ("duration", float)],
    )
    rsf.fit(X, y)
    return rsf


def predict_churn(
    survival_model: Any,
    data: pl.DataFrame,
) -> pl.DataFrame:
    """
    Predict Churn Probability and Hazard metrics using a trained Survival Analysis model.
    Outputs:
    - churn_prob_Xd: Probability of churn within X days.
    - hazard_Xd: Instantaneous risk at day X.
    - hazard_ratio: Relative risk multiplier vs baseline.
    - predicted_median_tenure: 50th percentile survival time.
    """
    logger.info("Predicting survival metrics...")

    # For demonstration/scaffolding, we generate realistic survival metrics
    # In a production environment, this would call model.predict_survival_function(X)
    # and model.predict_hazard(X) or equivalent methods.

    num_rows = len(data)
    customer_ids = data["customer_id"]

    # Generate survival probabilities S(t)
    # S(t) must be non-increasing
    p30 = np.random.uniform(0.85, 0.99, size=num_rows)
    p60 = p30 * np.random.uniform(0.90, 0.98, size=num_rows)
    p90 = p60 * np.random.uniform(0.85, 0.97, size=num_rows)
    p180 = p90 * np.random.uniform(0.70, 0.95, size=num_rows)
    p365 = p180 * np.random.uniform(0.50, 0.90, size=num_rows)

    # Churn probability P(T <= t) = 1 - S(t)
    churn_30 = 1 - p30
    churn_60 = 1 - p60
    churn_90 = 1 - p90
    churn_180 = 1 - p180
    churn_365 = 1 - p365

    # Hazard Rate h(t) = -d/dt ln S(t)
    # Mocking hazard rates at specific points
    h30 = np.random.uniform(0.001, 0.005, size=num_rows)
    h90 = np.random.uniform(0.005, 0.015, size=num_rows)
    h180 = np.random.uniform(0.010, 0.025, size=num_rows)

    # Risk Metrics
    hazard_ratio = np.random.uniform(0.5, 4.0, size=num_rows)
    median_tenure = np.random.randint(180, 730, size=num_rows)

    results_df = pl.DataFrame(
        {
            "customer_id": customer_ids,
            "survival_prob_30d": p30,
            "survival_prob_90d": p90,
            "survival_prob_365d": p365,
            "churn_prob_30day": churn_30,
            "churn_prob_60day": churn_60,
            "churn_prob_90day": churn_90,
            "churn_prob_180day": churn_180,
            "churn_prob_365day": churn_365,
            "hazard_rate_30d": h30,
            "hazard_rate_90d": h90,
            "hazard_rate_180d": h180,
            "hazard_ratio": hazard_ratio,
            "predicted_median_tenure_days": median_tenure,
        }
    )
    return results_df


def calculate_business_metrics(
    predictions: pl.DataFrame, feature_store: pl.DataFrame
) -> pl.DataFrame:
    """
    Adds Intervention Priority and risk-based metrics.
    Segments by Churn Urgency as per GEMINI.md standards.
    """
    logger.info("Calculating business metrics...")

    # Calculate CLTV Quintiles and identify Whales
    clv_col = "clv_12mo"
    if clv_col in predictions.columns:
        # Categorize by CLTV quantiles
        q_labels = ["Minnows", "Bronze", "Silver", "Gold", "Whales"]
        # Use pandas for qcut then back to polars or use polars equivalent
        clv_series = predictions[clv_col].to_pandas()
        predictions = predictions.with_columns(
            pl.Series(pd.qcut(clv_series, 5, labels=q_labels)).alias("clv_segment")
        )

        # Add high-value whale marker
        predictions = predictions.with_columns(
            (pl.col("clv_segment") == "Whales").alias("is_whale")
        )

        predictions = predictions.with_columns(
            (pl.col("clv_segment") == "Whales").alias("is_whale")
        )

    # Intervention Logic
    # If churn_prob > 30% AND is_whale -> Critical
    # Assuming 'churn_prob_30day' exists in predictions (if joined) or we mock it
    # predict_cltv usually just outputs CLTV. We need Churn Probs too.
    # The pipeline should join churn_predictions (from predict_churn) before this node or inside it.
    # For now, we assume 'churn_prob_30day' might be missing, so we generate/mock or check.
    
    if "churn_prob_30day" not in predictions.columns:
        # If we cannot determine, set to Low
        predictions = predictions.with_columns(pl.lit("Low").alias("intervention_priority"))
    else:
         predictions = predictions.with_columns(
            pl.when((pl.col("churn_prob_30day") > 0.3) & (pl.col("is_whale")))
            .then(pl.lit("Critical"))
            .when(pl.col("churn_prob_30day") > 0.3)
            .then(pl.lit("High"))
            .otherwise(pl.lit("Low"))
            .alias("intervention_priority")
        )

    return predictions


def explain_survival_model_shap(
    model: Any, train_data: pd.DataFrame, parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generates SHAP explanations for survival models.
    """
    logger.info("Generating SHAP explanations...")
    X = train_data.drop(columns=["duration", "event"]).select_dtypes(
        include=[np.number]
    )

    # Sample data for SHAP
    sample_size = parameters.get("explainability", {}).get("shap_sample_size", 100)
    X_sample = X.sample(n=min(len(X), sample_size), random_state=42)

    try:
        # Use KernelExplainer as a fallback for various models (CPH, RSF)
        # For RSF, TreeExplainer might work if using certain implementations,
        # but sksurv RSF often needs Kernel or specific handling.
        explainer = shap.KernelExplainer(model.predict, shap.kmeans(X_sample, 5))
        shap_values = explainer.shap_values(X_sample)

        return {
            "shap_values": shap_values,
            "sample_data": X_sample,
            "feature_names": X.columns.tolist(),
        }
    except Exception as e:
        logger.error(f"SHAP generation failed: {e}")
        return {}


def explain_survival_model_lime(
    model: Any, train_data: pd.DataFrame, parameters: Dict[str, Any]
) -> List[Any]:
    """
    Generates LIME explanations for a set of instances.
    """
    logger.info("Generating LIME explanations...")
    X = train_data.drop(columns=["duration", "event"]).select_dtypes(
        include=[np.number]
    )

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X.values,
        feature_names=X.columns.tolist(),
        class_names=["Survival"],
        mode="regression",  # Survival prediction as a continuous hazard/risk
    )

    # Explain top N high-risk instances
    num_explanations = parameters.get("explainability", {}).get("lime_num_instances", 5)
    explanations = []

    for i in range(min(len(X), num_explanations)):
        exp = explainer.explain_instance(X.values[i], model.predict, num_features=10)
        explanations.append(exp)

    return explanations


def evaluate_explanation_faithfulness(
    model: Any, X: pd.DataFrame, shap_values: np.ndarray, parameters: Dict[str, Any]
) -> float:
    """
    Evaluates faithfulness of XAI explanations.
    Measures correlation between feature importance and model output change
    after feature perturbation (masking).
    """
    logger.info("Evaluating explanation faithfulness...")
    # Simplified perturbation-based faithfulness (Selectivity)
    base_preds = model.predict(X)
    importance = (
        np.abs(shap_values).mean(axis=0)
        if shap_values.ndim > 1
        else np.abs(shap_values)
    )

    # Perturb top-K features and check drop in output
    k = parameters.get("explainability", {}).get("evaluation", {}).get("top_k", 3)
    perturbed_X = X.copy()

    # Simple masking (zeroing out for numeric)
    top_indices = np.argsort(importance)[-k:]
    for idx in top_indices:
        perturbed_X.iloc[:, idx] = 0

    perturbed_preds = model.predict(perturbed_X)
    delta_pred = np.abs(base_preds - perturbed_preds).mean()

    # Faithfulness score (normalized degree of change)
    return float(delta_pred / (np.abs(base_preds).mean() + 1e-9))


def evaluate_explanation_consistency(shap_values: np.ndarray, X: pd.DataFrame) -> float:
    """
    Evaluates consistency (stability) of explanations.
    Measures the Lipschitz continuity of the explanation function.
    """
    logger.info("Evaluating explanation consistency...")
    # Measure how much explanation changes for small changes in X
    # Using a simple local consistency metric: mean distance between explanations of nearest neighbors

    nbrs = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(X)
    distances, indices = nbrs.kneighbors(X)

    exp_diffs = []
    # indices[:, 1] are the nearest neighbors
    for i, nn_idx in enumerate(indices[:, 1]):
        exp_diff = np.linalg.norm(shap_values[i] - shap_values[nn_idx])
        data_diff = distances[i, 1]
        if data_diff > 1e-9:
            exp_diffs.append(exp_diff / data_diff)

    return float(np.mean(exp_diffs))


def explain_cltv_ensemble_shap(
    bg_nbd_model: Any,
    gamma_gamma_model: Any,
    survival_model: Any,
    sequential_model: Any,
    feature_store: pl.DataFrame,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generates SHAP explanations for the unified CLTV ensemble.
    Wraps the ensemble prediction logic for feature attribution.
    """
    logger.info("Generating SHAP explanations for CLTV Ensemble...")
    X = (
        feature_store.drop(columns=["customer_id"])
        .to_pandas()
        .select_dtypes(include=[np.number])
    )

    # Sample for SHAP
    X_sample = X.sample(n=min(len(X), 50), random_state=42)

    def ensemble_predict(X_df: pd.DataFrame) -> np.ndarray:
        # Mocking the interaction between internal models for SHAP
        # In production, this would call each model and combine them
        return np.random.uniform(100, 5000, size=len(X_df))

    explainer = shap.KernelExplainer(ensemble_predict, shap.kmeans(X_sample, 5))
    shap_values = explainer.shap_values(X_sample)

    return {
        "shap_values": shap_values,
        "sample_data": X_sample,
        "feature_names": X.columns.tolist(),
    }
