from typing import Dict, Any, List
import polars as pl
import pandas as pd
import logging
import gc
import numpy as np
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
import shap
import lime
import lime.lime_tabular
from sklearn.neighbors import NearestNeighbors
import torch
import torchtuples as tt
from pycox.models import DeepSurv
from lifelines.utils import concordance_index

logger = logging.getLogger(__name__)


def clear_device_cache():
    """Release unoccupied cached memory for available devices (CUDA, XPU, MPS)."""
    # Force Python garbage collection first
    gc.collect()

    try:
        # CUDA (NVIDIA)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # XPU (Intel)
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()

        # MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception as e:
        # Log warning but don't crash processing
        logger.warning(f"Failed to clear device cache: {e}")


def prepare_model_input(
    feature_store: pl.DataFrame, survival_data_prepared: pl.DataFrame
) -> pd.DataFrame:
    """
    Merges Feature Store (Behavioral/Transactions) with Survival Targets (T, E).
    Returns a Pandas DataFrame ready for modeling.
    """
    logger.info("Merging feature store and survival data...")

    # Ensure IDs are strings
    fs = feature_store.with_columns(pl.col("customer_id").cast(pl.String))
    sd = survival_data_prepared.with_columns(pl.col("customer_id").cast(pl.String))

    # Inner join - we need both features and target
    # Note: survival_data_prepared has t_start, t_stop, event.
    # If we stick to single-row per user (static model), t_stop is duration T.

    merged = fs.join(sd, on="customer_id", how="inner")

    # Convert to Pandas
    df = merged.to_pandas()

    # Standardize column names for downstream models
    if "t_stop" in df.columns:
        df["duration"] = df["t_stop"]
        # If t_start exists and > 0, we might need to handle interval censoring/truncation
        # But for DeepSurv standard, we assume right censored only or use start-stop
        # For now, map t_stop -> duration.
    elif "T" in df.columns:
        df["duration"] = df["T"]

    if "E" in df.columns and "event" not in df.columns:
        df["event"] = df["E"]

    return df


def evaluate_survival_model(model: Any, test_data: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluates survival model performance (C-index).
    """
    logger.info("Evaluating model performance...")

    # Separate X and y
    X = test_data.drop(
        columns=["duration", "event", "t_start", "t_stop", "E", "T"], errors="ignore"
    ).select_dtypes(include=[np.number])
    T = test_data["duration"]
    E = test_data["event"]

    metrics = {}

    # Calculate C-index
    try:
        # DeepSurv/PyCox
        if hasattr(model, "predict_surv_df"):
            # pycox c_index needs risk scores (higher risk = lower survival time)
            # predict() returns log_partial_hazard, which is risk score.
            risk_scores = model.predict(X.values.astype("float32")).flatten()
            c_index = concordance_index(
                T, -risk_scores, E
            )  # wait, higher risk -> shorter time. c-index expects risk?
            # Lifelines c_index: (event_times, predicted_scores, event_observed)
            # If higher score = higher risk (shorter survival), concordance is high?
            # Standard C-index: correctness of ranking.
            # pycox documentation says: concordance_index(durations, -risk, events)
            metrics["c_index"] = concordance_index(T, -risk_scores, E)

        # CoxPH/RSF (lifelines/sklearn)
        elif hasattr(model, "predict_partial_hazard"):
            risk_scores = model.predict_partial_hazard(X)
            metrics["c_index"] = concordance_index(T, -risk_scores, E)
        elif hasattr(model, "predict"):
            # RSF predicts risk score usually?
            preds = model.predict(X)
            metrics["c_index"] = concordance_index(T, -preds, E)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        metrics["error"] = 1.0

    return metrics


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
    logger.info("Predicting survival metrics using DeepSurv...")
    clear_device_cache()

    if data.is_empty():
        logger.warning("No data provided for prediction.")
        return pl.DataFrame()

    customer_ids = data["customer_id"]

    # 1. Prepare Features
    # Drop non-feature columns (IDs, timestamps)
    # Ensure strict alignment with training features is handled upstream or via feature store schema
    feature_cols = [
        c
        for c in data.columns
        if c not in ["customer_id", "event_timestamp", "churn_label", "date"]
    ]
    X_pd = data.select(feature_cols).to_pandas()
    X = X_pd.values.astype("float32")

    # 2. Predict Survival Functions S(t) using pycox
    # Returns DataFrame where index is time, columns are samples
    try:
        surv_df = survival_model.predict_surv_df(X)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        # Fallback for compilation/model errors
        return pl.DataFrame({"customer_id": customer_ids})

    # 3. Extract Metrics for Horizons
    horizons = [30, 60, 90, 180, 365]
    results = {"customer_id": customer_ids}
    times = surv_df.index.values

    for h in horizons:
        # Step function lookups
        idx = np.searchsorted(times, h, side="right") - 1
        if idx < 0:
            probs = np.ones(len(customer_ids))
        else:
            t_col = times[idx]
            probs = surv_df.loc[t_col].values

        results[f"survival_prob_{h}d"] = probs
        results[f"churn_prob_{h}day"] = 1 - probs

        # Hazard approximation: h(t) = f(t)/S(t) ~ -(S(t+dt)-S(t)) / (dt * S(t))
        # Or just use cumulative hazard H(t) = -ln(S(t)) implies S(t) = exp(-H(t))
        # Hazard Rate at specific point is derivative.
        # We'll use 1 - survival as proxy for cumulative risk for now in result map.
        # For instantaneous hazard output, we can approximate slope.
        # Simple approximation: h(30) ~ (S(0)-S(30))/30 if S(0)=1
        results[f"hazard_rate_{h}d"] = (1 - probs) / h

    # 4. Leading Indicators & Risk
    # Hazard Ratio: We can use the negative log of S(t at median) / baseline?
    # Or simply 1 - S(30) normalized.
    # DeepSurv outputs log-risk directly via `predict` usually, but pycox wrapper emphasizes surv_df
    # Let's get the partial log likelihood risk score if available
    try:
        # pycox DeepSurv predict returns log-risk (linear predictor)
        risk_scores = survival_model.predict(X).flatten()
        results["hazard_ratio"] = np.exp(risk_scores)  # risk relative to baseline
    except:
        results["hazard_ratio"] = np.ones(len(customer_ids))

    # 5. Median Tenure (Time where S(t) <= 0.5)
    median_tenure = []
    for i in range(len(customer_ids)):
        # Get column i of surv_df
        # Find first index where prob <= 0.5
        s_series = surv_df.iloc[:, i]
        below_median = s_series[s_series <= 0.5]
        if below_median.empty:
            val = times[-1] * 1.5  # Censored/Long
        else:
            val = below_median.index[0]
        median_tenure.append(val)

    results["predicted_median_tenure_days"] = median_tenure

    return pl.DataFrame(results)


def calculate_business_metrics(
    predictions: pl.DataFrame, feature_store: pl.DataFrame
) -> pl.DataFrame:
    """
    Adds Intervention Priority and risk-based metrics.
    Segments by Churn Urgency as per GEMINI.md standards.
    """
    logger.info("Calculating business metrics...")

    # Calculate Risk Segment and Intervention Priority
    df = predictions.with_columns(
        [
            pl.when(pl.col("churn_prob_30day") > 0.15)
            .then(pl.lit("Critical"))
            .when(pl.col("churn_prob_30day") > 0.10)
            .then(pl.lit("High"))
            .when(pl.col("churn_prob_30day") > 0.05)
            .then(pl.lit("Medium"))
            .otherwise(pl.lit("Low"))
            .alias("intervention_priority"),
            pl.when(pl.col("churn_prob_30day") > 0.15)
            .then(pl.lit("Imminent (0-7 days)"))
            .when(pl.col("churn_prob_90day") > 0.25)
            .then(pl.lit("At-Risk (8-30 days)"))
            .otherwise(pl.lit("Passive (31-90 days)"))
            .alias("risk_segment"),
        ]
    )

    # Add Intervention Window
    # Recommended: 14-21 days before predicted termination if risk high
    df = df.with_columns(
        pl.lit("14-21 days before predicted risk peak").alias("intervention_window")
    )

    return df


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
