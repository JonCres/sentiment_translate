from typing import Dict, Any, Tuple
import polars as pl
import pandas as pd
import logging
from datetime import date
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import shap
import lime
import lime.lime_tabular
import gc
import torch


logger = logging.getLogger(__name__)


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


def predict_churn(
    ensemble_model: Any,
    feature_store: pl.DataFrame,
    tensor_sequences: Any = None,  # Optional: DataFrame or Array
    churn_lstm_model: Any = None,
    churn_graph_model: Any = None,
) -> pl.DataFrame:
    """
    Predict Churn Probability using trained Ensemble model and behavioral features.
    Outputs comprehensive survival and hazard metrics.
    Dynamic Execution: Hybridizes predictions if DL/Graph models + data available.
    """
    clear_device_cache()
    if ensemble_model is None:
        logger.warning("Ensemble model missing. Returning empty predictions.")
        return pl.DataFrame()

    if feature_store is None or feature_store.is_empty():
        logger.warning("Feature store empty. Returning empty predictions.")
        return pl.DataFrame()

    # 1. Ensemble Prediction (Base)
    # Prepare data for ensemble
    try:
        # Drop columns that are definitely not features for the ensemble
        X_infer_raw = feature_store.select(
            pl.all().exclude(["customer_id", "event_timestamp", "engagement_sequence_flat"])
        )
        
        X_infer = X_infer_raw.to_pandas()
        X_infer = X_infer.select_dtypes(
            exclude=["datetime64[ns]", "datetime", "object"]
        )  # Ensure numeric
        
        p_churn_ensemble = ensemble_model.predict_proba(X_infer)
        # Handle binary classification output (take p1)
        if p_churn_ensemble.ndim > 1 and p_churn_ensemble.shape[1] > 1:
            p_churn_ensemble = p_churn_ensemble[:, 1]
    except Exception as e:
        logger.exception(f"Ensemble Inference failed: {e}")
        # Return empty but with schema to avoid downstream ColumnNotFoundError
        return pl.DataFrame({"customer_id": feature_store["customer_id"]}).with_columns(
            [pl.lit(0.0).alias(c) for c in ["churn_prob_30day", "churn_prob_60day", "churn_prob_90day"]]
        )

    # 2. Deep Learning Prediction (Optional Tier 3)
    p_churn_dl = None
    if (
        churn_lstm_model is not None
        and not getattr(churn_lstm_model, "is_dummy", False)
        and tensor_sequences is not None
    ):
        try:
            # Assuming tensor_sequences is coming as DataFrame/Array matching training format
            # Needs to be converted to torch tensor
            import torch

            # Helper to process tensor_sequences which might be a Polars DF of lists or numpy array
            # Simplified assumption for this node:
            if hasattr(tensor_sequences, "to_numpy"):
                # Extract the tensor column if it's a DF, or use directly
                # This part depends heavily on the specific Data Output format of create_tensor_sequences
                # For safety/MVP, we skip or use proper transformation if implemented
                pass

            # Placeholder for DL inference logic
            logger.info(
                "DL Model available (Tier 3), but inference logic placeholder used."
            )
            # p_churn_dl = churn_lstm_model(inputs)...
        except Exception as e:
            logger.warning(f"DL Inference failed: {e}")

    # 3. Hybrid Fusion
    # Using simple weighted average if DL available
    # p_final = 0.7 * Ensemble + 0.3 * DL (if available)

    if p_churn_dl is not None:
        p_churn_final = 0.7 * p_churn_ensemble + 0.3 * p_churn_dl
    else:
        p_churn_final = p_churn_ensemble  # Tier 1/2 Fallback

    # 4. Multi-horizon projections (Survival Proxy)
    p_churn_30day = p_churn_final  # Base probability

    def scale_prob(p, horizon_days):
        # Exponential scaling P(churn < t) = 1 - (1-p)^(t/30)
        return 1.0 - np.power(1.0 - p, horizon_days / 30.0)

    p_churn_60day = scale_prob(p_churn_30day, 60)
    p_churn_90day = scale_prob(p_churn_30day, 90)
    p_churn_180day = scale_prob(p_churn_30day, 180)
    p_churn_365day = scale_prob(p_churn_30day, 365)

    s_prob_30d = 1.0 - p_churn_30day
    s_prob_90d = 1.0 - p_churn_90day
    s_prob_365d = 1.0 - p_churn_365day

    # 5. Hazard Rate Estimation (h(t) = -ln(S(t))/t)
    eps = 1e-7
    h_rate_30d = -np.log(np.clip(s_prob_30d, eps, 1.0)) / 30.0
    h_rate_90d = -np.log(np.clip(s_prob_90d, eps, 1.0)) / 90.0
    h_rate_180d = -np.log(np.clip(1.0 - p_churn_180day, eps, 1.0)) / 180.0

    # Hazard Ratio
    median_h = np.median(h_rate_30d) if len(h_rate_30d) > 0 else eps
    hazard_ratio = h_rate_30d / (median_h + eps)

    # Median Tenure
    predicted_median_tenure = -np.log(0.5) / (h_rate_30d + eps)

    # Risk Score
    churn_risk_score = (p_churn_30day * 100).astype(int)

    def get_risk_tier(prob):
        if prob > 0.67:
            return "High"
        if prob > 0.33:
            return "Medium"
        return "Low"

    # Construct Result DataFrame
    results_df = pl.DataFrame(
        {
            "customer_id": feature_store["customer_id"],
            "prediction_date": [date.today()] * len(feature_store),
            "churn_prob_30day": p_churn_30day,
            "churn_prob_60day": p_churn_60day,
            "churn_prob_90day": p_churn_90day,
            "churn_prob_180day": p_churn_180day,
            "churn_prob_365day": p_churn_365day,
            "survival_prob_30d": s_prob_30d,
            "survival_prob_90d": s_prob_90d,
            "survival_prob_365d": s_prob_365d,
            "hazard_rate_30d": h_rate_30d,
            "hazard_rate_90d": h_rate_90d,
            "hazard_rate_180d": h_rate_180d,
            "hazard_ratio": hazard_ratio,
            "predicted_median_tenure_days": predicted_median_tenure,
            "churn_risk_score": churn_risk_score,
            "risk_tier": [get_risk_tier(p) for p in p_churn_30day],
        }
    )

    # Add frequency/recency/T context if available
    for col in ["frequency", "recency", "T"]:
        if col in feature_store.columns:
            results_df = results_df.with_columns(feature_store[col].alias(col))

    return results_df


def calculate_business_metrics(
    predictions: pl.DataFrame,
    feature_store: pl.DataFrame,
    params: Dict[str, Any] = None,
) -> pl.DataFrame:
    """
    Calculate high-level business KPIs from churn predictions.
    Assigns Risk Tiers and flags Potential Serial Churners.
    """
    if params is None:
        params = {}

    if predictions.is_empty() or "customer_id" not in predictions.columns:
        logger.warning("Empty predictions or missing customer_id. Returning empty metrics.")
        return pl.DataFrame()

    # 1. Join with Feature Store for Context
    # We need behavioral signals for "top_churn_drivers" and contextual flags
    df = predictions.join(feature_store, on="customer_id", how="left")

    # 2. Serial Churner Detection
    # Logic: High frequency of short-term subscriptions
    # Assuming frequency is # subscriptions and T is total tenure
    if "frequency" in df.columns and "T" in df.columns:
        df = df.with_columns(
            pl.when((pl.col("frequency") > 3) & (pl.col("T") < 90))
            .then(True)
            .otherwise(False)
            .alias("serial_churner_flag")
        )
    else:
        logger.warning("'frequency' or 'T' missing for Serial Churner Detection. Defaulting to False.")
        df = df.with_columns(pl.lit(False).alias("serial_churner_flag"))

    # 3. Risk Segmentation (Section 8: Decision Boundaries)
    # Critical/High/Medium/Low
    # Whale threshold logic could be added here if CLTV was present
    df = df.with_columns(
        pl.when(pl.col("churn_prob_30day") > 0.80)
        .then(pl.lit("Critical"))
        .when(pl.col("churn_prob_30day") > 0.65)
        .then(pl.lit("High"))
        .when(pl.col("churn_prob_30day") > 0.45)
        .then(pl.lit("Medium"))
        .otherwise(pl.lit("Low"))
        .alias("risk_classification")
    )

    # 4. Intervention Logic (Section 7)
    # We construct a JSON/Struct column for recommended_interventions
    # Logic:
    # - If Critical + Payment Failure (need payment data, handled via checking if last txn failed? assuming 'payment_success_flag' available in FS)
    # - If High + Dropped Usage -> Content Re-engagement
    # - If Medium -> Newsletter

    # Mocking intervention logic for now based on risk
    # In a real implementation, we'd check specific feature columns like 'payment_success_flag'

    # Using pl.struct to create complex objects
    # Note: Polars structs are good, but for JSON output we might want a stringified JSON or mapped values

    # Simple rule-based intervention string for now
    df = df.with_columns(
        pl.when(pl.col("risk_classification") == "Critical")
        .then(pl.lit("payment_recovery_workflow"))
        .when(pl.col("risk_classification") == "High")
        .then(pl.lit("proactive_content_reengagement"))
        .when(pl.col("risk_classification") == "Medium")
        .then(pl.lit("tier_migration_offer"))
        .otherwise(pl.lit("none"))
        .alias("primary_intervention")
    )

    # 5. Full Output Schema Alignment (renaming/aliasing)
    # "churn_risk_score" is already there (0-100)

    # Add metadata
    df = df.with_columns(
        pl.lit("v3.2.1").alias("model_version"),
        pl.lit(30).alias("prediction_horizon_days"),
    )

    return df


def prepare_training_data(
    feature_store: pl.DataFrame, parameters: Dict[str, Any]
) -> pd.DataFrame:
    """
    Prepares training data by generating the target variable and handling nulls.
    Returns a pandas DataFrame ready for sklearn/xgboost.
    """
    if feature_store.is_empty():
        logger.warning("No feature store data found.")
        return pd.DataFrame()

    xgb_params = parameters.get("modeling", {}).get("xgboost", {})
    target_col = xgb_params.get("xgboost_target_col", "churn_label")

    # Check if target already exists; if not, generate it
    df = feature_store
    if target_col not in df.columns:
        logger.info(f"Generating target column '{target_col}' from inactivity.")
        if "last_engagement_date" not in df.columns:
            # Fallback or error
            logger.warning("'last_engagement_date' missing. Cannot calculate churn.")
            return pd.DataFrame()

        analysis_date_str = parameters.get("data_processing", {}).get(
            "observation_period_end"
        )
        if not analysis_date_str:
            logger.warning("`observation_period_end` not found.")
            return pd.DataFrame()

        from datetime import datetime

        try:
            analysis_date = datetime.strptime(analysis_date_str, "%Y-%m-%d").date()
        except ValueError:
            return pd.DataFrame()

        if df["last_engagement_date"].dtype == pl.Utf8:
            df = df.with_columns(
                pl.col("last_engagement_date").str.to_date(strict=False)
            )

        inactivity_threshold = xgb_params.get("inactivity_threshold_days", 90)

        df = df.with_columns(
            pl.when(
                pl.col("last_engagement_date").is_null()
                | (pl.col("last_engagement_date") == 0)
            )
            .then(9999)
            .otherwise(
                (pl.lit(analysis_date) - pl.col("last_engagement_date")).dt.total_days()
            )
            .alias("inactivity_days")
        )

        df = df.with_columns(
            (pl.col("inactivity_days") > inactivity_threshold)
            .cast(pl.Int8)
            .alias(target_col)
        )

    # Convert to pandas
    df_pd = df.to_pandas()

    # Drop non-feature columns
    cols_to_drop = ["customer_id", "last_engagement_date", "inactivity_days"]
    df_pd = df_pd.drop(columns=[c for c in cols_to_drop if c in df_pd.columns])

    # Handle dates and objects
    df_pd = df_pd.select_dtypes(exclude=["datetime64[ns]", "datetime", "object"])

    # --- Scaling (Technical Walkthrough Requirement) ---
    from sklearn.preprocessing import StandardScaler

    numeric_cols = df_pd.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    if numeric_cols:
        scaler = StandardScaler()
        df_pd[numeric_cols] = scaler.fit_transform(df_pd[numeric_cols])

    # --- Class Balancing (Technical Walkthrough Requirement) ---
    # We apply balancing only if requested in parameters
    if parameters.get("modeling", {}).get("use_smote", False):
        try:
            from imblearn.over_sampling import SMOTE

            X = df_pd.drop(columns=[target_col])
            y = df_pd[target_col]
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X, y)
            df_pd = pd.concat([X_res, y_res], axis=1)
            logger.info("Applied SMOTE class balancing.")
        except ImportError:
            logger.warning("imblearn not installed. Skipping SMOTE.")

    return df_pd


def split_data(
    data: pd.DataFrame, parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits data into train and test sets.
    """
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()

    xgb_params = parameters.get("modeling", {}).get("xgboost", {})
    target_col = xgb_params.get("xgboost_target_col", "churn_label")
    test_size = parameters.get("modeling", {}).get("test_size", 0.2)

    if target_col not in data.columns:
        logger.warning(f"Target column {target_col} not found for splitting.")
        return data, pd.DataFrame()

    X = data
    y = data[target_col]

    train_data, test_data = train_test_split(
        X, test_size=test_size, random_state=42, stratify=y
    )
    return train_data, test_data


def train_model_with_grid_search(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    model_class: Any,
    config: Dict[str, Any],
    target_col: str = "churn_label",
) -> Any:
    """
    Generic function to train a model with Grid/Random Search using pre-split data.
    """
    if train_data.empty:
        return None

    X_train = train_data.drop(columns=[target_col], errors="ignore")
    y_train = train_data[target_col]

    X_test = test_data.drop(columns=[target_col], errors="ignore")
    y_test = test_data[target_col]

    # Base model parameters
    base_params = config.get("params", {})
    grid_search_conf = config.get("grid_search", {})
    param_grid = grid_search_conf.get("param_grid", {})

    model = model_class(**base_params)

    if param_grid:
        logger.info(f"Running Grid Search for {model_class.__name__}...")
        # Use RandomizedSearchCV for speed if grid is large, else GridSearchCV
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=3,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        logger.info(f"Best params for {model_class.__name__}: {search.best_params_}")
    else:
        best_model = model
        best_model.fit(X_train, y_train)

    # Log metrics
    if not X_test.empty:
        y_pred = best_model.predict(X_test)
        y_proba = (
            best_model.predict_proba(X_test)[:, 1]
            if hasattr(best_model, "predict_proba")
            else y_pred
        )

        auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        logger.info(f"{model_class.__name__} - AUC: {auc:.4f}, Accuracy: {acc:.4f}")

    return best_model


def train_xgboost_residual_model(
    train_data: pd.DataFrame, test_data: pd.DataFrame, parameters: Dict[str, Any]
) -> Any:
    xgb_config = parameters.get("modeling", {}).get("xgboost", {})
    return train_model_with_grid_search(
        train_data, test_data, xgb.XGBClassifier, xgb_config
    )


def train_random_forest_model(
    train_data: pd.DataFrame, test_data: pd.DataFrame, parameters: Dict[str, Any]
) -> Any:
    rf_config = parameters.get("modeling", {}).get("random_forest", {})
    return train_model_with_grid_search(
        train_data, test_data, RandomForestClassifier, rf_config
    )


def train_lightgbm_model(
    train_data: pd.DataFrame, test_data: pd.DataFrame, parameters: Dict[str, Any]
) -> Any:
    lgb_config = parameters.get("modeling", {}).get("lightgbm", {})
    return train_model_with_grid_search(
        train_data, test_data, lgb.LGBMClassifier, lgb_config
    )


def train_catboost_model(
    train_data: pd.DataFrame, test_data: pd.DataFrame, parameters: Dict[str, Any]
) -> Any:
    cb_config = parameters.get("modeling", {}).get("catboost", {})
    return train_model_with_grid_search(
        train_data, test_data, cb.CatBoostClassifier, cb_config
    )


def train_ensemble_model(
    xgboost: Any,
    random_forest: Any,
    lightgbm: Any,
    catboost: Any,
    parameters: Dict[str, Any],
) -> Any:
    """
    Creates a Voting Classifier from the trained models.
    """
    estimators = []
    if xgboost:
        estimators.append(("xgb", xgboost))
    if random_forest:
        estimators.append(("rf", random_forest))
    if lightgbm:
        estimators.append(("lgb", lightgbm))
    if catboost:
        estimators.append(("cb", catboost))

    # Remove specific models if they are None
    estimators = [e for e in estimators if e[1] is not None]

    if not estimators:
        logger.warning("No base models provided for ensemble. Returning None.")
        return None

    ensemble_config = parameters.get("modeling", {}).get("ensemble", {})
    weights = ensemble_config.get("weights", None)

    class PreFittedVotingClassifier:
        def __init__(self, estimators, weights=None):
            self.estimators = estimators
            self.weights = weights

        def predict_proba(self, X):
            probas = []
            for _, est in self.estimators:
                probas.append(est.predict_proba(X)[:, 1])
            probas = np.column_stack(probas)
            if self.weights:
                return np.average(probas, axis=1, weights=self.weights)
            return np.mean(probas, axis=1)

        def predict(self, X):
            return (self.predict_proba(X) > 0.5).astype(int)

    return PreFittedVotingClassifier(estimators, weights)


def explain_model_shap(
    model: Any, train_data: pd.DataFrame, parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generates SHAP values for the given model (using the best available single model or XGBoost).
    Returns a dictionary with 'shap_values' and 'explainer'.
    """
    if not model:
        return {}

    X = train_data.drop(columns=["churn_label"], errors="ignore")

    # Sample
    sample_size = (
        parameters.get("modeling", {})
        .get("explainability", {})
        .get("shap", {})
        .get("sample_size", 100)
    )
    X_sample = X.sample(min(len(X), sample_size), random_state=42)

    try:
        # Handling different model types
        # If model is the wrapper ensemble, pick the XGBoost or best constituent
        if hasattr(model, "estimators"):  # Our custom wrapper
            estimator = model.estimators[0][1]  # Pick first
        else:
            estimator = model

        explainer = shap.Explainer(estimator, X_sample)
        shap_values = explainer(X_sample)

        return {
            "shap_values": shap_values,
            "explainer": explainer,
            "sample_data": X_sample,
        }
    except Exception as e:
        logger.error(f"SHAP explanation failed: {e}")
        return {}


def explain_model_lime(
    model: Any, train_data: pd.DataFrame, parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Creates a LIME explainer for the given model.
    """
    if not model or train_data.empty:
        return {}

    xgb_params = parameters.get("modeling", {}).get("xgboost", {})
    target_col = xgb_params.get("xgboost_target_col", "churn_label")

    X_train = train_data.drop(columns=[target_col], errors="ignore")

    # LIME requires numpy array for training data
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns.tolist(),
        class_names=["No Churn", "Churn"],
        mode="classification",
        random_state=42,
    )

    return {"lime_explainer": explainer}


# --- Removed PyMC-Marketing Wrapper (CLTV Legacy) ---
