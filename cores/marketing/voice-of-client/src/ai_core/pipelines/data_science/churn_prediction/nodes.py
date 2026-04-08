import logging
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from typing import Dict, Any, Tuple
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from utils.device import get_device

logger = logging.getLogger(__name__)


def train_churn_predictor(
    data: pd.DataFrame, parameters: Dict[str, Any]
) -> Tuple[xgb.XGBClassifier, Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Train XGBoost to predict Churn Probability (0-1).

    Args:
        data: DataFrame with sentiment and emotion features.
        parameters: AI Modeling parameters.

    Returns:
        Trained XGBoost model, performance metrics, X_test and y_test.
    """
    churn_params = parameters.get("churn", {"enabled": True, "min_samples": 50})

    # 1. Activation & Data Threshold Check
    if not churn_params.get("enabled", True):
        logger.info("Churn model is disabled by configuration.")
        return (
            xgb.XGBClassifier(),
            {},
            pd.DataFrame({"dummy": []}),
            pd.DataFrame({"dummy": []}),
        )

    if len(data) < churn_params.get("min_samples", 1):
        logger.warning(
            f"Insufficient data for churn modeling: {len(data)} records found, {churn_params.get('min_samples')} required."
        )
        return (
            xgb.XGBClassifier(),
            {},
            pd.DataFrame({"dummy": []}),
            pd.DataFrame({"dummy": []}),
        )

    # Create target: Churn label (synthetic if not present)
    if "churn_label" not in data.columns:
        logger.info("Generating synthetic churn labels for training demonstration.")
        # Use normalized NPS from previous node
        nps_norm = (
            data["nps_score_norm"]
            if "nps_score_norm" in data.columns
            else (data.get("NPS_Score", 5) / 10.0)
        )

        # Risk proxy: Low NPS + Negative Emotions
        prob = (1 - nps_norm) * 0.7 + data.get("emo_anger", 0.1) * 0.3
        data["churn_label"] = (prob > 0.5).astype(int)

        if data["churn_label"].nunique() < 2 and len(data) >= 2:
            data.loc[data.index[0], "churn_label"] = 0
            data.loc[data.index[1], "churn_label"] = 1

    sentiment_cols = [c for c in data.columns if c.startswith("sent_")]
    emotion_cols = [c for c in data.columns if c.startswith("emo_")]
    feature_cols = sentiment_cols + emotion_cols

    if "nps_score_norm" in data.columns:
        feature_cols.append("nps_score_norm")
    if "Contract_Value" in data.columns:
        feature_cols.append("Contract_Value")

    logger.info(f"Training churn predictor with features: {feature_cols}")
    X = data[feature_cols].copy()
    y = data["churn_label"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    device_str = get_device(purpose="churn prediction", framework="xgboost")

    # Use specific churn params or fall back to general ratings params
    ratings_params = parameters.get("ratings", {})
    xgb_params = {
        "n_estimators": churn_params.get(
            "n_estimators", ratings_params.get("n_estimators", 200)
        ),
        "learning_rate": churn_params.get(
            "learning_rate", ratings_params.get("learning_rate", 0.05)
        ),
        "max_depth": churn_params.get("max_depth", ratings_params.get("max_depth", 6)),
        "objective": "binary:logistic",
        "random_state": 42,
    }

    if device_str in ["cuda", "sycl"]:
        xgb_params["tree_method"] = "hist"
        xgb_params["device"] = device_str

    classifier = xgb.XGBClassifier(**xgb_params)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)[:, 1]

    metrics = {
        "f1": float(f1_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_prob))
        if len(np.unique(y_test)) > 1
        else 0.0,
    }

    logger.info(
        f"Trained Churn Predictor. F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}"
    )

    y_test_df = y_test.to_frame()
    return classifier, metrics, X_test, y_test_df


def predict_churn(data: pd.DataFrame, model: xgb.XGBClassifier) -> pd.DataFrame:
    """Predict Churn Probability and add to DataFrame.

    Args:
        data: Input DataFrame with features.
        model: Trained XGBoost classifier.

    Returns:
        DataFrame with Predicted_Churn_Prob and Churn_Risk_Level.
    """
    if model is None or not hasattr(model, "feature_names_in_"):
        logger.warning("No model provided for churn prediction. Skipping.")
        return data

    feature_cols = model.feature_names_in_.tolist()
    missing_features = [c for c in feature_cols if c not in data.columns]

    if missing_features:
        logger.error(f"Missing features for churn prediction: {missing_features}")
        return data

    logger.info("Running churn inference...")
    data["Predicted_Churn_Prob"] = model.predict_proba(data[feature_cols])[:, 1]

    # Categorize Risk
    data["Churn_Risk_Level"] = pd.cut(
        data["Predicted_Churn_Prob"],
        bins=[0, 0.3, 0.7, 1.0],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    return data


def explain_predictions(
    model: xgb.XGBClassifier, X_test: pd.DataFrame
) -> Dict[str, Any]:
    """Calculate SHAP values for model explainability."""
    if model is None or X_test.empty or not hasattr(model, "feature_names_in_"):
        logger.warning("Skipping SHAP explanations: No model or test data provided.")
        return {"status": "skipped", "reason": "model_disabled_or_insufficient_data"}

    logger.info("Calculating SHAP explanations for Churn model...")

    try:
        X_np = X_test.values
        feature_names = X_test.columns.tolist()

        # Ensure background is representative
        if len(X_np) > 50:
            background = shap.kmeans(X_np, 10).data
        else:
            background = X_np

        def predict_fn(x):
            # SHAP KernelExplainer expects (samples, n_classes) for probability models
            return model.predict_proba(x)

        explainer = shap.KernelExplainer(predict_fn, background)
        shap_out = explainer.shap_values(X_np, silent=True)

        # Handle binary classification output shape (samples, features, 2) or [list of arrays]
        if isinstance(shap_out, list) and len(shap_out) > 1:
            vals = shap_out[1]
            base_val = explainer.expected_value[1]
        elif isinstance(shap_out, np.ndarray) and len(shap_out.shape) == 3:
            vals = shap_out[:, :, 1]
            base_val = explainer.expected_value[1]
        else:
            vals = shap_out
            base_val = explainer.expected_value

        def safe_float(v):
            if isinstance(v, (list, np.ndarray)):
                return float(np.array(v).flatten()[0])
            return float(v)

        return {
            "shap_values": np.array(vals).tolist(),
            "base_value": safe_float(base_val),
            "feature_names": feature_names,
            "instance_data": X_np.tolist(),
        }
    except Exception as e:
        logger.error(f"SHAP explanation failed: {e}", exc_info=True)
        return {"status": "error", "reason": str(e)}
