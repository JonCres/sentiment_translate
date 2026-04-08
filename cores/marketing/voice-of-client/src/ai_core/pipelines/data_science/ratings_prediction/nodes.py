import logging
import pandas as pd
import xgboost as xgb
from typing import Dict, Any, Tuple
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from utils.device import get_device

logger = logging.getLogger(__name__)


def predict_nps_and_csat(
    data: pd.DataFrame, parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Predict NPS and CSAT scores if they are missing or for benchmarking.

    Args:
        data: DataFrame with sentiment and emotion features.
        parameters: AI Modeling parameters.

    Returns:
        DataFrame with predicted scores and a dictionary of models/metrics.
    """
    ratings_params = parameters.get("ratings", {})
    nps_enabled = ratings_params.get("nps_prediction", {}).get("enabled", True)
    csat_enabled = ratings_params.get("csat_prediction", {}).get("enabled", True)

    if not nps_enabled and not csat_enabled:
        logger.info("NPS and CSAT prediction are disabled.")
        return data, {"status": "disabled"}

    # Identify features: sentiment from ABSA and emotions from MER
    sentiment_cols = [c for c in data.columns if c.startswith("sent_")]
    emotion_cols = [c for c in data.columns if c.startswith("emo_")]
    feature_cols = sentiment_cols + emotion_cols

    if not feature_cols:
        logger.warning("No sentiment or emotion features found for rating prediction.")
        return data, {"status": "no_features"}

    results_data = data.copy()
    models_metrics = {}

    device_str = get_device(purpose="ratings prediction", framework="xgboost")
    xgb_base_params = {
        "n_estimators": ratings_params.get("n_estimators", 200),
        "learning_rate": ratings_params.get("learning_rate", 0.05),
        "max_depth": ratings_params.get("max_depth", 6),
        "objective": "reg:squarederror",
        "random_state": 42,
    }
    if device_str in ["cuda", "sycl"]:
        xgb_base_params["tree_method"] = "hist"
        xgb_base_params["device"] = device_str

    for target, enabled, target_range in [
        ("NPS_Score", nps_enabled, (0, 10)),
        ("CSAT_Score", csat_enabled, (1, 5)),
    ]:
        if not enabled:
            continue

        pred_col = f"Predicted_{target}"
        norm_col = f"{target.lower()}_norm"

        # Check if target exists and has enough non-null values to train
        if target in data.columns and data[target].notna().sum() > 20:
            logger.info(f"Training prediction model for {target}...")
            train_df = data[data[target].notna()]
            X = train_df[feature_cols]
            y = train_df[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = xgb.XGBRegressor(**xgb_base_params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            metrics = {
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred)),
            }
            models_metrics[target] = metrics
            logger.info(
                f"{target} Model - MAE: {metrics['mae']:.4f}, R2: {metrics['r2']:.4f}"
            )

            # General prediction
            results_data[pred_col] = model.predict(results_data[feature_cols]).clip(
                *target_range
            )

            # Fill missing originals with predictions
            results_data[target] = results_data[target].fillna(results_data[pred_col])
        else:
            logger.warning(
                f"Insufficient data to train {target} model. Using heuristic based on sentiment."
            )
            # Heuristic: map average sentiment to the range
            # avg_sentiment is now 1-5 stars.
            avg_sentiment = (
                results_data[sentiment_cols].mean(axis=1)
                if sentiment_cols
                else pd.Series(3.0, index=results_data.index)
            )

            if target == "NPS_Score":
                # Map 1-5 to 0-10: (score - 1) / 4 * 10
                results_data[pred_col] = ((avg_sentiment - 1) / 4.0 * 10).clip(0, 10)
            else:
                # Map 1-5 to 1-5 (direct)
                results_data[pred_col] = avg_sentiment.clip(1, 5)

            if target not in results_data.columns:
                results_data[target] = results_data[pred_col]
            else:
                results_data[target] = results_data[target].fillna(
                    results_data[pred_col]
                )

            models_metrics[target] = {"status": "heuristic"}

        # Normalize to 0-1 for downstream tasks
        if target == "NPS_Score":
            results_data[norm_col] = (results_data[target] / 10.0).clip(0, 1)
        else:
            results_data[norm_col] = ((results_data[target] - 1) / 4.0).clip(0, 1)

    return results_data, models_metrics
