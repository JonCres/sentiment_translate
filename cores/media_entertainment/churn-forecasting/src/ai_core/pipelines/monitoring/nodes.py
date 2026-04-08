import logging
from typing import Dict, Any
import polars as pl
import mlflow

logger = logging.getLogger(__name__)


def monitor_data_drift(data: pl.DataFrame, params: Dict[str, Any]) -> Dict[str, float]:
    """
    Check for data drift by comparing current data stats with thresholds.
    Basic implementation checking if key statistics are within range.

    Logs drift metrics to MLflow if an active run exists.
    """
    logger.info("Running data drift detection...")
    metrics = {}

    # Example: Check if average purchase value is within expected range
    if "monetary_value" in data.columns:
        mean_monetary = data["monetary_value"].mean()
        metrics["mean_monetary_value"] = mean_monetary

        # Thresholds could be in params, here specific hardcoded or placeholder
        if mean_monetary is not None and mean_monetary < 0:
            logger.warning(f"Mean monetary value is negative: {mean_monetary}")

    if "frequency" in data.columns:
        mean_freq = data["frequency"].mean()
        metrics["mean_frequency"] = mean_freq

    # Log to MLflow if run is active
    if mlflow.active_run():
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and value is not None:
                mlflow.log_metric(f"drift_{key}", float(value))
        logger.info(f"Logged {len(metrics)} drift metrics to MLflow")

    return metrics


def monitor_model_performance(
    data: pl.DataFrame, params: Dict[str, Any]
) -> Dict[str, float]:
    """
    Monitor model performance if actuals are available.
    For this setup, we might just log predicted characteristics.

    Logs performance metrics to MLflow if an active run exists.
    """
    logger.info("Running model performance monitoring...")
    metrics = {}

    if "churn_prob_30day" in data.columns:
        mean_churn_prob = data["churn_prob_30day"].mean()
        metrics["mean_predicted_churn_prob"] = mean_churn_prob
        logger.info(f"Mean Predicted Churn Prob (30d): {mean_churn_prob}")

    # Log to MLflow if run is active
    if mlflow.active_run():
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and value is not None:
                mlflow.log_metric(f"perf_{key}", float(value))
        logger.info(f"Logged {len(metrics)} performance metrics to MLflow")

    return metrics
