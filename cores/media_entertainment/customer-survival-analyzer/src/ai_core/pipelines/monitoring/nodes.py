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
    Monitor survival model performance.

    Logs hazard metrics to MLflow if an active run exists.
    """
    logger.info("Running survival performance monitoring...")
    metrics = {}

    if "churn_prob_30day" in data.columns:
        mean_risk_30 = data["churn_prob_30day"].mean()
        metrics["mean_churn_prob_30day"] = mean_risk_30
        logger.info(f"Mean Predicted 30-day Churn Risk: {mean_risk_30}")

    if "churn_prob_90day" in data.columns:
        mean_risk_90 = data["churn_prob_90day"].mean()
        metrics["mean_churn_prob_90day"] = mean_risk_90
        logger.info(f"Mean Predicted 90-day Churn Risk: {mean_risk_90}")

    if "predicted_median_tenure_days" in data.columns:
        mean_tenure = data["predicted_median_tenure_days"].mean()
        metrics["mean_predicted_median_tenure"] = mean_tenure
        logger.info(f"Mean Predicted Median Tenure: {mean_tenure} days")

    # Log to MLflow if run is active
    if mlflow.active_run():
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and value is not None:
                mlflow.log_metric(f"perf_{key}", float(value))
        logger.info(f"Logged {len(metrics)} performance metrics to MLflow")

    return metrics
