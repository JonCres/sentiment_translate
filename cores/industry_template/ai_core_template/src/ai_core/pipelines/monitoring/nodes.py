import logging
from typing import Dict, Any
import pandas as pd
import mlflow

logger = logging.getLogger(__name__)


def monitor_data_drift(data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, float]:
    """
    Check for data drift by comparing current data stats with thresholds.
    Basic implementation checking if key statistics are within range.
    
    Logs drift metrics to MLflow if an active run exists.
    """
    logger.info("Running data drift detection...")
    metrics = {}

    # Placeholder logic for template
    for col in data.select_dtypes(include=['number']).columns:
        mean_val = data[col].mean()
        metrics[f"mean_{col}"] = mean_val
        logger.info(f"Mean of {col}: {mean_val}")

    # Log to MLflow if run is active
    if mlflow.active_run():
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and value is not None:
                mlflow.log_metric(f"drift_{key}", float(value))
        logger.info(f"Logged {len(metrics)} drift metrics to MLflow")

    return metrics


def monitor_model_performance(
    predictions: pd.DataFrame, params: Dict[str, Any]
) -> Dict[str, float]:
    """
    Monitor model performance if actuals are available.
    
    Logs performance metrics to MLflow if an active run exists.
    """
    logger.info("Running model performance monitoring...")
    metrics = {}

    # Placeholder logic for template
    if not predictions.empty:
        # Assuming predictions might have a 'target' and 'prediction' column
        # or just logging some characteristics
        metrics["n_predictions"] = len(predictions)
        logger.info(f"Monitoring performance for {len(predictions)} records")

    # Log to MLflow if run is active
    if mlflow.active_run():
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and value is not None:
                mlflow.log_metric(f"perf_{key}", float(value))
        logger.info(f"Logged {len(metrics)} performance metrics to MLflow")

    return metrics

