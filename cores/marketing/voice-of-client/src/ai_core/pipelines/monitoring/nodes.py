import logging
from typing import Dict, Any
import mlflow

logger = logging.getLogger(__name__)


def monitor_data_drift(data: Any, params: Dict[str, Any]) -> Dict[str, float]:
    """
    Check for data drift by comparing current data stats with thresholds.
    Basic implementation checking if key statistics are within range.

    Logs drift metrics to MLflow if an active run exists.
    """
    logger.info("Running data drift detection...")
    metrics = {}

    # Identify numeric columns - handle both Pandas and Polars
    if hasattr(data, "select_dtypes"):
        # Pandas
        numeric_cols = data.select_dtypes(include=["number"]).columns
    else:
        # Polars
        import polars as pl

        numeric_cols = [
            col
            for col in data.columns
            if data.schema[col]
            in [pl.Int64, pl.Float64, pl.Int32, pl.Float32, pl.Decimal]
        ]

    for col in numeric_cols:
        mean_val = data[col].mean()
        # Handle the case where mean() might return a Series (Polars) or scalar
        if hasattr(mean_val, "item"):
            mean_val = mean_val.item()

        metrics[f"mean_{col}"] = float(mean_val) if mean_val is not None else 0.0
        logger.info(f"Mean of {col}: {mean_val}")

    # Log to MLflow if run is active
    if mlflow.active_run():
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and value is not None:
                mlflow.log_metric(f"drift_{key}", float(value))
        logger.info(f"Logged {len(metrics)} drift metrics to MLflow")

    return metrics


def monitor_model_performance(
    predictions: Any, params: Dict[str, Any]
) -> Dict[str, float]:
    """
    Monitor model performance if actuals are available.

    Logs performance metrics to MLflow if an active run exists.
    """
    logger.info("Running model performance monitoring...")
    metrics = {}

    # Identify if empty - handle both Pandas and Polars
    is_empty = False
    if hasattr(predictions, "empty"):
        is_empty = predictions.empty
    elif hasattr(predictions, "is_empty"):
        is_empty = predictions.is_empty()
    else:
        is_empty = len(predictions) == 0

    if not is_empty:
        # Assuming predictions might have a 'target' and 'prediction' column
        # or just logging some characteristics
        metrics["n_predictions"] = float(len(predictions))
        logger.info(f"Monitoring performance for {len(predictions)} records")

    # Log to MLflow if run is active
    if mlflow.active_run():
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and value is not None:
                mlflow.log_metric(f"perf_{key}", float(value))
        logger.info(f"Logged {len(metrics)} performance metrics to MLflow")

    return metrics
