"""
MLflow Tracking Utilities for AI Core Pipelines.

This module provides centralized functions for MLflow experiment tracking,
model registration, and Prefect-MLflow run linking.

Usage:
    from src.utils.mlflow_tracking import setup_mlflow_tracking, start_mlflow_run

    # Configure MLflow from parameters
    setup_mlflow_tracking(mlops_config)

    # Start a run with Prefect linking
    with mlflow.start_run():
        link_prefect_run_id()
        # ... pipeline execution ...
"""

import logging
from typing import Dict, Any, Optional

import mlflow

logger = logging.getLogger(__name__)


def setup_mlflow_tracking(mlops_params: Dict[str, Any]) -> None:
    """
    Configure MLflow tracking from pipeline parameters.

    Args:
        mlops_params: Dictionary containing 'tracking_uri' and 'experiment_name'.
    """
    tracking_uri = mlops_params.get("tracking_uri", "http://localhost:5000")
    experiment_name = mlops_params.get("experiment_name", "default_experiment")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    logger.info(f"MLflow tracking configured: URI={tracking_uri}, Experiment={experiment_name}")


def start_mlflow_run(
    run_name: Optional[str] = None,
    nested: bool = False,
    tags: Optional[Dict[str, str]] = None
) -> str:
    """
    Start an MLflow run with optional Prefect context linking.

    Args:
        run_name: Optional name for the run.
        nested: Whether this is a nested run (for sub-pipelines).
        tags: Additional tags to set on the run.

    Returns:
        The MLflow run_id.
    """
    run = mlflow.start_run(run_name=run_name, nested=nested)

    # Link Prefect flow_run_id if running in Prefect context
    link_prefect_run_id()

    # Set additional tags
    if tags:
        for key, value in tags.items():
            mlflow.set_tag(key, value)

    logger.info(f"MLflow run started: {run.info.run_id}")
    return run.info.run_id


def link_prefect_run_id() -> Optional[str]:
    """
    Link the current Prefect flow_run_id to the active MLflow run.

    Returns:
        The Prefect flow_run_id if successfully linked, None otherwise.
    """
    try:
        from prefect.context import get_run_context

        run_context = get_run_context()
        prefect_id = str(run_context.flow_run.id)
        mlflow.set_tag("prefect_flow_run_id", prefect_id)
        logger.info(f"Linked Prefect flow_run_id: {prefect_id}")
        return prefect_id
    except Exception:
        # Not running in Prefect context or import failed
        return None


def log_monitoring_metrics(metrics: Dict[str, Any], prefix: str = "") -> None:
    """
    Log monitoring metrics to the active MLflow run.

    Args:
        metrics: Dictionary of metric name to value.
        prefix: Optional prefix for metric names (e.g., 'drift_').
    """
    if not mlflow.active_run():
        logger.warning("No active MLflow run. Metrics not logged.")
        return

    for key, value in metrics.items():
        if isinstance(value, (int, float)) and value is not None:
            metric_name = f"{prefix}{key}" if prefix else key
            mlflow.log_metric(metric_name, float(value))
            logger.debug(f"Logged metric: {metric_name}={value}")


def log_model_with_registration(
    model: Any,
    artifact_path: str,
    registered_model_name: str,
    flavor: str = "sklearn"
) -> None:
    """
    Log a model to MLflow with optional Model Registry registration.

    Args:
        model: The trained model object.
        artifact_path: Path within the artifact store.
        registered_model_name: Name for the Model Registry.
        flavor: MLflow flavor ('sklearn', 'xgboost', 'pytorch', etc.).
    """
    if not mlflow.active_run():
        logger.warning("No active MLflow run. Model not logged.")
        return

    try:
        if flavor == "sklearn":
            import mlflow.sklearn
            mlflow.sklearn.log_model(
                model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name
            )
        elif flavor == "xgboost":
            import mlflow.xgboost
            mlflow.xgboost.log_model(
                model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name
            )
        else:
            # Generic fallback
            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=model,
                registered_model_name=registered_model_name
            )

        logger.info(f"Model registered: {registered_model_name} at {artifact_path}")
    except Exception as e:
        logger.error(f"Failed to log model {registered_model_name}: {e}")


def end_mlflow_run(status: str = "FINISHED") -> None:
    """
    End the active MLflow run with the specified status.

    Args:
        status: Run status ('FINISHED', 'FAILED', 'KILLED').
    """
    if mlflow.active_run():
        mlflow.end_run(status=status)
        logger.info(f"MLflow run ended with status: {status}")
