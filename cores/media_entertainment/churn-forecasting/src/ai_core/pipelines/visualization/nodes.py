import logging
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import pandas as pd
import shap
import lime
import lime.lime_tabular
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    accuracy_score,
)
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_shap_summary(shap_data: Dict[str, Any]) -> Any:
    """
    Generates a SHAP summary plot.
    """
    if not shap_data:
        logger.warning("No SHAP data available to plot.")
        return None

    shap_values = shap_data.get("shap_values")
    X_sample = shap_data.get("sample_data")

    if shap_values is None:
        return None

    try:
        # Create a figure
        f = plt.figure()
        # Summary plot
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.close(f)
        return f
    except Exception as e:
        logger.error(f"Failed to plot SHAP summary: {e}")
        return None


def plot_lime_explanation(
    model: Any,
    data: pd.DataFrame,  # Should be the X dataset
    instance_index: int = 0,
) -> Any:
    """
    Generates a LIME explanation for a single instance.
    This might be better suited for the Streamlit app directly, but we can generate a static plot here for reporting.
    """
    # ... Implementation optional for pipeline, critical for App ...
    pass


def evaluate_models(
    test_data: pd.DataFrame,
    parameters: Dict[str, Any],
    xgboost_model: Any = None,
    random_forest_model: Any = None,
    lightgbm_model: Any = None,
    catboost_model: Any = None,
    ensemble_model: Any = None,
) -> pd.DataFrame:
    """
    Evaluates all available models on test set and returns metrics.
    """
    models = {
        "XGBoost": xgboost_model,
        "RandomForest": random_forest_model,
        "LightGBM": lightgbm_model,
        "CatBoost": catboost_model,
        "Ensemble": ensemble_model,
    }

    xgb_params = parameters.get("modeling", {}).get("xgboost", {})
    target_col = xgb_params.get("xgboost_target_col", "churn_label")

    if test_data.empty:
        return pd.DataFrame()

    X_test = test_data.drop(columns=[target_col], errors="ignore")
    y_test = test_data[target_col]

    metrics_list = []

    for name, model in models.items():
        if model is None:
            continue

        try:
            # Handle PyMC models differently if passed (not expected here, but good practice)
            # Assuming scikit-learn interface for all
            y_pred = model.predict(X_test)
            y_proba = (
                model.predict_proba(X_test)[:, 1]
                if hasattr(model, "predict_proba")
                else y_pred
            )

            auc = roc_auc_score(y_test, y_proba)
            acc = accuracy_score(y_test, y_pred)

            metrics_list.append({"Model": name, "AUC": auc, "Accuracy": acc})
        except Exception as e:
            logger.warning(f"Failed to evaluate {name}: {e}")

    return pd.DataFrame(metrics_list)


def plot_model_comparison(
    test_data: pd.DataFrame,
    parameters: Dict[str, Any],
    xgboost_model: Any = None,
    random_forest_model: Any = None,
    lightgbm_model: Any = None,
    catboost_model: Any = None,
    ensemble_model: Any = None,
) -> Any:
    """
    Plots ROC curves for all models.
    """
    models = {
        "XGBoost": xgboost_model,
        "RandomForest": random_forest_model,
        "LightGBM": lightgbm_model,
        "CatBoost": catboost_model,
        "Ensemble": ensemble_model,
    }

    xgb_params = parameters.get("modeling", {}).get("xgboost", {})
    target_col = xgb_params.get("xgboost_target_col", "churn_label")

    if test_data.empty:
        return None

    X_test = test_data.drop(columns=[target_col], errors="ignore")
    y_test = test_data[target_col]

    f = plt.figure(figsize=(10, 8))

    for name, model in models.items():
        if model is None:
            continue

        try:
            y_proba = (
                model.predict_proba(X_test)[:, 1]
                if hasattr(model, "predict_proba")
                else model.predict(X_test)
            )
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
        except Exception as e:
            logger.warning(f"Failed to plot ROC for {name}: {e}")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Model Comparison - ROC Curves")
    plt.legend()
    plt.close(f)
    return f
