import logging
from typing import Any, Dict
import matplotlib.pyplot as plt
import pandas as pd
import shap
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_shap_summary(shap_data: Dict[str, Any]) -> Any:
    """
    Generates a SHAP summary plot for survival risk.
    """
    if not shap_data:
        logger.warning("No SHAP data available to plot.")
        return None

    shap_values = shap_data.get("shap_values")
    X_sample = shap_data.get("sample_data")

    if shap_values is None:
        return None

    try:
        f = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title("SHAP Feature Importance (Survival Risk)")
        plt.tight_layout()
        plt.close(f)
        return f
    except Exception as e:
        logger.error(f"Failed to plot SHAP summary: {e}")
        return None


def plot_survival_curves(
    model: Any, data: pd.DataFrame, parameters: Dict[str, Any]
) -> Any:
    """
    Visualizes survival curves S(t) for individual users or segments.
    """
    logger.info("Plotting survival curves...")
    X = data.drop(columns=["duration", "event"], errors="ignore").select_dtypes(
        include=[np.number]
    )

    # Sample 5 users to plot
    X_plot = X.head(5)

    try:
        f = plt.figure(figsize=(10, 6))

        if hasattr(model, "predict_survival_function"):
            # lifelines / sksurv style
            surv_funcs = model.predict_survival_function(X_plot)
            for i, surv in enumerate(surv_funcs):
                label = f"User {X_plot.index[i]}"
                if isinstance(surv, pd.Series):
                    plt.step(surv.index, surv.values, where="post", label=label)
                else:
                    # sksurv returns an array of functions or similar
                    plt.step(surv.x, surv.y, where="post", label=label)
        else:
            # Fallback/DeepSurv style might need different handling
            logger.warning("Model does not support predict_survival_function directly.")

        plt.ylim(0, 1.05)
        plt.xlabel("Days (t)")
        plt.ylabel("Survival Probability S(t)")
        plt.title("Individual Survival Curves")
        plt.legend()
        plt.grid(True)
        plt.close(f)
        return f
    except Exception as e:
        logger.error(f"Failed to plot survival curves: {e}")
        return None


def evaluate_models(
    test_data: pd.DataFrame,
    parameters: Dict[str, Any],
    cox_ph_model: Any = None,
    rsf_model: Any = None,
    deepsurv_model: Any = None,
    nmtlr_model: Any = None,
) -> pd.DataFrame:
    """
    Evaluates survival models using C-index.
    """
    models = {
        "CoxPH": cox_ph_model,
        "RSF": rsf_model,
        "DeepSurv": deepsurv_model,
        "NMTLR": nmtlr_model,
    }

    if test_data.empty:
        return pd.DataFrame()

    # Pre-calculate C-index for available models
    metrics_list = []

    # For survival, we need duration and event
    y_test = np.array(
        [(bool(e), d) for e, d in zip(test_data["event"], test_data["duration"])],
        dtype=[("event", bool), ("duration", float)],
    )
    X_test = test_data.drop(columns=["duration", "event"]).select_dtypes(
        include=[np.number]
    )

    for name, model in models.items():
        if model is None:
            continue

        try:
            # Different models have different scoring
            if name == "CoxPH":
                c_index = model.concordance_index_
            elif hasattr(model, "score"):
                c_index = model.score(X_test, y_test)
            else:
                c_index = 0.5  # Placeholder

            metrics_list.append({"Model": name, "C-Index": c_index})
        except Exception as e:
            logger.warning(f"Failed to evaluate {name}: {e}")

    return pd.DataFrame(metrics_list)


def plot_model_comparison(metrics: pd.DataFrame) -> Any:
    """
    Plots a bar chart comparing model performance.
    """
    if metrics.empty:
        return None

    f = plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics, x="Model", y="C-Index")
    plt.title("Survival Model Comparison (C-Index)")
    plt.ylim(0.4, 1.0)
    plt.close(f)
    return f
