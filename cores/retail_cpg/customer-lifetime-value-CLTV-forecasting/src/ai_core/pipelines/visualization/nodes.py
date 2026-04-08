import logging
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes.plotting import (
    plot_frequency_recency_matrix,
    plot_probability_alive_matrix,
    plot_period_transactions,
)
import polars as pl
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def plot_lifetimes_metrics(
    bg_nbd_model: BetaGeoFitter, data: pl.DataFrame, params: Dict[str, Any]
):
    """
    Generate and save CLTV specific plots:
    - Frequency/Recency Matrix
    - Probability Alive Matrix
    - Expected Number of Future Transactions (Period Transactions)
    """
    output_dir = Path(params.get("plots_dir", "data/08_reporting/plots"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Frequency/Recency Matrix
    if params.get("frequency_recency_matrix", True):
        plt.figure(figsize=(10, 8))
        plot_frequency_recency_matrix(bg_nbd_model)
        plt.title("Frequency-Recency Matrix")
        plt.savefig(output_dir / "frequency_recency_matrix.png")
        plt.close()
        logger.info("Saved frequency_recency_matrix.png")

    # 2. Probability Alive Matrix
    if params.get("probability_alive_matrix", True):
        plt.figure(figsize=(10, 8))
        plot_probability_alive_matrix(bg_nbd_model)
        plt.title("Probability Alive Matrix")
        plt.savefig(output_dir / "probability_alive_matrix.png")
        plt.close()
        logger.info("Saved probability_alive_matrix.png")

    # 3. Period Transactions (Model Fit)
    if params.get("period_transactions_plot", True):
        plt.figure(figsize=(10, 8))
        plot_period_transactions(bg_nbd_model)
        plt.title("Expected vs Actual Frequency of Repeat Transactions")
        plt.savefig(output_dir / "period_transactions.png")
        plt.close()
        logger.info("Saved period_transactions.png")

def calculate_business_impact_kpis(
    predictions: pl.DataFrame, params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate high-level Business Impact KPIs for reporting.
    """
    baselines = params.get("baselines", {})
    
    # 1. Average CLTV
    avg_cltv = predictions["clv_12mo"].mean()
    
    # 2. Retention Rate (High-Value)
    # Define high value as Whale or High-Value from our simplified segments
    high_value_mask = pl.col("clv_segment").is_in(["Whale", "High-Value"])
    high_value_df = predictions.filter(high_value_mask)
    
    # Use len() instead of .count() to avoid ambiguous truth value error
    n_high_value = len(high_value_df)
    if n_high_value > 0:
        # P(Alive) > 0.8 as a proxy relative to others
        retained_count = high_value_df.filter(pl.col("p_alive") > 0.8).height
        retention_high_value = retained_count / n_high_value
    else:
        retention_high_value = 0.0

    # 3. Acquisition Efficiency (CAC:CLV)
    # Assuming CAC is provided or global
    cac = params.get("cac", 130) # Default from example ratio maybe
    acquisition_efficiency = cac / avg_cltv if avg_cltv > 0 else 0.0

    # 4. Segment Concentration (Pareto)
    total_rev = predictions["clv_12mo"].sum()
    top_20_count = int(len(predictions) * 0.2)
    top_20_rev = predictions.sort("clv_12mo", descending=True).head(top_20_count)["clv_12mo"].sum()
    segment_concentration = top_20_rev / total_rev if total_rev > 0 else 0.0

    # 5. Incremental ROI (Example)
    # (Optimized CLV - Baseline CLV) * num_customers
    incremental_value = (avg_cltv - baselines.get("avg_cltv", 325)) * len(predictions)

    kpis = {
        "avg_cltv": avg_cltv,
        "retention_high_value": retention_high_value,
        "acquisition_efficiency": acquisition_efficiency,
        "segment_concentration": segment_concentration,
        "incremental_value": incremental_value,
        "customer_count": len(predictions)
    }
    
    logger.info(f"Calculated Business Impact KPIs: {kpis}")
    return kpis


def plot_business_impact_kpis(kpis: Dict[str, Any], params: Dict[str, Any]):
    """
    Generate visualizations for KPIs.
    """
    output_dir = Path(params.get("plots_dir", "data/08_reporting/plots"))
    output_dir.mkdir(parents=True, exist_ok=True)
    baselines = params.get("baselines", {})

    # 1. Comparison with Baseline (Average CLTV)
    plt.figure(figsize=(8, 6))
    plt.bar(["Baseline", "Target (Predicted)"], [baselines.get("avg_cltv", 325), kpis["avg_cltv"]], color=["grey", "blue"])
    plt.title("Customer Lifetime Value: Baseline vs Target")
    plt.ylabel("Avg CLTV ($)")
    plt.savefig(output_dir / "kpi_cltv_comparison.png")
    plt.close()

    # 2. Pareto Chart (Concentration)
    # (Simple version for now)
    plt.figure(figsize=(8, 6))
    plt.bar(["Top 20%", "Other 80%"], [kpis["segment_concentration"], 1 - kpis["segment_concentration"]], color=["gold", "silver"])
    plt.title("Revenue Concentration (Pareto Efficiency)")
    plt.ylabel("% of Total Predicted Value")
    plt.savefig(output_dir / "kpi_pareto_concentration.png")
    plt.close()
