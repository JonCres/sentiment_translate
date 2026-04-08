"""Visualization nodes for generating charts and plots."""

import logging
import os
import gc
import torch
from pathlib import Path
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import (
    plot_frequency_recency_matrix,
    plot_probability_alive_matrix,
    plot_period_transactions,
)
import polars as pl
import pandas as pd
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

logger = logging.getLogger(__name__)

# Try to import Groq, make it optional
try:
    from groq import Groq

    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq not installed. LLM interpretation will be unavailable.")


def clear_device_cache():
    """Release unoccupied cached memory for available devices (CUDA, XPU, MPS)."""
    # Force Python garbage collection first
    gc.collect()

    try:
        # CUDA (NVIDIA)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # XPU (Intel)
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()

        # MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception as e:
        # Log warning but don't crash processing
        logger.warning(f"Failed to clear device cache: {e}")


def plot_lifetimes_metrics(
    bg_nbd_model: BetaGeoFitter, data: pl.DataFrame, params: Dict[str, Any]
) -> None:
    """Generate and save diagnostic plots for BG/NBD model validation and interpretation.

    Creates three standard lifetimes library visualizations to assess model fit quality
    and understand customer behavior patterns. These plots are essential for validating
    that the BG/NBD model has converged properly and makes reasonable predictions.

    Generated Plots:
    1. **Frequency-Recency Matrix**: Heatmap showing customer distribution across
       purchase frequency and recency dimensions. Reveals customer cohort patterns.
    2. **Probability Alive Matrix**: Heatmap showing P(Alive) conditional on frequency
       and recency. Helps identify dormant vs active customer zones.
    3. **Period Transactions**: Model fit diagnostic comparing predicted vs actual
       repeat purchase counts over time. Validates model calibration.

    Args:
        bg_nbd_model: Fitted BetaGeoFitter model to visualize
        data: RFM summary DataFrame (not directly used by lifetimes plots but
            available for custom visualizations)
        params: Configuration dictionary containing:
            - plots_dir (str): Output directory path (default: 'data/08_reporting/plots')
            - frequency_recency_matrix (bool): Generate frequency-recency plot (default: True)
            - probability_alive_matrix (bool): Generate probability alive plot (default: True)
            - period_transactions_plot (bool): Generate model fit diagnostic (default: True)

    Returns:
        None. Saves PNG files to plots_dir:
            - frequency_recency_matrix.png
            - probability_alive_matrix.png
            - period_transactions.png

    Note:
        - Creates plots_dir if it doesn't exist
        - Closes matplotlib figures after saving to prevent memory leaks
        - Logs confirmation message after each plot is saved

    Example:
        >>> params = {
        ...     'plots_dir': 'data/08_reporting/plots',
        ...     'frequency_recency_matrix': True
        ... }
        >>> plot_lifetimes_metrics(bgf_model, rfm_data, params)
        INFO:Saved frequency_recency_matrix.png
    """
    output_dir = Path(params.get("plots_dir", "data/08_reporting/plots"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure model has the data attached for plotting functions
    # This is strictly required for lifetimes plotting utilities, and can be lost during MLflow logging
    if not hasattr(bg_nbd_model, "data") or bg_nbd_model.data is None:
        logger.info("Re-attaching data to BG/NBD model for plotting...")
        bg_nbd_model.data = data.to_pandas()

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
        plt.close()
        logger.info("Saved period_transactions.png")


def plot_strategic_kpis(cltv_predictions: pl.DataFrame, params: Dict[str, Any]) -> None:
    """Generate executive-level KPI visualizations for business stakeholders.

    Creates four strategic dashboards that translate CLTV predictions into actionable
    business metrics for marketing, finance, and product teams. These plots support
    data-driven decision making for customer acquisition, retention, and monetization.

    Generated KPI Dashboards:
    1. **Revenue per Customer**: Bar chart comparing average CLTV vs top segment (Q5_Elite)
       - Quantifies value concentration in high-value customers
    2. **Acquisition Efficiency**: CAC vs LTV comparison with ratio overlay
       - Validates unit economics (target: LTV:CAC ≥ 3:1)
    3. **Revenue by Segment**: Total revenue potential per customer segment (Q1-Q5)
       - Identifies which segments drive most value
    4. **Churn Risk Distribution**: Histogram of 30-day churn probabilities
       - Shows proportion of customers at immediate risk

    Args:
        cltv_predictions: CLTV prediction DataFrame with columns:
            - clv_12mo or predicted_cltv (float): 12-month CLTV forecast
            - clv_segment (str, optional): Customer segment (Q1_Low to Q5_Elite)
            - p_churn_30day (float, optional): 30-day churn probability (0-1)
            - customer_id (str): Customer identifier
        params: Configuration dictionary containing:
            - plots_dir (str): Output directory path (default: 'data/08_reporting/plots')
            - cac_value (float): Customer Acquisition Cost for LTV:CAC calculation (default: $50)

    Returns:
        None. Saves PNG files to plots_dir:
            - kpi_revenue_per_customer.png
            - kpi_acquisition_efficiency.png
            - kpi_revenue_by_segment.png (if clv_segment available)
            - kpi_churn_risk_dist.png (if p_churn_30day available)

    Raises:
        Warning (logged): If required columns (clv_12mo/predicted_cltv) are missing

    Note:
        - Automatically creates plots_dir if it doesn't exist
        - Uses top 20% of customers if clv_segment column unavailable
        - Color palette optimized for accessibility (colorblind-friendly)
        - Logs confirmation messages after saving each plot

    Business Interpretation:
        - **Revenue per Customer**: High ratio (>5x) indicates whale dependency risk
        - **LTV:CAC Ratio**: <3:1 suggests unprofitable acquisition; >5:1 indicates
          under-investment in growth
        - **Revenue by Segment**: Imbalanced distribution signals retention opportunity
        - **Churn Risk**: Right-skewed distribution indicates systemic churn issue

    Example:
        >>> params = {'plots_dir': 'data/08_reporting/plots', 'cac_value': 75.0}
        >>> plot_strategic_kpis(cltv_predictions, params)
        INFO:Saved Strategic KPI plots including Segment Revenue and Churn Risk.
    """
    valid_cols = ["predicted_cltv", "clv_12mo"]
    if not any(col in cltv_predictions.columns for col in valid_cols):
        logger.warning(
            f"CLTV Predictions missing one of {valid_cols}. Skipping KPI plots."
        )
        return

    output_dir = Path(params.get("plots_dir", "data/08_reporting/plots"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Needs pandas for plotting usually
    df = cltv_predictions.to_pandas()

    # 1. Revenue Per Customer (Overall vs High Value)
    plt.figure(figsize=(10, 6))
    if "clv_12mo" in df.columns:
        clv_col = "clv_12mo"
    else:
        clv_col = "predicted_cltv"  # Fallback

    avg_cltv = df[clv_col].mean()
    # High Value segment (Q5_Elite or top 20%)
    if "clv_segment" in df.columns:
        top_segment_avg = df[df["clv_segment"] == "Q5_Elite"][clv_col].mean()
    else:
        high_value_cut = df[clv_col].quantile(0.8)
        top_segment_avg = df[df[clv_col] >= high_value_cut][clv_col].mean()

    plt.bar(
        ["Average Customer", "Top Segment (Elite)"],
        [avg_cltv, top_segment_avg],
        color=["#667eea", "#764ba2"],
    )
    plt.title("Strategic Impact: Revenue per Customer")
    plt.ylabel("Predicted CLTV ($)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(output_dir / "kpi_revenue_per_customer.png")
    plt.close()

    # 2. Acquisition Efficiency (CAC vs LTV)
    cac = params.get("cac_value", 50.0)
    ltv_cac_ratio = avg_cltv / cac if cac > 0 else 0

    plt.figure(figsize=(8, 6))
    bars = plt.bar(
        ["CAC Cost", "Avg CLTV"], [cac, avg_cltv], color=["#e53e3e", "#3182ce"]
    )
    plt.title(f"Acquisition Efficiency (LTV:CAC = {ltv_cac_ratio:.1f}:1)")
    plt.bar_label(bars, fmt="$%.0f")
    plt.savefig(output_dir / "kpi_acquisition_efficiency.png")
    plt.close()

    # 3. Revenue per Segment
    if "clv_segment" in df.columns:
        plt.figure(figsize=(10, 6))
        segment_rev = df.groupby("clv_segment")[clv_col].sum().sort_index()
        segment_rev.plot(kind="bar", color="#4fd1c5")
        plt.title("Total Revenue Potential by Segment")
        plt.ylabel("Total Predicted CLTV ($)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "kpi_revenue_by_segment.png")
        plt.close()

    # 4. Churn Risk Distribution
    if "p_churn_30day" in df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(df["p_churn_30day"], bins=30, color="#f6ad55", edgecolor="white")
        plt.title("Customer Churn Risk Distribution (Next 30 Days)")
        plt.xlabel("Probability of Churn/Inactivity")
        plt.ylabel("Number of Customers")
        plt.axvline(
            df["p_churn_30day"].mean(),
            color="red",
            linestyle="dashed",
            linewidth=1,
            label=f"Mean: {df['p_churn_30day'].mean():.2f}",
        )
        plt.legend()
        plt.savefig(output_dir / "kpi_churn_risk_dist.png")
        plt.close()

    logger.info("Saved Strategic KPI plots including Segment Revenue and Churn Risk.")


def _prepare_cltv_summary(
    df: pd.DataFrame,
    chart_type: str,
) -> str:
    """Prepare targeted CLTV data summary optimized for LLM interpretation context.

    Generates a concise statistical summary tailored to specific analysis objectives
    (Overview vs Churn). This summary is fed to Groq LLM to generate business insights,
    so it focuses on relevant metrics and omits unnecessary detail.

    Context-Aware Summarization:
    - **Overview Focus**: Emphasizes CLTV financial projections, revenue concentration
    - **Churn Focus**: Emphasizes retention indicators, at-risk customer counts

    Args:
        df: CLTV predictions DataFrame (Pandas) with columns:
            - frequency / recency / monetary_value: Core RFM features
            - clv_12mo or predicted_cltv: CLTV forecast
            - p_alive (optional): Retention probability
            - clv_segment (optional): Customer segment
        chart_type: Analysis focus identifier (e.g., 'CLTV Global Overview',
            'Churn Risk & Retention Strategy'). Determines which metrics to include.

    Returns:
        Multiline string containing:
            - Analysis type and customer base size
            - Core behavioral metrics (avg frequency, monetary value, recency)
            - Context-specific KPIs:
                - For Overview: Mean/Max/Total CLTV, revenue projections
                - For Churn: P(Alive) statistics, at-risk customer counts
            - Segment distribution (if available)

    Note:
        - All metrics formatted as dictionaries for easy parsing by LLM
        - Handles missing columns gracefully (returns 'N/A')
        - "Critical Risk" defined as P(Alive) < 20%
        - "At-Risk" defined as P(Alive) < 50%

    Example:
        >>> summary = _prepare_cltv_summary(df, 'CLTV Global Overview')
        >>> # Returns:
        >>> # Analysis Type: CLTV Global Overview
        >>> # Total Customer Base: 10000
        >>> # Core Behavioral Metrics: {'Avg Frequency': 3.5, ...}
        >>> # CLTV Financial Projections: {'Mean CLTV': 425, ...}
    """
    summary_parts = []
    summary_parts.append(f"Analysis Type: {chart_type}")
    summary_parts.append(f"Total Customer Base: {len(df)}")

    # Core Metrics
    metrics = {
        "Avg Frequency": df["frequency"].mean() if "frequency" in df.columns else "N/A",
        "Avg Monetary Value": df["monetary_value"].mean()
        if "monetary_value" in df.columns
        else "N/A",
        "Avg Recency": df["recency"].mean() if "recency" in df.columns else "N/A",
    }
    summary_parts.append(f"Core Behavioral Metrics: {metrics}")

    # Targeted Metrics: CLTV focus
    if "Overview" in chart_type:
        clv_col = "clv_12mo" if "clv_12mo" in df.columns else "predicted_cltv"
        if clv_col in df.columns:
            clv_stats = {
                "Mean CLTV": df[clv_col].mean(),
                "Total Projected Value": df[clv_col].sum(),
                "Max CLTV": df[clv_col].max(),
            }
            summary_parts.append(f"CLTV Financial Projections ({clv_col}): {clv_stats}")

    # Targeted Metrics: Churn focus
    if "Churn" in chart_type:
        if "p_alive" in df.columns:
            alive_stats = {
                "Avg P(Alive)": df["p_alive"].mean(),
                "Critical Risk Customers (<20% P(Alive))": len(df[df["p_alive"] < 0.2]),
                "At-Risk Customers (<50% P(Alive))": len(df[df["p_alive"] < 0.5]),
            }
            summary_parts.append(f"Retention & Churn Indicators: {alive_stats}")

    # Segments (Shared)
    if "clv_segment" in df.columns:
        segment_counts = df["clv_segment"].value_counts().to_dict()
        summary_parts.append(f"Customer Segments Distribution: {segment_counts}")

    return "\n".join(str(p) for p in summary_parts)


def interpret_cltv_visualizations(
    cltv_predictions: pl.DataFrame, params: Dict[str, Any]
) -> Dict[str, str]:
    """Generate AI-powered strategic insights for CLTV visualizations using Groq LLM.

    Leverages large language models (via Groq API) to automatically interpret CLTV
    predictions and generate executive summaries. Produces two distinct reports:
    business growth strategy and churn risk mitigation. This enables non-technical
    stakeholders to understand model outputs and take action.

    Generated Interpretations:
    1. **CLTV Overview Report**: Business health, revenue concentration (whale analysis),
       acquisition efficiency (LTV:CAC), growth optimization recommendations
    2. **Churn Risk Report**: Retention diagnostics, at-risk segment identification,
       churn mitigation strategies, silent churn detection

    LLM Configuration:
    - Provider: Groq (fast inference for llama3/mixtral models)
    - Default model: llama3-70b-8192 (configurable via params)
    - Temperature: 0.6 (balanced creativity/accuracy)
    - Output format: Markdown with business-oriented sections

    Args:
        cltv_predictions: CLTV prediction DataFrame with columns:
            - Standard CLTV features: clv_12mo, p_alive, p_churn_30day
            - RFM features: frequency, recency, monetary_value
            - Segments: clv_segment
        params: Configuration dictionary containing:
            - llm.model (str): Groq model ID (default: 'llama3-70b-8192')
            - llm.temperature (float): Sampling temperature (default: 0.6)
            - Environment variable GROQ_API_KEY must be set

    Returns:
        Dictionary containing:
            - 'cltv_overview' (str): Markdown-formatted business growth analysis
            - 'churn_risk_analysis' (str): Markdown-formatted retention strategy
            - 'status' (str): 'success' or 'skipped'
            - 'reason' (str, if skipped): Explanation for why generation was skipped

    Raises:
        Exception (caught): LLM API errors logged but don't halt pipeline.
            Returns error message string in interpretation field.

    Note:
        - Requires GROQ_API_KEY environment variable (obtain from console.groq.com)
        - Groq client and library must be installed (pip install groq)
        - Falls back gracefully if API unavailable (returns skipped status)
        - System prompt explicitly prohibits code generation (enforces business language)
        - Each interpretation makes 1 API call (~300-500 tokens output)

    Warning:
        - API calls cost money (check Groq pricing)
        - Rate limits apply (default: 30 req/min for free tier)
        - Generated interpretations may hallucinate; validate critical claims

    Example:
        >>> os.environ['GROQ_API_KEY'] = 'gsk_...'
        >>> params = {'llm': {'model': 'llama3-70b-8192', 'temperature': 0.6}}
        >>> interpretations = interpret_cltv_visualizations(cltv_pred, params)
        >>> print(interpretations['cltv_overview'])
        # Markdown report with strategic growth insights...
        >>> print(interpretations['status'])
        'success'
    """
    logger.info("Generating AI interpretations for CLTV visualizations...")

    api_key = os.environ.get("GROQ_API_KEY")
    if not GROQ_AVAILABLE or not api_key:
        logger.warning(
            "Groq not available or GROQ_API_KEY not set. Skipping interpretations."
        )
        return {
            "status": "skipped",
            "reason": "Groq API not available or GROQ_API_KEY not set",
        }

    # Initialize Groq client
    client = Groq(api_key=api_key)

    # Convert polars to pandas for easier summary generation
    df = cltv_predictions.to_pandas()

    # Get LLM configuration from params
    llm_config = params.get("llm", {})
    model = llm_config.get("model", "llama3-70b-8192")
    temperature = llm_config.get("temperature", 0.6)

    interpretations = {}

    system_prompt_base = """You are a Principal MLOps Architect and Senior Data Scientist specializing in Media & Entertainment CLTV.
CRITICAL: DO NOT PROVIDE ANY PYTHON CODE OR CODE BLOCKS. 
All analysis must be in natural language, business-oriented, and provide actionable recommendations.
Format your response in markdown with clear, descriptive sections."""

    configs = [
        {
            "name": "cltv_overview",
            "chart_type": "CLTV Global Overview & Strategic Growth",
            "focus": "Overall business health, revenue concentration (Whales), and acquisition efficiency (LTV:CAC).",
            "custom_instruction": "Structure the report into 'Strategic Growth Insights' and 'Value Optimization Recommendations'.",
        },
        {
            "name": "churn_risk_analysis",
            "chart_type": "Churn Risk & Retention Strategy",
            "focus": "Identifying at-risk segments, impact of churn reduction, and silent churn detection.",
            "custom_instruction": "Structure the report into 'Retention Diagnostics' and 'Churn Mitigation Strategies'.",
        },
    ]

    for config in configs:
        try:
            summary = _prepare_cltv_summary(df, config["chart_type"])
            user_prompt = f"Data Summary:\n{summary}\n\nStrategic Focus: {config['focus']}\n\n{config['custom_instruction']}"

            logger.info(f"Calling LLM for interpretation: {config['name']}...")
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt_base},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )

            interpretations[config["name"]] = completion.choices[0].message.content
            logger.info(f"Successfully generated interpretation for {config['name']}")
        except Exception as e:
            logger.error(
                f"Error generating interpretation for {config['name']}: {str(e)}"
            )
            interpretations[config["name"]] = (
                f"Error generating interpretation: {str(e)}"
            )

    interpretations["status"] = "success"
    return interpretations


def save_cltv_interpretations(
    interpretations: Dict[str, str], params: Dict[str, Any]
) -> str:
    """Save LLM-generated CLTV interpretations to markdown files for dashboard integration.

    Persists AI-generated business insights as markdown files that can be displayed
    in Streamlit dashboards, included in reports, or shared with stakeholders. Each
    interpretation type gets its own file for modular dashboard composition.

    Args:
        interpretations: Dictionary from interpret_cltv_visualizations() containing:
            - 'cltv_overview' (str): Business growth analysis markdown
            - 'churn_risk_analysis' (str): Retention strategy markdown
            - 'status' (str): Generation status ('success' or 'skipped')
            - 'reason' (str, optional): Skip reason if status != 'success'
        params: Configuration dictionary containing:
            - plots_dir (str): Base output directory (default: 'data/08_reporting/plots')
            - Interpretations saved to plots_dir/interpretations/

    Returns:
        Status message string:
            - Success: "Successfully saved N interpretation files to {path}"
            - Skipped: "Interpretations skipped: {reason}"

    Side Effects:
        Creates markdown files in plots_dir/interpretations/:
            - cltv_overview.md
            - churn_risk_analysis.md

    Note:
        - Automatically creates interpretations directory if it doesn't exist
        - Skips 'status' and 'reason' keys from interpretations dict
        - UTF-8 encoding for special characters (currency symbols, etc.)
        - Logs individual file save confirmations

    Example:
        >>> interpretations = interpret_cltv_visualizations(cltv_pred, params)
        >>> result = save_cltv_interpretations(interpretations, params)
        INFO:Saved cltv_overview.md
        INFO:Saved churn_risk_analysis.md
        >>> print(result)
        'Successfully saved 2 interpretation files to data/08_reporting/plots/interpretations'
    """
    logger.info("Saving CLTV visualization interpretations...")

    if interpretations.get("status") != "success":
        logger.warning(
            f"Interpretations not available: {interpretations.get('reason', 'unknown')}"
        )
        return f"Interpretations skipped: {interpretations.get('reason', 'unknown')}"

    output_dir = (
        Path(params.get("plots_dir", "data/08_reporting/plots")) / "interpretations"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    for viz_name, interpretation in interpretations.items():
        if viz_name == "status":
            continue

        filename = f"{viz_name}.md"
        filepath = output_dir / filename

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(interpretation)
            saved_files.append(str(filepath))
            logger.info(f"Saved {filename}")
        except Exception as e:
            logger.error(f"Error saving {filename}: {str(e)}")

    result_msg = (
        f"Successfully saved {len(saved_files)} interpretation files to {output_dir}"
    )
    logger.info(result_msg)

    return result_msg


def _load_slm_pipeline(model_id: str):
    """Load Small Language Model (SLM) with hardware acceleration for local inference.

    Initializes a HuggingFace transformers text-generation pipeline optimized for
    available hardware (CUDA/XPU/MPS/CPU).

    Hardware Auto-Detection Priority:
    1. CUDA (NVIDIA GPUs) - Best performance
    2. XPU (Intel Arc GPUs) - Good performance with native PyTorch XPU
    3. MPS (Apple Silicon) - Good performance on M1/M2/M3
    4. CPU - Fallback (logs warning about degraded performance)

    Args:
        model_id: HuggingFace model identifier.

    Returns:
        transformers.Pipeline: Text-generation pipeline.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    clear_device_cache()

    logger.info(f"Loading SLM model: {model_id}...")
    try:
        # Determine device
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = "xpu"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        if device == "cpu":
            logger.warning(
                "⚠️ ALERT: Using CPU for SLM inference. Performance will be significantly degraded."
            )
        else:
            logger.info(f"Using hardware accelerator: {device} for SLM inference")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        # In Native XPU (PyTorch 2.6+), we can use device_map="auto" or "xpu"
        # but for some models, loading to CPU first and then moving is safer for memory.
        # However, "Aborted!" crashes often happen when transferring to XPU.

        load_kwargs = {
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "eager",
        }

        if device == "xpu":
            # For XPU, we load with specific device_map if accelerate allows,
            # otherwise fall back to manual move but with empty cache.
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, device_map={"": "xpu:0"}, **load_kwargs
                )
            except Exception as e:
                logger.info(
                    f"Direct XPU load failed ({e}), falling back to CPU->XPU move..."
                )
                model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
                torch.xpu.empty_cache()
                model = model.to("xpu")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto" if device != "cpu" else None, **load_kwargs
            )

        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return pipe
    except Exception as e:
        logger.error(f"Failed to load SLM model {model_id}: {str(e)}")
        return None


def interpret_model_parameters_slm(
    bg_nbd_model: BetaGeoFitter,
    gamma_gamma_model: GammaGammaFitter,
    params: Dict[str, Any],
) -> None:
    """Generate technical interpretation of BG/NBD and Gamma-Gamma model parameters using local SLM.

    Uses a Small Language Model (SLM) running locally to explain the meaning of fitted
    probabilistic model parameters in accessible technical language. Helps data scientists
    understand what the learned parameters reveal about customer behavior without needing
    to manually interpret Beta/Gamma distribution shapes.

    Model Parameters Explained:
    - **BG/NBD**: r, alpha (transaction rate heterogeneity), a, b (churn probability shape)
    - **Gamma-Gamma**: p, q, v (monetary value distribution parameters)

    SLM generates insights like:
    - "High r/alpha ratio indicates most customers are low-frequency purchasers"
    - "Low a/b ratio suggests churn happens gradually rather than suddenly"
    - "High p/q suggests consistent spending across customers"

    Args:
        bg_nbd_model: Fitted BetaGeoFitter with params_ attribute containing
            r, alpha, a, b parameters
        gamma_gamma_model: Fitted GammaGammaFitter with params_ attribute containing
            p, q, v parameters
        params: Configuration dictionary containing:
            - plots_dir (str): Output directory (default: 'data/08_reporting/plots')
            - slm.model (str): HuggingFace model ID (default: 'microsoft/Phi-4-mini-instruct')
            - Interpretation saved to plots_dir/interpretations/model_parameters.md

    Returns:
        None. Saves markdown file with parameter interpretation.

    Note:
        - Loads SLM locally (no API key required) for privacy and cost savings
        - Generates ~400 tokens of technical explanation
        - Temperature: 0.7 for balanced creativity/accuracy
        - Falls back gracefully if SLM loading fails

    Example:
        >>> params = {'slm': {'model': 'microsoft/Phi-4-mini-instruct'}}
        >>> interpret_model_parameters_slm(bgf, ggf, params)
        INFO:Loading SLM model: microsoft/Phi-4-mini-instruct...
        INFO:Saved model_parameters.md
        >>> # File contains markdown like:
        >>> # "The BG/NBD parameters indicate a customer base with..."
    """
    slm_params = params.get("slm", {})
    model_id = slm_params.get("model", "microsoft/Phi-4-mini-instruct")

    pipe = _load_slm_pipeline(model_id)
    if not pipe:
        return

    output_dir = (
        Path(params.get("plots_dir", "data/08_reporting/plots")) / "interpretations"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract parameters
    bg_params = bg_nbd_model.params_
    gg_params = gamma_gamma_model.params_

    bg_summary = (
        f"BG/NBD (Transaction Process):\n"
        f"- r (Transaction rate shape): {bg_params.get('r', 0.0):.4f}\n"
        f"- alpha (Transaction rate scale): {bg_params.get('alpha', 0.0):.4f}\n"
        f"- a (Dropout probability shape alpha): {bg_params.get('a', 0.0):.4f}\n"
        f"- b (Dropout probability shape beta): {bg_params.get('b', 0.0):.4f}\n"
    )

    gg_summary = (
        f"Gamma-Gamma (Monetary Value):\n"
        f"- p (Shape of transaction value): {gg_params.get('p', 0.0):.4f}\n"
        f"- q (Shape of latent mean transaction value): {gg_params.get('q', 0.0):.4f}\n"
        f"- v (Scale of latent mean transaction value): {gg_params.get('v', 0.0):.4f}\n"
    )

    context = (
        f"Model Parameters Summary:\n{bg_summary}\n{gg_summary}\n\n"
        "Explain what these parameters imply about the customer base's behavior "
        "(e.g., are they frequent buyers? is churn high? is monetary value variance high?). "
        "Provide a technical yet accessible summary for data scientists."
    )

    messages = [
        {
            "role": "system",
            "content": "You are a Principal MLOps Architect explaining probabilistic model parameters.",
        },
        {"role": "user", "content": context},
    ]

    try:
        output = pipe(
            messages,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.7,
            use_cache=False,
        )
        generated = output[0]["generated_text"]
        content = (
            generated[-1]["content"] if isinstance(generated, list) else str(generated)
        )

        with open(output_dir / "model_parameters.md", "w") as f:
            f.write(f"### AI Model Parameter Interpretation (SLM)\n\n{content}")
        logger.info("Saved model_parameters.md")
    except Exception as e:
        logger.error(f"Failed to interpret model parameters: {str(e)}")


def interpret_customer_prediction_cards(
    cltv_predictions: pl.DataFrame, params: Dict[str, Any]
) -> None:
    """Generate personalized AI interpretations for individual customer CLTV predictions.

    Uses a local Small Language Model to automatically write 2-sentence strategic summaries
    for each customer's prediction card. Enables customer success teams to quickly understand
    customer value profiles and prioritize interventions without manual analysis.

    Generated interpretations synthesize:
    - CLTV forecast magnitude (high/medium/low value)
    - Churn risk level (immediate vs safe vs at-risk)
    - Purchase behavior patterns (frequency, transaction value)
    - Strategic recommendations (retain, nurture, win-back)

    Customer Prioritization:
    - Sorts customers by CLTV (descending) before interpretation
    - Interprets top N customers (default: 50) to focus on high-value accounts
    - Configurable via params['slm']['max_customers']

    Args:
        cltv_predictions: CLTV prediction DataFrame with columns:
            - customer_id (str): Customer identifier (used for filename)
            - clv_12mo or predicted_cltv (float): CLTV forecast
            - p_alive (float): Retention probability
            - p_churn_30day (float): 30-day churn risk
            - expected_transactions_12mo (float): Expected purchase count
            - expected_value_per_transaction (float): Expected avg transaction value
            - clv_segment (str): Customer segment (Q1-Q5)
            - cohort_month (str): Acquisition cohort
        params: Configuration dictionary containing:
            - plots_dir (str): Base output directory (default: 'data/08_reporting/plots')
            - slm.model (str): HuggingFace model ID (default: 'microsoft/Phi-4-mini-instruct')
            - slm.max_customers (int): Max customers to interpret (default: 50)
            - Interpretations saved to plots_dir/interpretations/customers/{customer_id}.md

    Returns:
        None. Saves one markdown file per customer:
            - {customer_id}.md containing 2-sentence strategic summary

    Note:
        - Progress bar (tqdm) shows real-time generation status
        - Each interpretation: ~150 tokens at temperature 0.7
        - Failures logged as warnings but don't halt batch processing
        - SLM loaded once and reused for all customers (efficient)

    Warning:
        - Inference time scales linearly with max_customers
        - 50 customers × 5 seconds/customer ≈ 4 minutes on CPU
        - Use GPU/XPU acceleration for faster batch processing
        - Generated interpretations may occasionally hallucinate; spot-check quality

    Example:
        >>> params = {
        ...     'slm': {
        ...         'model': 'microsoft/Phi-4-mini-instruct',
        ...         'max_customers': 10
        ...     }
        ... }
        >>> interpret_customer_prediction_cards(cltv_pred, params)
        INFO:Loading SLM model: microsoft/Phi-4-mini-instruct...
        INFO:Generating interpretations for 10 customers...
        Generating AI Customer Insights: 100%|████████████| 10/10
        INFO:Finished customer interpretations. Files saved in .../customers

        >>> # Example output file: C12345.md
        >>> # "This customer represents exceptional long-term value with a predicted
        >>> # 12-month CLTV of $1,250 and a strong 87% probability of remaining active.
        >>> # Their consistent purchase behavior and high transaction values indicate
        >>> # they are a prime candidate for VIP retention programs and upsell campaigns."
    """
    output_dir = (
        Path(params.get("plots_dir", "data/08_reporting/plots"))
        / "interpretations"
        / "customers"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = cltv_predictions.to_pandas()

    slm_params = params.get("slm", {})
    model_id = slm_params.get("model", "microsoft/Phi-4-mini-instruct")
    max_customers = slm_params.get("max_customers", 50)

    # Sort by CLTV to interpret high-value customers first if we are limiting
    clv_col = "clv_12mo" if "clv_12mo" in df.columns else "predicted_cltv"
    if clv_col in df.columns:
        df = df.sort_values(by=clv_col, ascending=False)

    df_to_interpret = df.head(max_customers)

    pipe = _load_slm_pipeline(model_id)
    if not pipe:
        return

    logger.info(f"Generating interpretations for {len(df_to_interpret)} customers...")

    from tqdm import tqdm

    for _, row in tqdm(
        df_to_interpret.iterrows(),
        total=len(df_to_interpret),
        desc="Generating AI Customer Insights",
    ):
        customer_id = row["customer_id"]

        # Prepare metrics context
        metrics_str = (
            f"- 12-Month Predicted CLTV: ${row.get(clv_col, 0):,.2f}\n"
            f"- Probability Alive: {row.get('p_alive', 0):.1%}\n"
            f"- 30-Day Churn Risk: {row.get('p_churn_30day', 0):.1%}\n"
            f"- Expected Transactions (12m): {row.get('expected_transactions_12mo', 0):.2f}\n"
            f"- Avg. Transaction Value: ${row.get('expected_value_per_transaction', 0):,.2f}\n"
            f"- Segment: {row.get('clv_segment', 'N/A')}\n"
            f"- Cohort: {row.get('cohort_month', 'N/A')}"
        )

        # {model_id} instruction format
        messages = [
            {
                "role": "system",
                "content": "You are a Principal AI and Data Engineer specializing in MLOps and CLTV analysis. Provide concise, strategic business interpretations for customer prediction cards.",
            },
            {
                "role": "user",
                "content": f"Interpret these prediction metrics for Customer {customer_id} and provide a 2-sentence strategic summary:\n{metrics_str}",
            },
        ]

        try:
            # Generate interpretation
            output = pipe(
                messages,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                use_cache=False,
            )

            # Extract interpretation from pipeline output
            generated = output[0]["generated_text"]
            if isinstance(generated, list):
                # If pipeline returns the full conversation list, take the last message content
                interpretation = generated[-1]["content"]
            else:
                interpretation = str(generated)

            # Save to file
            with open(output_dir / f"{customer_id}.md", "w") as f:
                f.write(
                    f"### AI Prediction Interpretation (SLM: {model_id.split('/')[-1]})\n\n"
                )
                f.write(interpretation.strip())
        except Exception as e:
            logger.warning(
                f"Failed to generate interpretation for customer {customer_id}: {str(e)}"
            )

    logger.info(f"Finished customer interpretations. Files saved in {output_dir}")
