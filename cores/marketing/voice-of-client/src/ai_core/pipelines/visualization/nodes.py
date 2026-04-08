"""Visualization nodes for generating charts and plots."""

import logging
import json
import os
import gc
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
from utils.device import get_device
from groq import Groq
from dotenv import load_dotenv
from tqdm import tqdm

# Load environmental variables from .env if present
load_dotenv()

# Try to import HuggingFace transformers for SLM-based labeling
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch

    SLM_AVAILABLE = True
except ImportError:
    SLM_AVAILABLE = False

logger = logging.getLogger(__name__)

# Global variable to cache the loaded SLM model and tokenizer
_cached_slm_model = None
_cached_slm_tokenizer = None
_cached_slm_pipeline = None


def clear_device_cache():
    """Release unoccupied cached memory for available devices (CUDA, XPU, MPS)."""
    if not SLM_AVAILABLE:
        return

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


def load_slm_model(model_name: str):
    # ... (existing code remains same)
    global _cached_slm_model, _cached_slm_tokenizer, _cached_slm_pipeline

    if _cached_slm_pipeline is not None:
        return _cached_slm_model, _cached_slm_tokenizer, _cached_slm_pipeline

    if not SLM_AVAILABLE:
        return None, None, None

    # Clear cache before attempting to load a new model
    clear_device_cache()

    try:
        logger.info(f"Loading SLM for visualization interpretation: {model_name}")
        device = get_device("SLM inference")
        torch_dtype = torch.float16 if device != "cpu" else torch.float32
        device_map = device if device in ["cuda", "xpu", "mps"] else "cpu"

        _cached_slm_tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=False, use_fast=True
        )
        _cached_slm_tokenizer.pad_token = _cached_slm_tokenizer.eos_token
        _cached_slm_tokenizer.pad_token_id = _cached_slm_tokenizer.eos_token_id

        _cached_slm_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
        )
        _cached_slm_model.generation_config.pad_token_id = (
            _cached_slm_tokenizer.eos_token_id
        )

        _cached_slm_pipeline = pipeline(
            "text-generation",
            model=_cached_slm_model,
            tokenizer=_cached_slm_tokenizer,
        )
        return _cached_slm_model, _cached_slm_tokenizer, _cached_slm_pipeline
    except Exception as e:
        logger.error(f"Failed to load SLM model {model_name}: {e}")
        return None, None, None


def generate_groq_interpretation(
    summary: str, context: str, params: Dict[str, Any]
) -> str:
    """Generate a high-quality business insight using Groq Cloud API."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found in environment. Skipping Groq insights.")
        return ""

    try:
        client = Groq(api_key=api_key)
        prompt = f"""
        You are a B2B Strategic Account Intelligence Expert. 
        Analyze the following data summary and provide a professional, concise, and actionable business insight (max 3 sentences).
        Focus on 'The Why' and 'The What Now'.
        
        Context: {context}
        Data Summary: {summary}
        
        Insight:
        """

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.65,
            max_completion_tokens=512,
            stream=False,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error calling Groq for interpretation: {e}")
        return ""


def generate_interpretation(summary: str, context: str, model_name: str) -> str:
    """Generate a natural language interpretation using SLM."""
    if not SLM_AVAILABLE:
        return ""

    # Clear cache before inference
    clear_device_cache()

    model, tokenizer, pipe = load_slm_model(model_name)
    if not pipe:
        return ""

    messages = [
        {
            "role": "system",
            "content": "You are a Strategic Account Intelligence expert. Analyze the following data and provide a concise (2-3 sentences) business insight for an Account Manager or VP of Success.",
        },
        {
            "role": "user",
            "content": f"Context: {context}\nData Summary: {summary}\n\nProvide a professional business interpretation:",
        },
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=100,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        interpretation = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        ).strip()
        return interpretation
    except Exception as e:
        logger.error(f"Error calling SLM for interpretation: {e}")
        return ""


def create_feature_distribution_plots(
    features_data: pd.DataFrame, params: Dict[str, Any], slm_model: str
) -> Dict[str, Any]:
    """Create distribution plots and AI interpretations for numeric features."""
    logger.info("Creating feature distribution plots...")

    # Select numeric columns
    numeric_cols = features_data.select_dtypes(include=[np.number]).columns[:10]

    if len(numeric_cols) == 0:
        logger.warning("No numeric columns found for distribution plots")
        return {"status": "skipped", "reason": "no_numeric_columns"}

    # Create figure with subplots
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1) if len(numeric_cols) > 1 else np.array([[axes]])

    axes = axes.flatten()
    interpretations = {}
    groq_insights = {}

    for idx, col in enumerate(numeric_cols):
        sns.histplot(features_data[col], kde=True, ax=axes[idx])
        axes[idx].set_title(f"Distribution of {col}")
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel("Frequency")

        # Generate interpretation for each feature
        stats_summary = features_data[col].describe().to_string()

        # 1. Local SLM Insight
        interp = generate_interpretation(
            stats_summary,
            f"Statistical distribution analysis of the feature '{col}'.",
            slm_model,
        )
        if interp:
            interpretations[col] = interp

        # 2. Groq Strategic Insight
        groq_interp = generate_groq_interpretation(
            stats_summary,
            f"Strategic statistical analysis of the customer behavior feature '{col}'.",
            params,
        )
        if groq_interp:
            groq_insights[col] = groq_interp

    # Hide extra subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    return {
        "plot": fig,
        "type": "feature_distributions",
        "n_features": len(numeric_cols),
        "interpretations": interpretations,
        "groq_insights": groq_insights,
        "status": "success",
    }


def create_correlation_heatmap(
    features_data: pd.DataFrame, params: Dict[str, Any], slm_model: str
) -> Dict[str, Any]:
    """Create correlation heatmap and AI interpretation."""
    logger.info("Creating correlation heatmap...")

    # Select numeric columns
    numeric_data = features_data.select_dtypes(include=[np.number])

    if numeric_data.shape[1] < 2:
        logger.warning("Need at least 2 numeric columns for correlation heatmap")
        return {"status": "skipped", "reason": "insufficient_numeric_columns"}

    # Calculate correlation matrix
    corr_matrix = numeric_data.corr()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=16, pad=20)

    plt.tight_layout()

    # AI Interpretation
    high_corrs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(corr_matrix.iloc[i, j]) > 0.5:
                high_corrs.append(
                    f"{cols[i]} and {cols[j]}: {corr_matrix.iloc[i, j]:.2f}"
                )

    summary = "High correlations detected:\n" + (
        "\n".join(high_corrs[:10]) if high_corrs else "None"
    )

    # 1. Local SLM
    interpretation = generate_interpretation(
        summary,
        "Inter-feature correlation analysis for all numeric variables in the dataset.",
        slm_model,
    )

    # 2. Groq
    groq_interpretation = generate_groq_interpretation(
        summary,
        "Strategic correlation matrix analysis exploring dependencies between customer metrics.",
        params,
    )

    return {
        "plot": fig,
        "type": "correlation_heatmap",
        "n_features": numeric_data.shape[1],
        "interpretation": interpretation,
        "groq_interpretation": groq_interpretation,
        "status": "success",
    }


def create_model_evaluation_plots(
    model: Any,
    X_test: pd.DataFrame,
    y_test: Any,
    params: Dict[str, Any],
    slm_model: str,
) -> Dict[str, Any]:
    """Create model evaluation visualizations and AI interpretations."""
    if model is None or X_test.empty:
        logger.warning(
            "Skipping model evaluation plots: No model or test data provided."
        )
        return {"status": "skipped", "reason": "model_disabled_or_insufficient_data"}

    logger.info("Creating model evaluation plots...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Check if it's a classification problem
    is_classification = hasattr(model, "predict_proba")
    interpretation = None
    groq_interpretation = None

    if is_classification and len(np.unique(y_test)) == 2:
        # Binary classification: ROC curve and confusion matrix
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        axes[0].plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
        )
        axes[0].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title("ROC Curve")
        axes[0].legend(loc="lower right")
        axes[0].grid(True, alpha=0.3)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1])
        axes[1].set_title("Confusion Matrix")
        axes[1].set_xlabel("Predicted Label")
        axes[1].set_ylabel("True Label")

        summary = f"Binary Classification Model Performance:\nAUC-ROC: {roc_auc:.4f}\nConfusion Matrix:\n{cm}"
        interpretation = generate_interpretation(
            summary, "Evaluation of the churn prediction model performance.", slm_model
        )
        groq_interpretation = generate_groq_interpretation(
            summary,
            "Strategic health assessment of the churn prediction model.",
            params,
        )

    else:
        # Regression or multiclass: Actual vs Predicted
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            "r--",
            lw=2,
            label="Perfect Prediction",
        )
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Actual vs Predicted Values")
        ax.legend()
        ax.grid(True, alpha=0.3)

        summary = "Regression/General Model Performance: Scatter plot of Actual vs Predicted values shows model alignment."
        interpretation = generate_interpretation(
            summary, "Model evaluation via Actual vs Predicted comparison.", slm_model
        )
        groq_interpretation = generate_groq_interpretation(
            summary, "Strategic accuracy assessment of the predictive model.", params
        )

    plt.tight_layout()

    return {
        "plot": fig,
        "type": "model_evaluation",
        "is_classification": is_classification,
        "interpretation": interpretation,
        "groq_interpretation": groq_interpretation,
        "status": "success",
    }


def create_analysis_visualizations(
    correlation_data: Dict[str, Any], params: Dict[str, Any], slm_model: str
) -> Dict[str, Any]:
    """Create visualizations and AI interpretations for NLP-Business correlations."""
    if correlation_data.get("status") != "success":
        logger.warning("Correlation analysis failed or skipped. Skipping plots.")
        return {"status": "skipped"}

    logger.info("Creating NLP-Business correlation plots...")

    plots = {}
    interpretations = {}
    groq_insights = {}
    correlations = correlation_data.get("correlations", {})

    for target, target_corrs in correlations.items():
        if not target_corrs:
            continue

        df_corrs = pd.Series(target_corrs).sort_values(ascending=False)

        # Plot top 10 and bottom 10
        top_n = 10
        df_plot = pd.concat(
            [df_corrs.head(top_n), df_corrs.tail(top_n)]
        ).drop_duplicates()

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ["green" if x > 0 else "red" for x in df_plot.values]
        df_plot.plot(kind="barh", ax=ax, color=colors)
        ax.set_title(f"Key Drivers for {target}")
        ax.set_xlabel("Correlation Coefficient")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        plots[target] = fig

        # AI Interpretation for each target metric
        summary = f"Top Positive Drivers for {target}:\n{df_corrs.head(5).to_string()}\n\nTop Negative Drivers:\n{df_corrs.tail(5).to_string()}"

        # 1. Local SLM
        interp = generate_interpretation(
            summary,
            f"Business driver analysis correlating NLP features with {target}.",
            slm_model,
        )
        if interp:
            interpretations[target] = interp

        # 2. Groq Strategic
        groq_interp = generate_groq_interpretation(
            summary,
            f"Strategic analysis of high-impact drivers influencing customer satisfaction and NPS ({target}).",
            params,
        )
        if groq_interp:
            groq_insights[target] = groq_interp

    return {
        "plots": plots,
        "type": "analysis_correlations",
        "interpretations": interpretations,
        "groq_insights": groq_insights,
        "status": "success",
    }


def generate_feedback_level_insights(
    data: pd.DataFrame, params: Dict[str, Any]
) -> Dict[str, str]:
    """Generate AI insights for each individual feedback entry using Groq."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found. Skipping individual feedback insights.")
        return {}

    logger.info(f"Generating Groq insights for {len(data)} feedback entries...")
    client = Groq(api_key=api_key)
    insights = {}

    # Identify outcome columns
    sent_cols = [c for c in data.columns if c.startswith("sent_")]
    emo_cols = [c for c in data.columns if c.startswith("emo_")]
    pred_cols = [
        "Predicted_NPS_Score",
        "Predicted_CSAT_Score",
        "Predicted_Churn_Prob",
        "Churn_Risk_Level",
    ]

    # Process only a subset or all if small (using 50 as safety limit or full set if requested)
    # The user asked for "each collapsible row", so we'll do all but log if it's large.
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Generating AI Insights"):
        interaction_id = row["Interaction_ID"]

        # Build outcomes summary
        outcomes = {
            "Feedback": row["Interaction_Payload"],
            "Topic": row.get("Topic_Name"),
            "Aspect_Sentiments": {
                c.replace("sent_", ""): row[c] for c in sent_cols if c in data.columns
            },
            "Emotions": {
                c.replace("emo_", ""): row[c] for c in emo_cols if c in data.columns
            },
            "Predictions": {c: row[c] for c in pred_cols if c in data.columns},
            "Context": {
                "Polarity": row.get("sentiment_polarity"),
                "Intensity": row.get("sentiment_intensity"),
            },
        }

        prompt = f"""
        Analyze this customer feedback and its AI-model outcomes. 
        Provide a concise, 2-sentence strategic summary for an Account Manager.
        Outcomes: {json.dumps(outcomes)}
        """

        try:
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_completion_tokens=256,
                stream=False,
            )
            insights[interaction_id] = completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate insight for {interaction_id}: {e}")
            insights[interaction_id] = "Strategic insight unavailable."

    return insights


def generate_temporal_insights(data: pd.DataFrame, params: Dict[str, Any]) -> str:
    """Generate AI insights for temporal trends using Groq.

    This function reduces the dataframe to essential temporal metrics
    and generates strategic insights about trends over time.

    Args:
        data: Enriched feedback dataframe with temporal data
        params: Visualization parameters

    Returns:
        Temporal insight string (empty string if generation fails)
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found. Skipping temporal insights.")
        return ""

    if data.empty or "Timestamp" not in data.columns:
        logger.warning("No temporal data available for insights.")
        return ""

    logger.info("Generating Groq temporal insights...")

    try:
        # Convert timestamp to datetime
        df = data.copy()
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"])

        if df.empty:
            return ""

        # Create monthly aggregation for reduced data size
        df["Period"] = df["Timestamp"].dt.to_period("M").dt.start_time

        # Calculate advocacy index function
        def advocacy_index_fn(x):
            promoters = (x >= 9).sum()
            detractors = (x <= 6).sum()
            total = len(x)
            return (promoters - detractors) / total * 100 if total > 0 else 0

        # Aggregate key metrics by period (REDUCED DATA)
        temporal_summary = (
            df.groupby("Period")
            .agg(
                {
                    "NPS_Score": ["mean", "std", "count"],
                    "CSAT_Score": ["mean", "std"],
                    "sentiment_polarity": "mean",
                    "sentiment_intensity": "mean",
                }
            )
            .reset_index()
        )

        # Flatten MultiIndex columns before merging
        temporal_summary.columns = [
            "_".join(col).strip("_") if isinstance(col, tuple) else col
            for col in temporal_summary.columns
        ]

        # Calculate advocacy index separately
        advocacy_by_period = (
            df.groupby("Period")["NPS_Score"].apply(advocacy_index_fn).reset_index()
        )
        advocacy_by_period.columns = ["Period", "Advocacy_Index"]

        # Merge advocacy index
        temporal_summary = temporal_summary.merge(advocacy_by_period, on="Period")

        # Convert Period to string for JSON serialization
        temporal_summary["Period"] = temporal_summary["Period"].astype(str)

        # Calculate period-over-period changes for latest period
        if len(temporal_summary) >= 2:
            latest = temporal_summary.iloc[-1]
            previous = temporal_summary.iloc[-2]

            changes = {
                "NPS_change": float(
                    latest["NPS_Score_mean"] - previous["NPS_Score_mean"]
                ),
                "CSAT_change": float(
                    latest["CSAT_Score_mean"] - previous["CSAT_Score_mean"]
                ),
                "Advocacy_change": float(
                    latest["Advocacy_Index"] - previous["Advocacy_Index"]
                ),
                "Volume_change": int(
                    latest["NPS_Score_count"] - previous["NPS_Score_count"]
                ),
                "Sentiment_change": float(
                    latest["sentiment_polarity_mean"]
                    - previous["sentiment_polarity_mean"]
                ),
            }
        else:
            changes = {}

        # Add topic trends if available
        topic_trends = {}
        if "Topic_Name" in df.columns:
            topic_df = df[~df["Topic_Name"].isin(["Outlier", "Uncategorized"])]
            if not topic_df.empty:
                # Get top 3 topics overall
                top_topics = topic_df["Topic_Name"].value_counts().head(3).to_dict()
                topic_trends = {str(k): int(v) for k, v in top_topics.items()}

        # Create reduced summary for Groq (minimize token usage)
        reduced_summary = {
            "periods": temporal_summary.to_dict(orient="records"),
            "latest_changes": changes,
            "top_topics": topic_trends,
            "total_periods": len(temporal_summary),
            "date_range": {
                "start": str(temporal_summary["Period"].iloc[0]),
                "end": str(temporal_summary["Period"].iloc[-1]),
            },
        }

        # Generate Groq insight
        client = Groq(api_key=api_key)
        prompt = f"""
        You are a B2B Customer Success Intelligence Expert analyzing temporal trends in customer feedback.
        
        Analyze the following temporal data and provide a strategic, actionable insight (max 4 sentences).
        Focus on:
        1. Key trends in customer satisfaction (NPS, CSAT, Advocacy Index)
        2. Notable changes or inflection points
        3. Strategic recommendations for the Customer Success team
        
        Temporal Data Summary:
        {json.dumps(reduced_summary, indent=2)}
        
        Provide a professional, concise strategic insight:
        """

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.65,
            max_completion_tokens=512,
            stream=False,
        )

        insight = completion.choices[0].message.content.strip()
        logger.info("Successfully generated temporal insights")
        return insight

    except Exception as e:
        logger.error(f"Error generating temporal insights: {e}")
        return ""


def save_visualizations(
    feature_dist: Dict[str, Any],
    correlation: Dict[str, Any],
    model_eval: Dict[str, Any],
    analysis_viz: Dict[str, Any],
    feedback_insights: Dict[str, str],
    temporal_insights: str,
    params: Dict[str, Any],
) -> str:
    """Save all visualization plots and AI interpretations to files."""
    logger.info("Saving visualization plots and AI interpretations...")

    output_dir = Path(params.get("output_dir", "data/08_reporting"))
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    all_interpretations = {
        "feature_distributions": feature_dist.get("interpretations", {}),
        "correlation_heatmap": correlation.get("interpretation"),
        "model_evaluation": model_eval.get("interpretation"),
        "analysis_drivers": analysis_viz.get("interpretations", {}),
        # Groq Insights
        "groq": {
            "feature_distributions": feature_dist.get("groq_insights", {}),
            "correlation_heatmap": correlation.get("groq_interpretation"),
            "model_evaluation": model_eval.get("groq_interpretation"),
            "analysis_drivers": analysis_viz.get("groq_insights", {}),
            "feedback_insights": feedback_insights,
            "temporal_trends": temporal_insights,
        },
    }

    # Map of (plot_data, filename_prefix)
    plot_map = [
        (feature_dist, "feature_distributions"),
        (correlation, "correlation_heatmap"),
        (model_eval, "model_evaluation"),
    ]

    for data, prefix in plot_map:
        if data.get("status") == "success":
            filepath = output_dir / f"{prefix}.png"
            data["plot"].savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close(data["plot"])
            saved_files.append(str(filepath))
            logger.info(f"Saved {prefix} to {filepath}")

    # Special handling for analysis_viz which has multiple plots
    if analysis_viz.get("status") == "success":
        for target, fig in analysis_viz["plots"].items():
            filepath = output_dir / f"driver_impact_{target.lower()}.png"
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close(fig)
            saved_files.append(str(filepath))
            logger.info(f"Saved driver impact plot for {target} to {filepath}")

    # Save AI Interpretations to JSON
    # We save this in the parent reporting directory, not the plots subdirectory
    report_root = output_dir.parent if output_dir.name == "plots" else output_dir
    interp_path = report_root / "ai_interpretations.json"
    with open(interp_path, "w") as f:
        json.dump(all_interpretations, f, indent=2)
    saved_files.append(str(interp_path))
    logger.info(f"Saved AI interpretations to {interp_path}")

    return f"Successfully saved {len(saved_files)} files (plots + interpretations) to {output_dir}"
