"""Visualization nodes for generating charts and plots."""
import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

logger = logging.getLogger(__name__)


def create_feature_distribution_plots(
    features_data: pd.DataFrame, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Create distribution plots for numeric features.
    
    Args:
        features_data: DataFrame containing feature data
        params: Visualization parameters
        
    Returns:
        Dictionary containing plot metadata
    """
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
    
    for idx, col in enumerate(numeric_cols):
        sns.histplot(features_data[col], kde=True, ax=axes[idx])
        axes[idx].set_title(f'Distribution of {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
    
    # Hide extra subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    return {
        "plot": fig,
        "type": "feature_distributions",
        "n_features": len(numeric_cols),
        "status": "success"
    }


def create_correlation_heatmap(
    features_data: pd.DataFrame, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Create correlation heatmap for numeric features.
    
    Args:
        features_data: DataFrame containing feature data
        params: Visualization parameters
        
    Returns:
        Dictionary containing plot metadata
    """
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
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title('Feature Correlation Heatmap', fontsize=16, pad=20)
    
    plt.tight_layout()
    
    return {
        "plot": fig,
        "type": "correlation_heatmap",
        "n_features": numeric_data.shape[1],
        "status": "success"
    }


def create_model_evaluation_plots(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """Create model evaluation visualizations.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        params: Visualization parameters
        
    Returns:
        Dictionary containing plot metadata
    """
    logger.info("Creating model evaluation plots...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Check if it's a classification problem
    is_classification = hasattr(model, 'predict_proba')
    
    if is_classification and len(np.unique(y_test)) == 2:
        # Binary classification: ROC curve and confusion matrix
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        axes[0].plot(fpr, tpr, color='darkorange', lw=2, 
                     label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve')
        axes[0].legend(loc="lower right")
        axes[0].grid(True, alpha=0.3)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
        axes[1].set_title('Confusion Matrix')
        axes[1].set_xlabel('Predicted Label')
        axes[1].set_ylabel('True Label')
        
    else:
        # Regression or multiclass: Actual vs Predicted
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Actual vs Predicted Values')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return {
        "plot": fig,
        "type": "model_evaluation",
        "is_classification": is_classification,
        "status": "success"
    }


def save_visualizations(
    feature_dist: Dict[str, Any],
    correlation: Dict[str, Any],
    model_eval: Dict[str, Any],
    params: Dict[str, Any]
) -> str:
    """Save all visualization plots to files.
    
    Args:
        feature_dist: Feature distribution plot data
        correlation: Correlation heatmap plot data
        model_eval: Model evaluation plot data
        params: Parameters including output directory
        
    Returns:
        Status message
    """
    logger.info("Saving visualization plots...")
    
    output_dir = Path(params.get("output_dir", "data/08_reporting"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    # Save feature distribution plot
    if feature_dist.get("status") == "success":
        filepath = output_dir / "feature_distributions.png"
        feature_dist["plot"].savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(feature_dist["plot"])
        saved_files.append(str(filepath))
        logger.info(f"Saved feature distributions to {filepath}")
    
    # Save correlation heatmap
    if correlation.get("status") == "success":
        filepath = output_dir / "correlation_heatmap.png"
        correlation["plot"].savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(correlation["plot"])
        saved_files.append(str(filepath))
        logger.info(f"Saved correlation heatmap to {filepath}")
    
    # Save model evaluation plot
    if model_eval.get("status") == "success":
        filepath = output_dir / "model_evaluation.png"
        model_eval["plot"].savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(model_eval["plot"])
        saved_files.append(str(filepath))
        logger.info(f"Saved model evaluation to {filepath}")
    
    return f"Successfully saved {len(saved_files)} visualization files to {output_dir}"
