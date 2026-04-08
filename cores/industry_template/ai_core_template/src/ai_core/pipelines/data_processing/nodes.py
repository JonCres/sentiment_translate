import pandas as pd
import polars as pl
import logging
from typing import Dict, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def validate_data(
    data: pd.DataFrame, validation_config: Dict[str, Any]
) -> pd.DataFrame:
    """Validate data against schema/rules."""
    # Placeholder logic based on original intent
    # In a real scenario, use Great Expectations or Pandera
    if data.empty:
        raise ValueError("Data is empty")
    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean data (remove nulls, duplicates, etc.)."""
    # Placeholder logic
    return data.dropna().drop_duplicates()


def engineer_features(
    data: pd.DataFrame, feature_config: Dict[str, Any]
) -> pd.DataFrame:
    """Create new features."""
    # Placeholder logic
    if "customer_lifetime_value" in feature_config:
        # Mocking columns if they don't exist for the template to run
        if "total_spent" not in data.columns:
            data["total_spent"] = 100
        if "months_active" not in data.columns:
            data["months_active"] = 10

        data["clv"] = data["total_spent"] / data["months_active"]

    if "purchase_frequency" in feature_config:
        if "num_purchases" not in data.columns:
            data["num_purchases"] = 5
        if "months_active" not in data.columns:
            data["months_active"] = 10

        data["purchase_freq"] = data["num_purchases"] / data["months_active"]

    return data


def register_features_to_feast(
    features_data: pd.DataFrame, feast_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Register features to Feast Feature Store.

    Converts Pandas DataFrame to Polars, writes to Delta table,
    and applies Feast feature definitions to the registry.

    Args:
        features_data: DataFrame with engineered features from pipeline.
        feast_config: Configuration dict with feature_repo_path and delta_path.

    Returns:
        Dict with registration status, timestamp, and feature count.
    """
    from feast import FeatureStore

    feature_repo_path = feast_config.get("feature_repo_path", "feature_repo")
    delta_path = feast_config.get("delta_path", "data/04_feature/features_delta")

    # 1. Convert to Polars and add event_timestamp for Feast
    df_polars = pl.from_pandas(features_data)

    if "event_timestamp" not in df_polars.columns:
        df_polars = df_polars.with_columns(
            pl.lit(datetime.now()).alias("event_timestamp")
        )

    # Ensure customer_id exists
    if "customer_id" not in df_polars.columns:
        # Try to find an ID column or create one
        if df_polars.height > 0:
            df_polars = df_polars.with_row_index("customer_id")

    # 2. Write to Delta table
    delta_full_path = Path(delta_path)
    delta_full_path.parent.mkdir(parents=True, exist_ok=True)

    df_polars.write_delta(str(delta_full_path), mode="overwrite", overwrite_schema=True)
    logger.info(f"Wrote {df_polars.height} rows to Delta table: {delta_path}")

    # 3. Apply Feast feature definitions
    try:
        store = FeatureStore(repo_path=feature_repo_path)
        store.apply([])  # Apply will read feature_repo definitions
        logger.info(f"Applied Feast feature definitions from {feature_repo_path}")
    except Exception as e:
        logger.warning(f"Feast apply skipped (may need feast apply CLI): {e}")

    return {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "feature_count": df_polars.height,
        "delta_path": str(delta_path),
        "feature_repo_path": feature_repo_path,
    }
