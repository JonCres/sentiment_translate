import polars as pl
from typing import Dict, Any, Tuple, Optional
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from lifetimes.utils import summary_data_from_transaction_data
import pandera.polars as pa
from ...schemas import TransactionSchema, RawTransactionSchema, SkeletonParams

from .skeleton import (
    map_to_transactions_skeleton,
    map_to_subscriptions_skeleton,
    map_to_engagement_skeleton,
    map_to_interaction_skeleton,
    map_to_qoe_skeleton,
    map_to_advertising_skeleton,
    map_to_tvod_skeleton,
)

logger = logging.getLogger(__name__)


@pa.check_output(RawTransactionSchema.to_schema(), lazy=True)
def create_transaction_skeleton(
    data: pl.DataFrame, params: Dict[str, Any]
) -> pl.DataFrame:
    """
    Node to convert raw data into the Transaction Skeleton.
    Wraps the logic in skeleton.py using parameters.
    """
    # Validate configuration using Pydantic
    config = SkeletonParams(**params)
    logger.info(f"Using configuration: {config.model_dump_json()}")

    return map_to_transactions_skeleton(data, params)


@pa.check_io(
    data=RawTransactionSchema.to_schema(), out=TransactionSchema.to_schema(), lazy=True
)
def clean_skeleton_data(data: pl.DataFrame) -> pl.DataFrame:
    """
    Clean the Skeleton Data:
    - Filter for positive amount_usd (returns/errors)
    - Drop missing Customer IDs (should be handled in skeleton but double check)
    """
    # Skeleton columns are fixed: customer_id, transaction_dt, amount_usd

    # Drop missing Customer IDs
    df = data.drop_nulls(subset=["customer_id"])

    # Filter out returns (negative value) and zero values
    df = df.filter(pl.col("amount_usd") > 0)

    return df


def transform_transaction_data(
    data: pl.DataFrame, params: Dict[str, Any]
) -> pl.DataFrame:
    """
    Transform transaction data into RFM (Recency, Frequency, Monetary) format.
    Uses lifetimes library which requires Pandas, so we convert temporarily.
    """
    # Skeleton columns are fixed
    date_col = "transaction_dt"
    customer_id_col = "customer_id"
    monetary_col = "amount_usd"

    observation_period_end = params.get("observation_period_end")

    # Convert to Pandas for lifetimes utils
    # This is necessary because re-implementing exact lifetimes logic
    # (handling limits, border cases) matches the library's expectations for the model.
    data_pd = data.to_pandas()

    rfm_df = summary_data_from_transaction_data(
        data_pd,
        customer_id_col=customer_id_col,
        datetime_col=date_col,
        monetary_value_col=monetary_col,
        observation_period_end=observation_period_end,
    )

    # Convert back to Polars
    # RFM df from lifetimes has Customer ID as index
    rfm_df = rfm_df.reset_index()

    # Calculate Churn Label
    # T = Age (Tenure), recency = Age at last purchase
    # Inactive duration = T - recency
    rfm_df["inactive_duration"] = rfm_df["T"] - rfm_df["recency"]

    # Threshold from params, default 90 days
    threshold = params.get("inactivity_threshold_days", 90)
    rfm_df["churn_label"] = (rfm_df["inactive_duration"] > threshold).astype(int)

    return pl.from_pandas(rfm_df)


def create_subscriptions_skeleton(
    data: pl.DataFrame, params: Dict[str, Any]
) -> pl.DataFrame:
    return map_to_subscriptions_skeleton(data, params)


def create_engagement_skeleton(
    data: pl.DataFrame, params: Dict[str, Any]
) -> pl.DataFrame:
    skeleton = map_to_engagement_skeleton(data, params)
    return skeleton if skeleton is not None else pl.DataFrame()


def create_interaction_skeleton(
    data: pl.DataFrame, params: Dict[str, Any]
) -> pl.DataFrame:
    return map_to_interaction_skeleton(data, params)


def create_qoe_skeleton(data: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
    return map_to_qoe_skeleton(data, params)


def create_advertising_skeleton(
    data: pl.DataFrame, params: Dict[str, Any]
) -> pl.DataFrame:
    return map_to_advertising_skeleton(data, params)


def create_tvod_skeleton(
    data: pl.DataFrame, params: Dict[str, Any]
) -> pl.DataFrame:
    return map_to_tvod_skeleton(data, params)


def create_survival_data(subscriptions: pl.DataFrame) -> pl.DataFrame:
    """
    Prepare data for sBG/Weibull: Calculate duration (T) and Event (E).
    Ensures customer_id is preserved.
    """

    # 1. Handle empty input while maintaining the customer_id column
    if subscriptions.is_empty():
        return pl.DataFrame(
            schema={
                "customer_id": subscriptions.schema.get("customer_id", pl.Utf8),
                "T": pl.Int64,
                "E": pl.Int32,
            }
        )

    # Rename subscription_id to customer_id to match expected column name
    if (
        "subscription_id" in subscriptions.columns
        and "customer_id" not in subscriptions.columns
    ):
        subscriptions = subscriptions.rename({"subscription_id": "customer_id"})

    # 2. Survival Logic (Placeholder)
    # This logic creates 'T' (Duration) and 'E' (Event/Censorship)
    survival_df = subscriptions.with_columns(
        [
            # Example: calculating duration in months between start and end date
            # (pl.col("end_date") - pl.col("start_date")).dt.total_days() // 30.alias("T"),
            # (pl.col("status") == "churned").cast(pl.Int32).alias("E")
            pl.lit(0, dtype=pl.Int64).alias("T"),
            pl.lit(0, dtype=pl.Int32).alias("E"),
        ]
    )

    # 3. Explicitly select the required columns to return
    return survival_df.select(["customer_id", "T", "E"])


def create_feature_store(
    transactions: pl.DataFrame,
    subscriptions: pl.DataFrame,
    engagement: Optional[pl.DataFrame] = None,
    interactions: Optional[pl.DataFrame] = None,
    qoe: Optional[pl.DataFrame] = None,
    advertising: Optional[pl.DataFrame] = None,
    tvod: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """
    Create a Feature Store for XGBoost Refinement and CLTV ensemble.
    Aggregates behavioral, interaction, and advertising data into customer-level features.
    """
    # Ensure consistent types for joining
    transactions = transactions.with_columns(pl.col("customer_id").cast(pl.String))
    subscriptions = subscriptions.with_columns(pl.col("customer_id").cast(pl.String))

    # Base is unique customers from transactions and subscriptions
    features = transactions.select(pl.col("customer_id")).unique()

    # 1. Join with subscription features (tenure, etc.)
    if not subscriptions.is_empty():
        sub_aggs = subscriptions.group_by("customer_id").agg(
            [
                pl.col("start_date").min().alias("first_subscription_date"),
                pl.count("subscription_id").alias("total_subscriptions"),
                pl.col("status").last().alias("current_status"),
            ]
        )
        features = features.join(sub_aggs, on="customer_id", how="left")

    # 2. Add Engagement Features
    if engagement is not None and not engagement.is_empty():
        logger.info("Aggregating engagement data...")
        pivoted_engagement = engagement.pivot(
            index="customer_id",
            columns="engagement_metric",
            values="engagement_value",
            aggregate_function="sum",
        ).fill_null(0)
        features = features.join(pivoted_engagement, on="customer_id", how="left")

    # 3. Add Interaction Features (watch-time velocity, binge)
    if interactions is not None and not interactions.is_empty():
        logger.info("Aggregating interaction data...")
        int_aggs = interactions.group_by("customer_id").agg(
            [
                pl.col("session_duration_min").sum().alias("total_watch_time_min"),
                pl.col("session_duration_min").mean().alias("avg_session_duration"),
                pl.count().alias("total_interactions"),
                pl.col("interaction_type")
                .filter(pl.col("interaction_type") == "binge")
                .count()
                .alias("binge_count"),
            ]
        )
        features = features.join(int_aggs, on="customer_id", how="left")

    # 4. Add Advertising Features (AVOD potential)
    if advertising is not None and not advertising.is_empty():
        logger.info("Aggregating advertising data...")
        ad_aggs = advertising.group_by("customer_id").agg(
            [
                pl.col("ad_completed").sum().alias("total_ads_completed"),
                pl.col("ad_duration_sec").sum().alias("total_ad_exposure_sec"),
                (pl.col("ad_completed").sum() / pl.count()).alias("ad_completion_rate"),
                pl.col("cpm_usd").mean().alias("avg_cpm"),
            ]
        )
        features = features.join(ad_aggs, on="customer_id", how="left")

    # 5. Add QoE Features (Technical friction)
    if qoe is not None and not qoe.is_empty():
        logger.info("Aggregating QoE data...")
        qoe_aggs = qoe.group_by("customer_id").agg(
            [
                pl.col("buffering_duration_sec").sum().alias("total_buffering_sec"),
                pl.col("startup_time_sec").mean().alias("avg_startup_time"),
                pl.col("error_count").sum().alias("total_playback_errors"),
                pl.col("bitrate_kbps").mean().alias("avg_bitrate"),
            ]
        )
        features = features.join(qoe_aggs, on="customer_id", how="left")

    # 6. Add TVOD Features (Transactional)
    if tvod is not None and not tvod.is_empty():
        logger.info("Aggregating TVOD data...")
        tvod_aggs = tvod.group_by("customer_id").agg(
            [
                pl.col("amount_usd").sum().alias("total_tvod_spend"),
                pl.count().alias("tvod_purchase_count"),
                pl.col("content_title").n_unique().alias("unique_tvod_titles"),
            ]
        )
        features = features.join(tvod_aggs, on="customer_id", how="left")

    return features.fill_null(0)


def register_churn_features_to_feast(
    processed_data: pl.DataFrame,
    survival_data: pl.DataFrame,
    feature_store_data: pl.DataFrame,
    feast_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Register Churn Prediction features to Feast Feature Store.

    Writes RFM, survival, and behavioral features to Delta tables
    and applies Feast feature definitions.

    Args:
        processed_data: RFM DataFrame from transaction processing.
        survival_data: Survival DataFrame (T, E) from subscription processing.
        feature_store_data: Behavioral features from engagement aggregation.
        feast_config: Configuration with feature_repo_path and delta paths.

    Returns:
        Dict with registration status for each feature set.
    """
    from feast import FeatureStore

    feature_repo_path = feast_config.get("feature_repo_path", "feature_repo")

    results = {}

    # Helper function to add event_timestamp and write Delta
    def write_features_to_delta(df: pl.DataFrame, delta_path: str, name: str) -> int:
        if df.is_empty():
            logger.warning(f"Skipping empty DataFrame for {name}")
            return 0

        if "event_timestamp" not in df.columns:
            df = df.with_columns(pl.lit(datetime.now()).alias("event_timestamp"))

        full_path = Path(delta_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)

        df.write_delta(
            str(full_path),
            mode="overwrite",
            delta_write_options={"schema_mode": "overwrite"},
        )
        logger.info(f"Wrote {df.height} rows to {delta_path}")
        return df.height

    # 1. Write RFM features
    rfm_path = feast_config.get("rfm_delta_path", "data/05_model_input/processed_data")
    results["rfm_count"] = write_features_to_delta(processed_data, rfm_path, "RFM")

    # 2. Write Survival features
    survival_path = feast_config.get(
        "survival_delta_path", "data/05_model_input/survival_data"
    )
    results["survival_count"] = write_features_to_delta(
        survival_data, survival_path, "Survival"
    )

    # 3. Write Behavioral features
    behavioral_path = feast_config.get(
        "behavioral_delta_path", "data/05_model_input/feature_store"
    )
    results["behavioral_count"] = write_features_to_delta(
        feature_store_data, behavioral_path, "Behavioral"
    )

    # 4. Apply Feast definitions
    try:
        store = FeatureStore(repo_path=feature_repo_path)
        store.apply([])
        logger.info(f"Applied Feast feature definitions from {feature_repo_path}")
        results["feast_apply"] = "success"
    except Exception as e:
        logger.warning(f"Feast apply skipped: {e}")
        results["feast_apply"] = str(e)

    results["status"] = "success"
    results["timestamp"] = datetime.now().isoformat()
    results["feature_repo_path"] = feature_repo_path

    return results


def create_tensor_sequences(
    engagement: pl.DataFrame, processed_data: pl.DataFrame, params: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create 3D tensor sequences for Deep Learning models (CNN-BiLSTM).
    Format: (Num_Customers, Sequence_Length, Num_Features)
    Also returns aligned labels.
    """
    import numpy as np

    if engagement.is_empty():
        return np.array([]), np.array([])

    # Parameters
    seq_len = (
        params.get("modeling", {}).get("deep_learning", {}).get("sequence_length", 90)
    )

    # 1. Sort by Customer and Date
    df = engagement.sort(["customer_id", "date"])

    # 2. Pivot or Collect features per day
    pivoted = df.pivot(
        index=["customer_id", "date"],
        columns="engagement_metric",
        values="engagement_value",
        aggregate_function="sum",
    ).fill_null(0)

    # 3. Join with Labels (processed_data has churn_label)
    # processed_data: customer_id, churn_label, ...
    # We need to ensure we only keep customers who have both engagement and a label.

    # Cast IDs to match
    pivoted = pivoted.with_columns(pl.col("customer_id").cast(pl.String))
    processed_data = processed_data.with_columns(pl.col("customer_id").cast(pl.String))

    # Get labels lookup
    labels_df = processed_data.select(["customer_id", "churn_label"]).unique()

    # feature columns (exclude ID and Date)
    feature_cols = [c for c in pivoted.columns if c not in ["customer_id", "date"]]

    customers = pivoted["customer_id"].unique().to_list()
    tensor_list = []
    labels_list = []

    for cust in customers:
        # Get Label
        cust_label_df = labels_df.filter(pl.col("customer_id") == cust)
        if cust_label_df.is_empty():
            continue  # Skip if no label found

        label = cust_label_df["churn_label"][0]

        # Get Data
        cust_df = pivoted.filter(pl.col("customer_id") == cust)
        # Extract numpy array of features
        feats = cust_df.select(feature_cols).to_numpy()

        # Pad or Truncate
        if len(feats) >= seq_len:
            # Take last N days
            seq = feats[-seq_len:]
        else:
            # Pad with zeros at the beginning (pre-padding)
            pad_len = seq_len - len(feats)
            padding = np.zeros((pad_len, len(feature_cols)))
            seq = np.vstack([padding, feats])

        tensor_list.append(seq)
        labels_list.append(label)

    return np.array(tensor_list), np.array(labels_list)
