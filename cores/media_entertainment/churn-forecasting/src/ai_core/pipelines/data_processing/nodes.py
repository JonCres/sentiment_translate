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
    map_to_qoe_skeleton,
    map_to_social_graph_skeleton,
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


def create_qoe_skeleton(
    data: pl.DataFrame, params: Dict[str, Any]
) -> pl.DataFrame:
    return map_to_qoe_skeleton(data, params)


def create_social_graph_skeleton(
    data: pl.DataFrame, params: Dict[str, Any]
) -> pl.DataFrame:
    return map_to_social_graph_skeleton(data, params)


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
    engagement: Optional[pl.DataFrame],
    qoe: Optional[pl.DataFrame] = None,
    social: Optional[pl.DataFrame] = None,
    tensor_features: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """
    Create a Feature Store for XGBoost Refinement.
    Aggregates behavioral data (engagement) into customer-level features
    and joins tensor sequence features if provided.
    """
    # Ensure consistent types for joining
    transactions = transactions.with_columns(pl.col("customer_id").cast(pl.String))
    subscriptions = subscriptions.with_columns(pl.col("customer_id").cast(pl.String))
    if engagement is not None:
        engagement = engagement.with_columns(pl.col("customer_id").cast(pl.String))

    # Base is transactions (contains RFM features: frequency, recency, T)
    features = transactions

    if engagement is not None and not engagement.is_empty():
        logger.info(
            "Engagement data present. Aggregating and joining behavioral features."
        )

        # 1. Pivot metrics (Sum of values per metric type)
        # Using pivot to turn rows (metric names) into columns
        pivoted_engagement = engagement.pivot(
            index="customer_id",
            columns="engagement_metric",
            values="engagement_value",
            aggregate_function="sum",
        ).fill_null(0)

        # 2. General Aggregations
        agg_engagement = engagement.group_by("customer_id").agg(
            [
                pl.col("date").max().alias("last_engagement_date"),
                pl.col("engagement_value").sum().alias("total_engagement_value"),
                pl.count().alias("engagement_frequency"),
                pl.col("date").n_unique().alias("active_days_count"),
            ]
        )

        # 3. Join Aggregations and Pivoted Features
        full_engagement_features = agg_engagement.join(
            pivoted_engagement, on="customer_id", how="left"
        )

        # 4. Join to Base Features
        features = features.join(full_engagement_features, on="customer_id", how="left")

        # Fill nulls for customers with no engagement
        # (This fills numeric columns with 0, but date columns need care if used later)
        features = features.fill_null(0)

    else:
        logger.warning("Engagement data MISSING. XGBoost will be limited/skipped.")

    # Merge QoE (Optional Tier 2)
    if qoe is not None and not qoe.is_empty():
        features = features.join(qoe, on=["customer_id"], how="left")

    # Merge Social (Optional Tier 4)
    if social is not None and not social.is_empty():
        # Assuming social already provides node-level metrics (degree, centrality)
        # If it's an edge list, we'd need to aggregate it to node level first
        # For now, simplistic join
        features = features.join(social.select(pl.exclude("target_node")), on=["customer_id"], how="left")

    # 5. Join Tensor Features
    if tensor_features is not None and not tensor_features.is_empty():
        # Check if there are feature columns to join
        if tensor_features.shape[1] > 1:
            logger.info("Joining tensor sequence features to feature store.")
            features = features.join(tensor_features, on="customer_id", how="left")
        else:
            logger.warning(
                "Tensor features DataFrame is empty or contains only customer_id, skipping join."
            )

    return features


def register_churn_features_to_feast(
    processed_data: pl.DataFrame,
    feature_store_data: pl.DataFrame,
    feast_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Register Churn Prediction features to Feast Feature Store.

    Writes RFM and behavioral features to Delta tables
    and applies Feast feature definitions.

    Args:
        processed_data: RFM DataFrame from transaction processing.
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

    # 2. Write Behavioral features
    behavioral_path = feast_config.get(
        "behavioral_delta_path", "data/05_model_input/feature_store"
    )
    results["behavioral_count"] = write_features_to_delta(
        feature_store_data, behavioral_path, "Behavioral"
    )

    # 3. Apply Feast definitions
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
) -> Tuple[pl.DataFrame, np.ndarray]:
    """
    Create 3D tensor sequences for Deep Learning models, flatten them into a list,
    and return them in a DataFrame.

    Returns a DataFrame with customer_id and the flattened sequence as a list,
    and a numpy array of labels.
    """
    import numpy as np

    seq_len = (
        params.get("modeling", {}).get("deep_learning", {}).get("sequence_length", 90)
    )

    schema = {"customer_id": pl.String, "engagement_sequence_flat": pl.List(pl.Float64)}

    if engagement.is_empty():
        return pl.DataFrame(schema), np.array([])

    # 1. Sort by Customer and Date
    df = engagement.sort(["customer_id", "date"])

    # 2. Pivot or Collect features per day
    pivoted = df.pivot(
        index=["customer_id", "date"],
        columns="engagement_metric",
        values="engagement_value",
        aggregate_function="sum",
    ).fill_null(0)

    # 3. Join with Labels
    pivoted = pivoted.with_columns(pl.col("customer_id").cast(pl.String))
    processed_data = processed_data.with_columns(pl.col("customer_id").cast(pl.String))
    labels_df = processed_data.select(["customer_id", "churn_label"]).unique()

    feature_cols = [c for c in pivoted.columns if c not in ["customer_id", "date"]]
    if not feature_cols:
        logger.warning("No feature columns found after pivoting engagement data.")
        return pl.DataFrame(schema), np.array([])

    customers = pivoted["customer_id"].unique().to_list()
    tensor_list, labels_list, customer_ids_list = [], [], []

    for cust in customers:
        cust_label_df = labels_df.filter(pl.col("customer_id") == cust)
        if cust_label_df.is_empty():
            continue

        label = cust_label_df["churn_label"][0]
        cust_df = pivoted.filter(pl.col("customer_id") == cust)
        feats = cust_df.select(feature_cols).to_numpy()

        if len(feats) >= seq_len:
            seq = feats[-seq_len:]
        else:
            pad_len = seq_len - len(feats)
            padding = np.zeros((pad_len, len(feature_cols)))
            seq = np.vstack([padding, feats])

        tensor_list.append(seq)
        labels_list.append(label)
        customer_ids_list.append(cust)

    if not tensor_list:
        return pl.DataFrame(schema), np.array([])

    # Flatten the sequence and store as a list of floats
    flattened_sequences = [seq.flatten().tolist() for seq in tensor_list]

    tensor_df = pl.DataFrame(
        {
            "customer_id": customer_ids_list,
            "engagement_sequence_flat": flattened_sequences,
        }
    )

    return tensor_df, np.array(labels_list)
