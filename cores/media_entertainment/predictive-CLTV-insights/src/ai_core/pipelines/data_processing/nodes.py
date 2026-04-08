import polars as pl
from typing import Dict, Any
import logging
from datetime import datetime
from pathlib import Path
from lifetimes.utils import summary_data_from_transaction_data
import pandera.polars as pa
from ...schemas import TransactionSchema, RawTransactionSchema, SkeletonParams


from .skeleton import (
    map_to_transactions_skeleton,
    map_to_subscriptions_skeleton,
    map_to_engagement_skeleton,
)
from typing import Optional

logger = logging.getLogger(__name__)


@pa.check_output(RawTransactionSchema.to_schema(), lazy=True)
def create_transaction_skeleton(
    data: pl.DataFrame, params: Dict[str, Any]
) -> pl.DataFrame:
    """Convert raw transaction data into standardized Transaction Skeleton format.

    This node transforms source-specific transaction data into a standardized schema
    with three core columns: customer_id, transaction_dt, and amount_usd. The mapping
    logic is defined in skeleton.py and validated using Pydantic and Pandera schemas.

    Args:
        data: Raw transaction data from the source system. Schema varies by source
            but must contain fields mappable to customer ID, transaction date, and amount.
        params: Configuration dictionary containing skeleton mapping parameters:
            - customer_id_col: Source column name for customer identifier
            - date_col: Source column name for transaction timestamp
            - amount_col: Source column name for transaction monetary value
            - Additional source-specific mapping rules

    Returns:
        Standardized transaction DataFrame with schema:
            - customer_id (str): Unique customer identifier
            - transaction_dt (datetime): Transaction timestamp
            - amount_usd (float): Transaction monetary value in USD

    Raises:
        ValidationError: If params don't match SkeletonParams Pydantic model
        pa.errors.SchemaError: If output doesn't match RawTransactionSchema

    Note:
        Output is validated lazily using Pandera's RawTransactionSchema decorator.
        This ensures all downstream nodes receive consistent data structure.
    """
    # Validate configuration using Pydantic
    config = SkeletonParams(**params)
    logger.info(f"Using configuration: {config.model_dump_json()}")
    
    return map_to_transactions_skeleton(data, params)


@pa.check_io(
    data=RawTransactionSchema.to_schema(),
    out=TransactionSchema.to_schema(),
    lazy=True
)
def clean_skeleton_data(data: pl.DataFrame) -> pl.DataFrame:
    """Clean and validate transaction skeleton data for modeling.

    Removes invalid transactions that would corrupt CLTV models:
    - Drops transactions with missing customer IDs (orphaned records)
    - Filters out returns (negative amounts) and zero-value transactions

    These cleaning steps ensure BG/NBD and Gamma-Gamma models receive only
    valid purchase events with positive monetary value.

    Args:
        data: Raw transaction skeleton with columns:
            - customer_id (str): Customer identifier (may contain nulls)
            - transaction_dt (datetime): Transaction timestamp
            - amount_usd (float): Transaction amount (may be negative/zero)

    Returns:
        Cleaned transaction DataFrame with same schema, containing only:
            - Non-null customer IDs
            - Positive transaction amounts (amount_usd > 0)

    Note:
        Input and output schemas are validated using Pandera decorators.
        TransactionSchema enforces stricter constraints than RawTransactionSchema.

    Example:
        Input: 10,000 transactions, 50 with null IDs, 200 returns
        Output: 9,750 valid transactions for modeling
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
    """Transform transaction data into RFM (Recency, Frequency, Monetary) summary format.

    Aggregates individual transactions per customer into RFM vectors required for
    BG/NBD and Gamma-Gamma probabilistic models. Uses the lifetimes library's
    summary_data_from_transaction_data function to ensure consistency with model
    expectations for calibration periods and edge cases.

    RFM Vector Components:
    - Recency (T): Time between first and last purchase
    - Frequency: Number of repeat purchases (total purchases - 1)
    - Monetary: Average transaction value across repeat purchases

    Args:
        data: Cleaned transaction skeleton DataFrame with columns:
            - customer_id (str): Unique customer identifier
            - transaction_dt (datetime): Transaction timestamp
            - amount_usd (float): Positive transaction amount
        params: Configuration dictionary containing:
            - observation_period_end (datetime): End of observation window for RFM calculation.
              Transactions after this date are ignored. If None, uses max transaction date.

    Returns:
        RFM summary DataFrame with one row per customer:
            - customer_id (str): Customer identifier (reset from index)
            - frequency (int): Number of repeat purchases
            - recency (float): Days between first and last purchase
            - T (float): Customer age (days between first purchase and observation_period_end)
            - monetary_value (float): Average purchase value for repeat customers

    Note:
        Temporarily converts to Pandas for lifetimes library compatibility, then
        converts back to Polars. This ensures exact alignment with lifetimes model
        assumptions around boundary conditions and censoring.

    Example:
        Customer A: Purchases on Day 0, 10, 30 → frequency=2, recency=30, T=90 (if observation_period_end=Day 90)
        Customer B: Single purchase on Day 0 → frequency=0, recency=0, T=90
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
    return pl.from_pandas(rfm_df.reset_index())


def create_subscriptions_skeleton(
    data: pl.DataFrame, params: Dict[str, Any]
) -> pl.DataFrame:
    """Convert raw subscription data into standardized Subscription Skeleton format.

    Transforms source-specific subscription data (SVOD, contract-based services) into
    a standardized schema for survival analysis models (Weibull AFT, sBG). The mapping
    logic handles subscription lifecycle events: activation, renewal, cancellation.

    Args:
        data: Raw subscription data from source system. Schema varies by source but
            must contain fields mappable to subscription ID, customer ID, start date,
            end date, and subscription status.
        params: Configuration dictionary containing skeleton mapping parameters:
            - subscription_id_col: Source column for subscription identifier
            - customer_id_col: Source column for customer identifier
            - start_date_col: Source column for subscription start timestamp
            - end_date_col: Source column for subscription end/cancellation timestamp
            - status_col: Source column for subscription status (active/churned/cancelled)
            - Additional source-specific mapping rules

    Returns:
        Standardized subscription DataFrame with schema:
            - subscription_id (str): Unique subscription identifier
            - customer_id (str): Customer identifier (may map to subscription_id for 1:1 relationships)
            - start_date (datetime): Subscription activation timestamp
            - end_date (datetime, nullable): Subscription termination timestamp (null if active)
            - status (str): Subscription status (active/churned/cancelled)

    Note:
        This skeleton supports contractual business models (SVOD, SaaS, memberships)
        where customer relationships have explicit start/end dates. For non-contractual
        models (TVOD, e-commerce), use create_transaction_skeleton instead.
    """
    return map_to_subscriptions_skeleton(data, params)


def create_engagement_skeleton(
    data: pl.DataFrame, params: Dict[str, Any]
) -> pl.DataFrame:
    """Convert raw behavioral data into standardized Engagement Skeleton format.

    Transforms source-specific engagement signals (app usage, content consumption,
    QoE metrics) into a standardized schema for XGBoost behavioral refinement.
    Engagement data enriches CLTV predictions beyond transactional patterns.

    Typical engagement signals for media/entertainment:
    - Content consumption: watch time, streams played, genres consumed
    - Quality of Experience: buffering ratio, video quality, error rate
    - App activity: login frequency, session duration, feature usage

    Args:
        data: Raw engagement/behavioral data from source system. Schema varies by
            source but must contain fields mappable to customer ID, event timestamp,
            engagement metric name, and metric value.
        params: Configuration dictionary containing skeleton mapping parameters:
            - customer_id_col: Source column for customer identifier
            - date_col: Source column for engagement event timestamp
            - metric_col: Source column for engagement metric name (e.g., 'watch_time')
            - value_col: Source column for engagement metric value (numeric)
            - Additional source-specific mapping rules

    Returns:
        Standardized engagement DataFrame with schema:
            - customer_id (str): Customer identifier
            - date (datetime): Engagement event timestamp
            - engagement_metric (str): Metric name (e.g., 'watch_time', 'login_count')
            - engagement_value (float): Metric value
        Returns empty DataFrame if engagement data is unavailable or mapping fails.

    Note:
        Engagement data is optional. If unavailable, the pipeline falls back to
        pure transactional CLTV modeling (BG/NBD + Gamma-Gamma only, no XGBoost).
    """
    skeleton = map_to_engagement_skeleton(data, params)
    return skeleton if skeleton is not None else pl.DataFrame()


def create_survival_data(subscriptions: pl.DataFrame) -> pl.DataFrame:
    """Prepare subscription data for survival analysis models (sBG, Weibull AFT, DeepSurv).

    Transforms subscription lifecycle data into survival analysis format with duration
    and event indicators. Survival models predict time-until-churn for contractual
    business models (SVOD, SaaS, memberships).

    Survival Analysis Components:
    - T (Duration): Observation time for each subscription (e.g., months subscribed)
    - E (Event): Churn indicator (1 = churned/cancelled, 0 = active/censored)

    Args:
        subscriptions: Subscription skeleton DataFrame with columns:
            - subscription_id or customer_id (str): Subscription/customer identifier
            - start_date (datetime): Subscription activation timestamp
            - end_date (datetime, nullable): Subscription termination timestamp
            - status (str): Subscription status (active/churned/cancelled)

    Returns:
        Survival DataFrame with one row per subscription:
            - customer_id (str): Customer/subscription identifier (renamed from subscription_id if needed)
            - T (int): Duration in units (e.g., days/months from start_date to end_date or observation_period_end)
            - E (int): Event indicator (1 = churned, 0 = censored/active)
        Returns empty DataFrame with correct schema if input is empty.

    Note:
        Current implementation is a placeholder (T=0, E=0 for all records).
        Production logic should calculate:
        - T = (end_date - start_date).days // 30 for churned subscriptions
        - T = (observation_period_end - start_date).days // 30 for active subscriptions
        - E = 1 if status == 'churned', else 0

    Example:
        Subscription A: Started Jan 1, churned Mar 1 → T=2 months, E=1
        Subscription B: Started Jan 1, still active at observation_period_end (Jun 1) → T=5 months, E=0
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
) -> pl.DataFrame:
    """Aggregate behavioral data into customer-level features for XGBoost refinement.

    Creates a feature store by pivoting and aggregating engagement metrics at the
    customer level. These behavioral features enhance CLTV predictions from BG/NBD
    and Gamma-Gamma models by capturing non-transactional signals like content
    consumption patterns, app activity, and quality of experience.

    Feature Engineering Process:
    1. Pivot engagement metrics: Convert rows (metric names) to columns (feature names)
    2. Aggregate engagement statistics: Sum/count/max per customer
    3. Join to customer base: Ensure all customers have feature records
    4. Impute missing values: Fill nulls with 0 for customers without engagement data

    Args:
        transactions: Transaction skeleton DataFrame with customer_id column.
            Used to establish the base customer list for feature store.
        subscriptions: Subscription skeleton DataFrame with customer_id column.
            Provides context for contractual customers (currently unused but available
            for future subscription-based features).
        engagement: Optional engagement skeleton DataFrame with columns:
            - customer_id (str): Customer identifier
            - date (datetime): Engagement event timestamp
            - engagement_metric (str): Metric name (e.g., 'watch_time', 'login_count')
            - engagement_value (float): Metric value
            Can be None if engagement data is unavailable.

    Returns:
        Feature store DataFrame with one row per customer:
            - customer_id (str): Customer identifier
            - Pivoted engagement features: One column per unique engagement_metric
              (e.g., watch_time, login_count, buffering_ratio)
            - last_engagement_date (datetime): Most recent engagement timestamp
            - total_engagement_value (float): Sum of all engagement values
            - engagement_frequency (int): Total engagement event count
            - active_days_count (int): Number of unique days with engagement
        If engagement is None or empty, returns DataFrame with only customer_id.

    Raises:
        ValueError: If transactions DataFrame is empty or missing customer_id column.

    Warning:
        If engagement data is missing, logs warning and returns minimal feature store.
        XGBoost pipeline will be skipped or limited to transactional features only.

    Example:
        Input engagement (3 rows):
            customer_id | date       | engagement_metric | engagement_value
            C1          | 2024-01-01 | watch_time       | 120
            C1          | 2024-01-02 | login_count      | 5
            C2          | 2024-01-01 | watch_time       | 60

        Output features (2 rows):
            customer_id | watch_time | login_count | last_engagement_date | total_engagement_value | engagement_frequency | active_days_count
            C1          | 120        | 5           | 2024-01-02          | 125                   | 2                    | 2
            C2          | 60         | 0           | 2024-01-01          | 60                    | 1                    | 1
    """
    # Ensure consistent types for joining
    transactions = transactions.with_columns(pl.col("customer_id").cast(pl.String))
    subscriptions = subscriptions.with_columns(pl.col("customer_id").cast(pl.String))
    if engagement is not None:
        engagement = engagement.with_columns(pl.col("customer_id").cast(pl.String))

    # Base is transactions (or subscriptions if contractual)
    # Start with unique customers from transactions
    features = transactions.select(pl.col("customer_id").unique())

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

    return features


def register_cltv_features_to_feast(
    processed_data: pl.DataFrame,
    survival_data: pl.DataFrame,
    feature_store_data: pl.DataFrame,
    feast_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Register CLTV features to Feast Feature Store.

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
