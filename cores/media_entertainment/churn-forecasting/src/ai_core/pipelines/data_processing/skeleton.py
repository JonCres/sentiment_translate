import polars as pl
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def map_to_transactions_skeleton(
    data: pl.DataFrame, params: Dict[str, Any]
) -> pl.DataFrame:
    """
    Map raw data to the Mandatory Transaction Skeleton (Section 2.1).
    Enforces FACT_TRANSACTIONS schema.
    """
    transactions_params = params.get("transactions", {})
    mapping = transactions_params.get("mapping", {})
    date_format = transactions_params.get("date_format")

    # Mandatory columns for Backbone
    required_cols = [
        "customer_id",
        "transaction_date",
        "transaction_value",
        "transaction_id",
    ]

    # 1. Rename columns based on mapping
    # Invert mapping to rename from raw -> skeleton
    # mapping is skeleton_col -> raw_col
    # we need replace_map = raw_col -> skeleton_col

    rename_map = {v: k for k, v in mapping.items() if v in data.columns}

    # Check if all mandatory raw columns are present
    missing_raw = [
        v for k, v in mapping.items() if k in required_cols and v not in data.columns
    ]
    if missing_raw:
        raise ValueError(
            f"Missing mandatory backbone columns in raw data: {missing_raw}. Update parameters.yml mapping."
        )

    skeleton = data.rename(rename_map)

    # 2. Type Casting & Formatting

    # Helper for Date Parsing
    if "transaction_date" in skeleton.columns:
        if skeleton.schema["transaction_date"] == pl.String and date_format:
            skeleton = skeleton.with_columns(
                pl.col("transaction_date").str.to_datetime(
                    format=date_format, strict=False
                )
            )
        elif skeleton.schema["transaction_date"] != pl.Datetime:
            skeleton = skeleton.with_columns(
                pl.col("transaction_date").cast(pl.Datetime)
            )

    # Helper for Numeric Casting
    for col in ["transaction_value", "quantity"]:
        if col in skeleton.columns:
            skeleton = skeleton.with_columns(pl.col(col).cast(pl.Float64))

    # 3. Deduce Total Value (Amount USD)
    # The skeleton requires 'amount_usd'. If we mapped 'transaction_value' (Price) and 'quantity', compute it.
    if "amount_usd" not in skeleton.columns:
        if "transaction_value" in skeleton.columns and "quantity" in skeleton.columns:
            skeleton = skeleton.with_columns(
                (pl.col("transaction_value") * pl.col("quantity")).alias("amount_usd")
            )
        elif "transaction_value" in skeleton.columns:
            # Assume transaction_value IS the total amount if quantity missing
            skeleton = skeleton.with_columns(
                pl.col("transaction_value").alias("amount_usd")
            )

    # 4. Add static/missing columns for Schema Compliance
    if "transaction_type" not in skeleton.columns:
        skeleton = skeleton.with_columns(pl.lit("PURCHASE").alias("transaction_type"))

    # 5. Final Select strictly enforcing schema
    final_cols = [
        "customer_id",
        "transaction_id",
        "transaction_date",
        "amount_usd",
        "transaction_type",
    ]

    # Ensure all exist
    for c in final_cols:
        if c not in skeleton.columns:
            # Depending on strictness, we might error or fill null.
            # Backbone is strict.
            if c == "transaction_id":
                # If no ID, generate one or use index? Better to error for now or warn.
                logger.warning(
                    "Missing transaction_id, using row index as proxy not recommended for PROD."
                )
                skeleton = skeleton.with_row_index("transaction_id").with_columns(
                    pl.col("transaction_id").cast(pl.String)
                )
            else:
                raise ValueError(f"Failed to construct mandatory skeleton column: {c}")

    # Ensure customer_id is String for consistent joining
    skeleton = skeleton.with_columns(pl.col("customer_id").cast(pl.String))

    # Drop missing Customer IDs as they are essential for CLTV
    skeleton = skeleton.drop_nulls(subset=["customer_id"])

    return skeleton.select(final_cols).rename({"transaction_date": "transaction_dt"})


def map_to_subscriptions_skeleton(
    data: pl.DataFrame, params: Dict[str, Any]
) -> pl.DataFrame:
    """
    Map raw data to Subscription Skeleton.
    Implements Deductive Logic for Churn Status.
    """
    final_cols = [
        "customer_id",
        "subscription_id",
        "start_date",
        "end_date",
        "plan_type",
        "status",
        "acquisition_channel",
    ]

    if data.is_empty():
        return pl.DataFrame(
            schema={c: pl.String for c in final_cols}
        )  # Simplistic schema

    subscriptions_params = params.get("subscriptions", {})
    mapping = subscriptions_params.get("mapping", {})
    defaults = subscriptions_params.get("defaults", {})
    date_format = subscriptions_params.get("date_format")

    # 1. Rename columns based on mapping
    rename_map = {v: k for k, v in mapping.items() if v in data.columns}
    skeleton = data.rename(rename_map)

    # Ensure customer_id is String
    if "customer_id" in skeleton.columns:
        skeleton = skeleton.with_columns(pl.col("customer_id").cast(pl.String))

    # 2. Add missing columns with nulls/defaults
    if "subscription_id" not in skeleton.columns:
        # Use row index as proxy ID
        skeleton = skeleton.with_row_index("subscription_id").with_columns(
            pl.col("subscription_id").cast(pl.String)
        )

    if "plan_type" not in skeleton.columns:
        default_plan = defaults.get("plan_type", "Standard")
        skeleton = skeleton.with_columns(pl.lit(default_plan).alias("plan_type"))

    if "status" not in skeleton.columns:
        default_status = defaults.get("status", "active")
        skeleton = skeleton.with_columns(pl.lit(default_status).alias("status"))

    if "acquisition_channel" not in skeleton.columns:
        default_channel = defaults.get("acquisition_channel", "Unknown")
        skeleton = skeleton.with_columns(
            pl.lit(default_channel).alias("acquisition_channel")
        )

    # 3. Cast Dates
    for col in ["start_date", "end_date"]:
        if col in skeleton.columns:
            if skeleton.schema[col] == pl.String:
                if date_format:
                    skeleton = skeleton.with_columns(
                        pl.col(col).str.to_date(format=date_format, strict=False)
                    )
                else:
                    try:
                        skeleton = skeleton.with_columns(pl.col(col).str.to_date(strict=False))
                    except pl.ComputeError as e:
                        raise ValueError(f"Could not auto-parse date column '{col}'. Please specify a 'date_format' in parameters.yml for subscriptions.") from e
            elif skeleton.schema[col] in [pl.Int64, pl.Float64]:
                skeleton = skeleton.with_columns(
                    pl.col(col).cast(pl.Utf8).str.to_date(format="%Y%m%d", strict=False)
                )

    # 4. Cast Status
    if "status" in skeleton.columns:
        # standardizing to 1 (churned) 0 (active)
        # Assuming input is_cancel: 1=cancel
        if skeleton.schema["status"] == pl.String:
            skeleton = skeleton.with_columns(
                pl.when(pl.col("status").str.to_lowercase() == "active").then(0)
                .otherwise(pl.col("status"))
                .alias("status")
            )
        skeleton = skeleton.with_columns(pl.col("status").cast(pl.Int32))


    # Select final
    final_sel = [c for c in final_cols if c in skeleton.columns]

    return skeleton.select(final_sel)


def map_to_engagement_skeleton(
    data: pl.DataFrame, params: Dict[str, Any]
) -> pl.DataFrame:
    """
    Map raw data to Engagement Skeleton.
    Conditional Flesh: Returns Empty with Schema if data missing.
    """
    final_cols = ["customer_id", "date", "engagement_metric", "engagement_value"]

    if data.is_empty():
        return pl.DataFrame(schema={c: pl.String for c in final_cols})

    engagement_params = params.get("engagement", {})
    mapping = engagement_params.get("mapping", {})
    defaults = engagement_params.get("defaults", {})

    # Create expressions to build the final skeleton DataFrame
    expressions = []
    
    for skeleton_col in final_cols:
        raw_col_name_from_mapping = mapping.get(skeleton_col)
        
        if skeleton_col in data.columns:
            # If the raw data already has the column with the desired skeleton name, use it directly.
            expressions.append(pl.col(skeleton_col))
        elif raw_col_name_from_mapping and raw_col_name_from_mapping in data.columns:
            # If a raw column mapped to the skeleton name exists, rename it.
            expressions.append(pl.col(raw_col_name_from_mapping).alias(skeleton_col))
        else:
            # If neither exists, try to use a default value.
            default_val = defaults.get(skeleton_col)
            
            if default_val is not None:
                expressions.append(pl.lit(default_val).alias(skeleton_col))
            elif skeleton_col == "customer_id":
                # For customer_id, if no source or default, it's a critical error.
                logger.error(f"Mandatory column '{skeleton_col}' not found in raw data, not mapped, and no default provided.")
                raise ValueError(f"Mandatory column '{skeleton_col}' not found and cannot be derived or defaulted. Check parameters.yml mapping and defaults. Raw data columns available: {data.columns}")
            else:
                # For other columns, if no default is provided, it might be an issue
                logger.warning(f"Optional column '{skeleton_col}' not found, not mapped, and no default provided. It will be created with nulls.")
                expressions.append(pl.lit(None).alias(skeleton_col)) # Create with nulls if no default
                
    skeleton = data.select(expressions)

    # 3. Cast Types (apply after all columns are established)
    if "date" in skeleton.columns and skeleton.schema["date"] == pl.Utf8:
        skeleton = skeleton.with_columns(pl.col("date").str.to_date(strict=False))

    # Support for granular timestamps (Tier 3 Deep Learning)
    if "event_timestamp" in skeleton.columns:
         if skeleton.schema["event_timestamp"] == pl.Utf8:
             skeleton = skeleton.with_columns(pl.col("event_timestamp").str.to_datetime(strict=False))
    
    if "engagement_value" in skeleton.columns:
        # Cast to Float64. If the default was "engagement_volume", this will fail.
        # Assuming defaults will eventually be numeric for engagement_value.
        skeleton = skeleton.with_columns(pl.col("engagement_value").cast(pl.Float64, strict=False))

    return skeleton


def map_to_qoe_skeleton(
    data: pl.DataFrame, params: Dict[str, Any]
) -> pl.DataFrame:
    """
    Map raw data to Quality of Experience (QoE) Skeleton.
    Section 2.2 of AI Canvas.
    """
    final_cols = [
        "customer_id",
        "event_timestamp",
        "buffering_duration_sec",
        "startup_time_sec",
        "bitrate_kbps",
        "error_count",
        "rebuffer_ratio"
    ]

    if data.is_empty():
        return pl.DataFrame(schema={c: pl.Float64 if c not in ["customer_id", "event_timestamp"] else pl.String for c in final_cols})

    qoe_params = params.get("qoe", {})
    mapping = qoe_params.get("mapping", {})
    defaults = qoe_params.get("defaults", {})

    expressions = []
    
    for skeleton_col in final_cols:
        raw_col = mapping.get(skeleton_col)
        
        if skeleton_col in data.columns:
            expressions.append(pl.col(skeleton_col))
        elif raw_col and raw_col in data.columns:
            expressions.append(pl.col(raw_col).alias(skeleton_col))
        else:
            default_val = defaults.get(skeleton_col)
            if default_val is not None:
                expressions.append(pl.lit(default_val).alias(skeleton_col))
            else:
                # Essential columns must exist or have defaults, else null
                expressions.append(pl.lit(None).alias(skeleton_col))

    skeleton = data.select(expressions)

    # Type Casting
    if "event_timestamp" in skeleton.columns and skeleton.schema["event_timestamp"] == pl.Utf8:
        skeleton = skeleton.with_columns(pl.col("event_timestamp").str.to_datetime(strict=False))

    return skeleton


def map_to_social_graph_skeleton(
    data: pl.DataFrame, params: Dict[str, Any]
) -> pl.DataFrame:
    """
    Map raw data to Social Graph Skeleton.
    Section 4 (Graph Layer).
    """
    final_cols = ["source_node", "target_node", "weight", "edge_type"]

    if data.is_empty():
        return pl.DataFrame(schema={c: pl.String for c in final_cols})

    social_params = params.get("social_graph", {})
    mapping = social_params.get("mapping", {})
    
    expressions = []
    for skeleton_col in final_cols:
        raw_col = mapping.get(skeleton_col)
        if skeleton_col in data.columns:
            expressions.append(pl.col(skeleton_col))
        elif raw_col and raw_col in data.columns:
            expressions.append(pl.col(raw_col).alias(skeleton_col))
        else:
            expressions.append(pl.lit(None).alias(skeleton_col))

    return data.select(expressions)
