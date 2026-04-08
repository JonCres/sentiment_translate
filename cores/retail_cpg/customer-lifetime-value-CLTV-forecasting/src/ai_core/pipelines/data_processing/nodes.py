import polars as pl
import logging
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path
from lifetimes.utils import summary_data_from_transaction_data
import pandera.polars as pa
from ...schemas import (
    SkeletonMappingParams,
    CleaningParams,
    RFMParams,
    FeastConfigParams,
    TransactionSkeletonSchema,
    RFMSchema,
    PIIParams  # Assuming this will be added to schemas
)
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger(__name__)


def mask_pii(data: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
    """
    Mask PII data using Microsoft Presidio.
    Scans specified text columns for PII entities and masks them.
    Also hashes specific identifier columns if configured.
    """
    # Validate params
    # config = PIIParams(**params)  # Uncomment when schema is ready
    
    # Initialize Presidio
    # Load only English to avoid warnings/downloads of other language models
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
    }
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(languages=["en"])
    
    analyzer = AnalyzerEngine(
        default_score_threshold=0.4, 
        registry=registry,
        nlp_engine=nlp_engine
    )
    anonymizer = AnonymizerEngine()
    
    # Identify columns to scan for PII (free text)
    text_cols_to_scan = params.get("text_columns", [])
    
    # Identify explicit PII columns to hash/drop (structured)
    pii_columns = params.get("pii_columns", ["email", "phone", "name"])
    
    # 1. Handle Structural PII (Hashing/Dropping)
    # For now, we'll hash them if they exist
    for col in pii_columns:
        if col in data.columns:
            logger.info(f"Hashing structured PII column: {col}")
            data = data.with_columns(
                pl.col(col).hash().cast(pl.Utf8).alias(col)
            )

    # 2. Handle Free Text PII (Masking)
    # Note: Presidio on large dataframes row-by-row is slow. 
    # For high volume, we might need a UDF or map_elements.
    if text_cols_to_scan:
        # Define masking function
        def anonymize_text(text: str) -> str:
            if not text:
                return text
            try:
                results = analyzer.analyze(text=text, entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS"], language='en')
                anonymized_result = anonymizer.anonymize(text=text, analyzer_results=results)
                return anonymized_result.text
            except Exception as e:
                logger.warning(f"PII masking failed for text: {e}")
                return text

        # Apply to each column
        # Warning: Performance bottleneck for millions of rows.
        # Ideally, this runs on a sample or batched.
        for col in text_cols_to_scan:
            if col in data.columns:
                logger.info(f"Scanning free text column for PII: {col}")
                data = data.with_columns(
                    pl.col(col).map_elements(anonymize_text, return_dtype=pl.Utf8).alias(col)
                )

    return data



def map_to_skeleton(data: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
    """
    Maps raw data columns to the mandatory skeleton column names.
    """
    # Validate params with Pydantic
    config = SkeletonMappingParams(**params)
    mapping = config.skeleton_mapping

    # Universal Mandatory (Non-Contractual)
    trans_map = mapping.get("transactional", {})

    # Conditional Mandatory (Contractual)
    cont_map = mapping.get("contractual", {})

    # Optional but High-Value (Muscle)
    muscle_map = mapping.get("muscle", {})

    # Combine all mappings
    full_mapping = {**trans_map, **cont_map, **muscle_map}

    # Inverse mapping for Polars rename (raw_name -> standard_name)
    rename_dict = {v: k for k, v in full_mapping.items() if v in data.columns}

    return data.rename(rename_dict)


@pa.check_output(TransactionSkeletonSchema.to_schema(), lazy=True)
def clean_data(data: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
    """
    Clean the data mapped to skeleton:
    - Convert date columns to date
    - Drop rows with missing Customer ID
    - Handle returns (profit correction)
    - Filter for positive quantity and price
    """
    # Validate params with Pydantic
    config = CleaningParams(**params)

    # Now we use standard skeleton names
    date_col = "transaction_date"
    customer_id_col = "customer_id"
    item_quantity_col = "transaction_count"
    item_value_col = "transaction_value"

    # Drop missing Customer IDs
    df = data.drop_nulls(subset=[customer_id_col])

    # Date formatting
    date_fmt = config.date_format

    if df.schema[date_col] == pl.String:
        if date_fmt:
            date_expr = pl.col(date_col).str.to_datetime(format=date_fmt, strict=False)
        else:
            date_expr = pl.col(date_col).str.to_datetime(strict=False)
    else:
        date_expr = pl.col(date_col).cast(pl.Datetime)

    df = df.with_columns(
        [
            date_expr.alias(date_col),
            pl.col(item_quantity_col).cast(pl.Float64),
            pl.col(item_value_col).cast(pl.Float64),
        ]
    )

    # Calculate TotalValue (Profit Correction)
    # Handle returns if return_flag is present
    if "return_flag" in df.columns:
        # If return_flag is boolean or 1/0
        df = df.with_columns(
            pl.when(pl.col("return_flag").cast(pl.Boolean))
            .then(-pl.col(item_quantity_col) * pl.col(item_value_col))
            .otherwise(pl.col(item_quantity_col) * pl.col(item_value_col))
            .alias("TotalValue")
        )
    else:
        df = df.with_columns(
            (pl.col(item_quantity_col) * pl.col(item_value_col)).alias("TotalValue")
        )

    # Filter out pure zeros or invalid data
    df = df.filter(pl.col("TotalValue") != 0.0)

    return df


@pa.check_io(
    data=TransactionSkeletonSchema.to_schema(),
    out=RFMSchema.to_schema(),
    lazy=True
)
def transform_transaction_data(
    data: pl.DataFrame, params: Dict[str, Any]
) -> pl.DataFrame:
    """
    Transform transaction data into RFM (Recency, Frequency, Monetary) format.
    Preserves acquisition_channel and calculates cohort_month.
    """
    # Validate params with Pydantic
    config = RFMParams(**params)
    
    date_col = "transaction_date"
    customer_id_col = "customer_id"
    observation_period_end = config.observation_period_end

    # 1. Calculate Metadata (Cohort and Channel)
    # cohort_month is the year-month of the first transaction
    metadata = data.group_by(customer_id_col).agg(
        [
            pl.col(date_col).min().dt.strftime("%Y-%m").alias("cohort_month"),
            pl.col("acquisition_channel").first().alias("acquisition_channel")
            if "acquisition_channel" in data.columns
            else pl.lit("Unknown").alias("acquisition_channel"),
        ]
    )

    # 2. Convert to Pandas for lifetimes utils
    data_pd = data.to_pandas()

    rfm_df = summary_data_from_transaction_data(
        data_pd,
        customer_id_col=customer_id_col,
        datetime_col=date_col,
        monetary_value_col="TotalValue",
        observation_period_end=observation_period_end,
    )
    # Convert back to Polars
    rfm_pl = pl.from_pandas(rfm_df.reset_index())

    # 3. Join with metadata
    return rfm_pl.join(metadata, on=customer_id_col, how="left")


def prepare_sequences(data: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
    """
    Prepare temporal sequences for LSTM modeling.
    Groups transactions by customer and creates a sequential feature set.
    """
    customer_id_col = "customer_id"
    date_col = "transaction_date"
    value_col = "TotalValue"

    # Ensure sorted by date
    data = data.sort([customer_id_col, date_col])

    # Aggregate to list of values per customer
    sequences = data.group_by(customer_id_col).agg(
        [
            pl.col(value_col).alias("value_sequence"),
            pl.col(date_col).alias("date_sequence"),
            pl.col(value_col).count().alias("sequence_length")
        ]
    )

    # Filter for minimum sequence length if required
    min_len = params.get("min_sequence_length", 2)
    sequences = sequences.filter(pl.col("sequence_length") >= min_len)

    logger.info(f"Generated sequences columns: {sequences.columns}")
    return sequences


def register_rfm_features_to_feast(
    processed_data: pl.DataFrame, feast_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Register RFM features to Feast Feature Store.

    Writes RFM features to Delta table and applies Feast feature definitions.

    Args:
        processed_data: RFM DataFrame from transaction processing.
        feast_config: Configuration with feature_repo_path and delta_path.

    Returns:
        Dict with registration status, timestamp, and feature count.
    """
    # Validate params with Pydantic
    config = FeastConfigParams(**feast_config)

    from feast import FeatureStore

    feature_repo_path = config.feature_repo_path
    delta_path = config.delta_path

    # 1. Add event_timestamp for Feast
    if "event_timestamp" not in processed_data.columns:
        processed_data = processed_data.with_columns(
            pl.lit(datetime.now()).alias("event_timestamp")
        )

    # 2. Write to Delta table
    full_path = Path(delta_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)

    processed_data.write_delta(
        str(full_path),
        mode="overwrite",
        delta_write_options={"schema_mode": "overwrite"},
    )
    logger.info(f"Wrote {processed_data.height} rows to Delta table: {delta_path}")

    # 3. Apply Feast definitions
    try:
        store = FeatureStore(repo_path=feature_repo_path)
        store.apply([])
        logger.info(f"Applied Feast feature definitions from {feature_repo_path}")
        feast_status = "success"
    except Exception as e:
        logger.warning(f"Feast apply skipped: {e}")
        feast_status = str(e)

    return {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "feature_count": processed_data.height,
        "delta_path": str(delta_path),
        "feature_repo_path": feature_repo_path,
        "feast_apply": feast_status,
    }