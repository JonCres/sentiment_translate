import polars as pl
import numpy as np
import re
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import pandera.polars as pa
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
from ...schemas import (
    VoCSkeletonSchema,
    ReviewFeaturesSchema,
    MappingParams,
    ValidationParams,
    CleaningParams,
    FeatureParams,
    TranslationParams
)

logger = logging.getLogger(__name__)
DetectorFactory.seed = 0


@pa.check_output(VoCSkeletonSchema.to_schema(), lazy=True)
def map_to_skeleton(df: pl.DataFrame, mapping_params: Dict[str, Any]) -> pl.DataFrame:
    """
    Map source data columns to the mandatory VoC skeleton schema.

    This function standardizes any source data format to the mandatory
    variables defined in the Voice of Customer AI walkthrough (Section 2).

    Mandatory Skeleton Variables:
    - Interaction_ID: Unique identifier for the interaction
    - Interaction_Payload: Raw content (text/audio/video)
    - Customer_ID: Customer identifier
    - Timestamp: Interaction datetime
    - Channel_ID: Source channel (Web_Review, Call_Center, etc.)
    - Target_Object_ID: Product/service identifier
    - Language_Code: ISO 639-1 language code
    - Rating: Numeric score if available

    Args:
        df: Source DataFrame with original column names
        mapping_params: Configuration with column mappings

    Returns:
        DataFrame with standardized skeleton column names
    """
    logger.info("Mapping source columns to VoC skeleton schema...")

    # Validate params with Pydantic
    params = MappingParams(**mapping_params)
    
    mandatory_mapping = params.mandatory
    optional_mapping = params.optional
    defaults = params.defaults

    # Start with original df
    result_df = df.clone()
    columns_mapped = []
    columns_defaulted = []

    # Process mandatory columns
    for skeleton_col, source_col in mandatory_mapping.items():
        if source_col and source_col in df.columns:
            # Map existing column to skeleton name
            result_df = result_df.with_columns(pl.col(source_col).alias(skeleton_col))
            columns_mapped.append(f"{source_col} → {skeleton_col}")
        elif skeleton_col in defaults:
            # Set default value
            default_val = defaults.get(skeleton_col, "unknown")
            result_df = result_df.with_columns(pl.lit(default_val).alias(skeleton_col))
            columns_defaulted.append(f"{skeleton_col} = '{default_val}'")
        else:
            # Check if skeleton column already exists
            if skeleton_col not in df.columns:
                logger.warning(
                    f"Mandatory column {skeleton_col} has no source mapping and no default"
                )

    # Process optional columns
    for skeleton_col, source_col in optional_mapping.items():
        if source_col and source_col in df.columns:
            result_df = result_df.with_columns(pl.col(source_col).alias(skeleton_col))
            columns_mapped.append(f"{source_col} → {skeleton_col} (optional)")

    # Apply default values for Channel_ID and Language_Code if not mapped
    if "Channel_ID" not in result_df.columns:
        result_df = result_df.with_columns(
            pl.lit(defaults.get("Channel_ID", "Web_Review")).alias("Channel_ID")
        )
        columns_defaulted.append("Channel_ID = 'Web_Review'")

    if "Language_Code" not in result_df.columns:
        result_df = result_df.with_columns(
            pl.lit(defaults.get("Language_Code", "en")).alias("Language_Code")
        )
        columns_defaulted.append("Language_Code = 'en'")

    # Log mapping summary
    logger.info(f"Column mappings applied: {len(columns_mapped)}")
    for mapping in columns_mapped:
        logger.debug(f"  {mapping}")
    logger.info(f"Default values set: {len(columns_defaulted)}")
    for default in columns_defaulted:
        logger.debug(f"  {default}")

    # Validate mandatory columns exist
    mandatory_cols = [
        "Interaction_ID",
        "Interaction_Payload",
        "Customer_ID",
        "Timestamp",
        "Target_Object_ID",
        "Rating",
    ]
    missing = [col for col in mandatory_cols if col not in result_df.columns]
    if missing:
        logger.warning(f"Missing mandatory skeleton columns after mapping: {missing}")

    logger.info(
        f"Skeleton mapping complete: {len(result_df)} records, {len(result_df.columns)} columns"
    )

    return result_df


def load_reviews_from_csv(
    raw_reviews: pl.DataFrame, params: Dict[str, Any]
) -> pl.DataFrame:
    """
    Load and sample Amazon reviews from CSV file.

    The raw_reviews DataFrame comes from the Kedro catalog (amazon_reviews_raw).
    This function applies optional sampling based on the parameters.

    Args:
        raw_reviews: DataFrame loaded from CSV via catalog
        params: Source parameters including optional sample_size

    Returns:
        DataFrame with reviews (sampled if sample_size is specified)
    """
    # In Polars, lazy execution is often preferred, but raw_reviews might be eager depending on catalog.
    # Assuming eager for now based on typical types.
    logger.info(f"Loading {len(raw_reviews)} reviews from CSV...")

    sample_size = params.get("sample_size", None)

    if sample_size and sample_size < len(raw_reviews):
        logger.info(f"Sampling {sample_size} reviews from {len(raw_reviews)} total")
        df = raw_reviews.sample(n=sample_size, seed=42)
    else:
        df = raw_reviews

    logger.info(f"Loaded {len(df)} reviews")
    return df


def generate_synthetic_reviews(n_reviews: int = 5000) -> pl.DataFrame:
    """Generate synthetic Amazon reviews for demo purposes"""

    np.random.seed(42)

    # Sample product categories
    categories = ["Electronics", "Books", "Home & Kitchen", "Clothing", "Sports"]

    # Sample review templates
    positive_templates = [
        "Great product! Exactly what I needed. {detail} Highly recommend!",
        "Excellent quality. {detail} Worth every penny!",
        "Amazing! {detail} Will definitely buy again.",
        "Perfect! {detail} Exceeded my expectations.",
        "Love it! {detail} Best purchase ever!",
    ]

    neutral_templates = [
        "It's okay. {detail} Does what it's supposed to do.",
        "Average product. {detail} Nothing special but works fine.",
        "Decent. {detail} Gets the job done.",
        "It's fine. {detail} As described.",
    ]

    negative_templates = [
        "Disappointed. {detail} Would not recommend.",
        "Not as expected. {detail} Quality is poor.",
        "Terrible experience. {detail} Waste of money.",
        "Very unhappy. {detail} Returned it immediately.",
        "Poor quality. {detail} Broke after a week.",
    ]

    details = [
        "Fast shipping and good packaging.",
        "Easy to use and set up.",
        "Material feels durable.",
        "Size is perfect for my needs.",
        "Color matches the description.",
        "Instructions were clear.",
        "Customer service was helpful.",
        "Price is reasonable.",
    ]

    data = []

    for i in range(n_reviews):
        rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3])

        if rating >= 4:
            template = np.random.choice(positive_templates)
        elif rating == 3:
            template = np.random.choice(neutral_templates)
        else:
            template = np.random.choice(negative_templates)

        detail = np.random.choice(details)
        review_text = template.format(detail=detail)

        data.append(
            {
                "review_id": f"review_{i}",
                "product_id": f"product_{np.random.randint(1, 1000)}",
                "review_text": review_text,
                "review_title": f"{'Great' if rating >= 4 else 'Poor' if rating <= 2 else 'Okay'} Product",
                "rating": rating,
                # Use standard python datetime, polars handles conversion
                "date": np.datetime64("now")
                - np.random.randint(0, 365) * np.timedelta64(1, "D"),
                "verified_purchase": bool(
                    np.random.choice([True, False], p=[0.8, 0.2])
                ),
                "helpful_votes": int(np.random.randint(0, 50)),
                "category": np.random.choice(categories),
            }
        )

    df = pl.DataFrame(data)
    # Ensure date is datetime
    df = df.with_columns(pl.col("date").cast(pl.Datetime))

    return df


@pa.check_io(df=VoCSkeletonSchema.to_schema(), out=VoCSkeletonSchema.to_schema(), lazy=True)
def translate_reviews(df: pl.DataFrame, translation_params: Dict[str, Any]) -> pl.DataFrame:
    """
    Translate review text to target language using deep-translator.

    Args:
        df: DataFrame with skeleton columns.
        translation_params: Configuration for translation.

    Returns:
        DataFrame with translated Interaction_Payload and updated Language_Code.
    """
    params = TranslationParams(**translation_params)

    if not params.enabled:
        logger.info("Translation disabled, skipping.")
        return df

    target_lang = params.target_language
    logger.info(f"Processing translations (target={target_lang}, detection=enabled)...")

    # Initialize Translator (we'll reuse it but instantiated inside processing function often better for serialization if needed, though we run locally here)
    # Actually, initializing once is better for performance if not pickled.
    translator = GoogleTranslator(source="auto", target=target_lang)

    def process_text_struct(val: str) -> Dict[str, str]:
        """
        Detects language and translates if necessary.
        Returns struct with 'text' and 'lang'.
        """
        if not val or len(val.strip()) == 0:
            return {"text": val, "lang": "unknown"}
        
        try:
            detected_lang = detect(val)
        except Exception:
            detected_lang = "unknown"
            
        # If detected language is NOT target language, translate
        # Also, if unknown, we might skip or try to translate. Here we skip if unknown.
        if detected_lang != target_lang and detected_lang != "unknown":
            try:
                # Check length limit
                text_to_trans = val[:4999] if len(val) > 4999 else val
                translated_text = translator.translate(text_to_trans)
                return {"text": translated_text, "lang": target_lang} # It is now target lang
            except Exception as e:
                # Fallback to original
                return {"text": val, "lang": detected_lang}
        else:
            # Already target lang or unknown
            return {"text": val, "lang": detected_lang if detected_lang != "unknown" else target_lang}

    # Apply struct mapping
    result_struct = df.select(
        pl.col("Interaction_Payload").map_elements(process_text_struct, return_dtype=pl.Struct({"text": pl.String, "lang": pl.String})).alias("processed")
    )
    
    # Unpack struct back into DataFrame
    result_df = df.with_columns(
        result_struct.select(pl.col("processed").struct.field("text")).to_series().alias("Interaction_Payload"),
        result_struct.select(pl.col("processed").struct.field("lang")).to_series().alias("Language_Code")
    )

    logger.info("Translation process complete.")
    return result_df


@pa.check_io(df=VoCSkeletonSchema.to_schema(), out=VoCSkeletonSchema.to_schema(), lazy=True)
def validate_reviews(
    df: pl.DataFrame, validation_params: Dict[str, Any]
) -> pl.DataFrame:
    """Validate review data quality using skeleton column names."""
    
    # Validate params with Pydantic
    params = ValidationParams(**validation_params)
    
    logger.info("Validating reviews...")

    initial_count = len(df)

    # Determine text column name (skeleton or legacy)
    text_col = (
        "Interaction_Payload" if "Interaction_Payload" in df.columns else "review_text"
    )
    rating_col = "Rating" if "Rating" in df.columns else "rating"

    # Check required columns
    required_cols = params.required_columns
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Remove rows with missing review text
    df = df.filter(pl.col(text_col).is_not_null())

    # Filter by review length
    min_len = params.min_review_length
    max_len = params.max_review_length

    df = df.filter(
        (pl.col(text_col).str.len_chars() >= min_len)
        & (pl.col(text_col).str.len_chars() <= max_len)
    )

    # Validate rating range
    rating_min, rating_max = params.rating_range
    df = df.filter(
        (pl.col(rating_col) >= rating_min) & (pl.col(rating_col) <= rating_max)
    )

    # Check null percentage
    # null_count returns a DF, we need to process it
    null_counts = df.null_count()
    total_rows = len(df)

    max_null = params.max_null_percentage

    for col in df.columns:
        null_pct = null_counts[col][0] / total_rows
        if null_pct > max_null:
            logger.warning(f"High null percentage detected in {col}: {null_pct}")

    final_count = len(df)
    logger.info(
        f"Validation complete: {initial_count} → {final_count} reviews ({initial_count - final_count} removed)"
    )

    return df


@pa.check_io(df=VoCSkeletonSchema.to_schema(), out=VoCSkeletonSchema.to_schema(), lazy=True)
def clean_reviews(df: pl.DataFrame, cleaning_params: Dict[str, Any]) -> pl.DataFrame:
    """Clean and preprocess review text using skeleton column names."""
    
    # Validate params with Pydantic
    params = CleaningParams(**cleaning_params)
    
    logger.info("Cleaning reviews...")

    # Determine text column name (skeleton or legacy)
    text_col = (
        "Interaction_Payload" if "Interaction_Payload" in df.columns else "review_text"
    )

    # We will build a chain of expressions for the text column
    text_expr = pl.col(text_col)

    # Remove HTML tags
    if params.remove_html:
        text_expr = text_expr.str.replace_all(r"<[^>]+>", "")

    # Remove URLs
    if params.remove_urls:
        text_expr = text_expr.str.replace_all(r"http\S+|www.\S+", "")

    # Convert to lowercase
    if params.lowercase:
        text_expr = text_expr.str.to_lowercase()

    # Remove extra spaces
    if params.remove_extra_spaces:
        text_expr = text_expr.str.replace_all(r"\s+", " ").str.strip_chars()

    df_clean = df.with_columns(text_expr.alias(text_col))

    # Remove empty reviews after cleaning
    df_clean = df_clean.filter(pl.col(text_col).str.len_chars() > 0)

    logger.info(f"Cleaning complete: {len(df_clean)} reviews")
    return df_clean


@pa.check_output(ReviewFeaturesSchema.to_schema(), lazy=True)
def engineer_features(df: pl.DataFrame, feature_params: Dict[str, Any]) -> pl.DataFrame:
    """Engineer features from review text using skeleton column names."""
    
    # Validate params with Pydantic
    params = FeatureParams(**feature_params)
    
    logger.info("Engineering features...")

    # Determine column names (skeleton or legacy)
    text_col = (
        "Interaction_Payload" if "Interaction_Payload" in df.columns else "review_text"
    )
    date_col = "Timestamp" if "Timestamp" in df.columns else "date"

    features = []

    if params.extract_length:
        features.append(pl.col(text_col).str.len_chars().cast(pl.Int64).alias("review_length"))

    if params.extract_word_count:
        features.append(pl.col(text_col).str.split(" ").list.len().cast(pl.Int64).alias("word_count"))

    if params.extract_exclamation_count:
        features.append(
            (
                pl.col(text_col).str.len_chars().cast(pl.Int64)
                - pl.col(text_col).str.replace_all("!", "").str.len_chars().cast(pl.Int64)
            ).alias("exclamation_count")
        )

    if params.extract_question_count:
        features.append(
            (
                pl.col(text_col).str.len_chars().cast(pl.Int64)
                - pl.col(text_col).str.replace_all(r"\?", "").str.len_chars().cast(pl.Int64)
            ).alias("question_count")
        )

    if params.extract_caps_ratio:
        features.append(
            pl.col(text_col)
            .map_elements(
                lambda x: sum(1 for c in x if c.isupper()) / len(x)
                if len(x) > 0
                else 0.0,
                return_dtype=pl.Float64,
            )
            .alias("caps_ratio")
        )

    # Time-based features
    time_features = []
    if date_col in df.columns:
        # Ensure it is datetime - handle potential string or date types
        if df.schema[date_col] == pl.String:
            start_df = df.with_columns(pl.col(date_col).str.to_datetime(strict=False))
        else:
            start_df = df.with_columns(pl.col(date_col).cast(pl.Datetime))
        time_features.append(pl.col(date_col).dt.year().alias("year"))
        time_features.append(pl.col(date_col).dt.month().alias("month"))
        time_features.append(pl.col(date_col).dt.weekday().alias("day_of_week"))

        df = start_df.with_columns(features + time_features)
    else:
        df = df.with_columns(features)

    logger.info(f"Feature engineering complete: {len(df.columns)} columns")
    return df


def write_delta_wsl(df: pl.DataFrame, path: str, mode: str = "overwrite", overwrite_schema: bool = True):
    """WSL-friendly Delta write using /tmp and shutil.copytree"""
    import shutil
    import tempfile
    import deltalake
    from pathlib import Path
    
    target_path = Path(path).absolute()
    schema_mode = "overwrite" if overwrite_schema else None
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "delta_table"
        
        deltalake.write_deltalake(
            str(tmp_path),
            df.to_arrow(),
            mode=mode,
            schema_mode=schema_mode
        )
        
        if target_path.exists():
            shutil.rmtree(target_path)
        
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(tmp_path, target_path)


def register_review_features_to_feast(
    review_features: pl.DataFrame, feast_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Register review features to Feast Feature Store.

    Writes engineered review features to Delta table and applies
    Feast feature definitions.

    Args:
        review_features: DataFrame with engineered review features.
        feast_config: Configuration with feature_repo_path and delta_path.

    Returns:
        Dict with registration status, timestamp, and feature count.
    """
    from feast import FeatureStore
    from pathlib import Path

    feature_repo_path = feast_config.get("feature_repo_path", "feature_repo")
    delta_path = feast_config.get("delta_path", "data/03_primary/review_features")

    # 1. Add event_timestamp for Feast
    if "event_timestamp" not in review_features.columns:
        review_features = review_features.with_columns(
            pl.lit(datetime.now()).alias("event_timestamp")
        )

    # Ensure interaction_id exists (entity key)
    id_col = (
        "Interaction_ID"
        if "Interaction_ID" in review_features.columns
        else "interaction_id"
    )
    if id_col not in review_features.columns:
        # Create one from row index
        review_features = review_features.with_row_index("interaction_id")
    elif id_col != "interaction_id":
        review_features = review_features.rename({id_col: "interaction_id"})

    # 2. Write to Delta table
    write_delta_wsl(review_features, delta_path, mode="overwrite", overwrite_schema=True)
    logger.info(f"Wrote {review_features.height} rows to Delta table: {delta_path}")

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
        "feature_count": review_features.height,
        "delta_path": str(delta_path),
        "feature_repo_path": feature_repo_path,
        "feast_apply": feast_status,
    }
