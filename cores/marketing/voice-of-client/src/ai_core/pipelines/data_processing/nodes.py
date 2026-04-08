import pandas as pd
import polars as pl
import logging
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from utils.device import get_device

# Set seed for reproducible results in langdetect
DetectorFactory.seed = 42

logger = logging.getLogger(__name__)


def map_to_skeleton(df: pd.DataFrame, mapping_params: Dict[str, Any]) -> pd.DataFrame:
    """Map source data columns to the mandatory VoC skeleton schema.

    Args:
        df: Source DataFrame.
        mapping_params: Mapping configuration (mandatory, optional, defaults).

    Returns:
        Standardized DataFrame.
    """
    logger.info("Mapping source columns to Marketing VoC skeleton schema...")

    mandatory_mapping = mapping_params.get("mandatory", {})
    optional_mapping = mapping_params.get("optional", {})
    defaults = mapping_params.get("defaults", {})

    # Start with an empty DataFrame to ensure we only keep what's mapped
    result_df = pd.DataFrame(index=df.index)

    # Process mandatory columns
    for skeleton_col, source_col in mandatory_mapping.items():
        if source_col and source_col in df.columns:
            result_df[skeleton_col] = df[source_col]
        elif skeleton_col in defaults:
            result_df[skeleton_col] = defaults.get(skeleton_col)
        else:
            if skeleton_col not in df.columns:
                logger.warning(f"Mandatory column {skeleton_col} missing.")

    # Process optional columns
    for skeleton_col, source_col in optional_mapping.items():
        if source_col and source_col in df.columns:
            result_df[skeleton_col] = df[source_col]
        elif skeleton_col in defaults:
            result_df[skeleton_col] = defaults.get(skeleton_col)

    logger.info(f"Skeleton mapping complete. Columns: {list(result_df.columns)}")
    return result_df


def validate_data(
    data: pd.DataFrame, validation_config: Dict[str, Any]
) -> pd.DataFrame:
    """Validate multi-channel feedback data using skeleton names.

    Args:
        data: Skeleton-mapped feedback data.
        validation_config: Validation rules.

    Returns:
        Validated DataFrame.
    """
    mandatory_cols = ["Interaction_ID", "Interaction_Payload"]
    missing_cols = [col for col in mandatory_cols if col not in data.columns]

    if missing_cols:
        logger.error(f"Missing mandatory skeleton columns: {missing_cols}")
        raise ValueError(f"Missing mandatory skeleton columns: {missing_cols}")

    if data.empty:
        logger.error("Data is empty")
        raise ValueError("Data is empty")

    return data


def detect_languages(data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Detect the language of the feedback text using Transformer models or langdetect.

    Args:
        data: Standardized DataFrame.
        config: Configuration dictionary.

    Returns:
        DataFrame with an added 'detected_language' column and 'language_score'.
    """
    lang_config = config.get("language_detection", {})
    if not lang_config.get("enabled", True):
        return data

    logger.info("Detecting languages for client feedback...")

    # Extract text column
    texts = data["Interaction_Payload"].astype(str).tolist()

    # 1. Try Transformer-based detection if model is specified
    model_name = lang_config.get("model_name")
    if model_name:
        try:
            import torch
            from transformers import pipeline

            device_str = get_device(purpose="language detection")

            # Map device string to Transformers pipeline compatible device argument
            if device_str == "cuda":
                device = 0
            elif device_str in ["xpu", "mps"]:
                device = torch.device(device_str)
            else:
                device = -1

            logger.info(f"Loading transformer model '{model_name}' on {device_str}...")

            classifier = pipeline(
                "text-classification",
                model=model_name,
                device=device,
                batch_size=lang_config.get("batch_size", 32),
            )

            # Run inference
            results = classifier(texts, truncation=True)

            # Parse results (some models use 'label', others might vary but papluca uses 'label')
            data["detected_language"] = [res["label"] for res in results]
            data["language_score"] = [res["score"] for res in results]

            unique_langs = data["detected_language"].unique()
            logger.info(
                f"Transformer Detection Complete. Languages found: {unique_langs}"
            )
            return data

        except Exception as e:
            logger.warning(
                f"Transformer language detection failed: {e}. Falling back to langdetect."
            )

    # 2. Fallback to langdetect
    def safe_detect(text):
        if not text or len(str(text).strip()) < 3:
            return lang_config.get("fallback_lang", "en"), 0.0
        try:
            return detect(str(text)), 1.0
        except Exception:
            return lang_config.get("fallback_lang", "en"), 0.0

    detections = [safe_detect(t) for t in texts]
    data["detected_language"] = [d[0] for d in detections]
    data["language_score"] = [d[1] for d in detections]

    unique_langs = data["detected_language"].unique()
    logger.info(f"Langdetect Complete. Languages detected: {unique_langs}")

    return data


def translate_feedback(data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Translate feedback to English if needed.

    Args:
        data: DataFrame with 'detected_language' column.
        config: Configuration dictionary.

    Returns:
        DataFrame with 'Interaction_Payload' translated to the target language.
    """
    trans_config = config.get("language_detection", {}).get("translation", {})
    if not trans_config.get("enabled", True):
        return data

    target_lang = trans_config.get("target_lang", "en")
    logger.info(f"Translating non-{target_lang} feedback to {target_lang}...")

    translator = GoogleTranslator(source="auto", target=target_lang)

    # Store the original Interaction_Payload before potential translation
    data["Original_Interaction_Payload"] = data["Interaction_Payload"]

    def perform_translation(row):
        text = row["Interaction_Payload"]
        lang = row.get("detected_language", "unknown")

        if lang == target_lang or not text:
            return text

        try:
            return translator.translate(str(text))
        except Exception as e:
            logger.warning(f"Translation failed for a record: {e}")
            return text

    # Only translate if detected language is different from target
    mask = data["detected_language"] != target_lang
    if mask.any():
        logger.info(f"Found {mask.sum()} records for translation.")
        data.loc[mask, "Interaction_Payload"] = data.loc[mask].apply(
            perform_translation, axis=1
        )
    else:
        logger.info("No records required translation.")

    return data


def clean_feedback_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean multi-channel feedback data using skeleton names.

    Args:
        data: Validated feedback data.

    Returns:
        Cleaned DataFrame.
    """
    # Standardize column types
    data["Interaction_Payload"] = data["Interaction_Payload"].fillna("").astype(str)

    if "Timestamp" not in data.columns or data["Timestamp"].isna().all():
        logger.warning(
            "Timestamp missing or empty. Temporal analysis will be disabled."
        )
        data["Timestamp"] = pd.NaT
    else:
        data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")

    # Clean scores
    if "NPS_Score" in data.columns:
        data["NPS_Score"] = pd.to_numeric(data["NPS_Score"], errors="coerce")
    if "CSAT_Score" in data.columns:
        data["CSAT_Score"] = pd.to_numeric(data["CSAT_Score"], errors="coerce")

    # Clean Revenue fields
    if "Contract_Value" in data.columns:
        data["Contract_Value"] = pd.to_numeric(
            data["Contract_Value"], errors="coerce"
        ).fillna(0)

    return data.drop_duplicates(subset=["Interaction_ID"])


def engineer_voc_features(
    data: pd.DataFrame, feature_config: Dict[str, Any]
) -> pd.DataFrame:
    """Engineer Marketing-specific features using skeleton names.

    Args:
        data: Cleaned feedback data.
        feature_config: Feature engineering parameters.

    Returns:
        Features DataFrame.
    """
    # Ensure NPS_Score and Contract_Value exist, even if missing from raw data
    if "NPS_Score" not in data.columns:
        data["NPS_Score"] = 5  # Neutral NPS score as default
        logger.warning("NPS_Score column not found, defaulting to 5.")
    if "Contract_Value" not in data.columns:
        data["Contract_Value"] = 0  # Default contract value
        logger.warning("Contract_Value column not found, defaulting to 0.")

    # 1. Normalize NPS (0-10)
    if "NPS_Score" in data.columns and not data["NPS_Score"].isna().all():
        nps_conf = feature_config.get("metrics_normalization", {}).get(
            "nps", {"min": 0, "max": 10}
        )
        data["nps_norm"] = (data["NPS_Score"] - nps_conf["min"]) / (
            nps_conf["max"] - nps_conf["min"]
        )
        data["nps_category"] = data["NPS_Score"].apply(
            lambda x: (
                "Promoter"
                if x >= 9
                else ("Detractor" if x <= 6 else "Passive")
                if pd.notna(x)
                else "Unknown"
            )
        )

    # 2. Revenue-Weighted Score Proxy
    if "Contract_Value" in data.columns and "nps_norm" in data.columns:
        # Higher ARR makes low NPS more critical
        total_arr = data["Contract_Value"].sum()
        data["revenue_weight"] = data["Contract_Value"] / (
            total_arr if total_arr > 0 else 1
        )
        data["weighted_risk_proxy"] = (1 - data["nps_norm"]) * data["revenue_weight"]

    # 3. PII Redaction using Microsoft Presidio
    feedback_col = "Interaction_Payload"
    if feature_config.get("pii_masking", {}).get("enabled", True):
        logger.info("PII Redaction via Microsoft Presidio enabled.")
        try:
            # Initialize with English only to avoid warnings about other language recognizers
            registry = RecognizerRegistry()
            registry.load_predefined_recognizers(languages=["en"])
            analyzer = AnalyzerEngine(default_score_threshold=0.4, registry=registry)
            anonymizer = AnonymizerEngine()

            def redact_text(text):
                if not text or len(str(text)) < 2:
                    return text
                results = analyzer.analyze(text=text, language="en")
                anonymized_result = anonymizer.anonymize(
                    text=text, analyzer_results=results
                )
                return anonymized_result.text

            data["feedback_text_masked"] = data[feedback_col].apply(redact_text)
        except Exception as e:
            logger.error(
                f"Presidio Redaction failed: {e}. Falling back to basic regex."
            )
            data["feedback_text_masked"] = data[feedback_col].str.replace(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                "[EMAIL]",
                regex=True,
            )
    else:
        data["feedback_text_masked"] = data[feedback_col]

    return data


def register_features_to_feast(
    features_data: pd.DataFrame, feast_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Register Voice of Client features to Feast Feature Store."""
    from feast import FeatureStore

    feature_repo_path = feast_config.get("feature_repo_path", "feature_repo")
    delta_path = feast_config.get("delta_path", "data/04_feature/client_features.delta")

    # Convert to Polars
    df_polars = pl.from_pandas(features_data)

    if "event_timestamp" not in df_polars.columns:
        df_polars = df_polars.with_columns(pl.col("Timestamp").alias("event_timestamp"))

    # Ensure interaction_id exists for Feast (standard name)
    if "interaction_id" not in df_polars.columns:
        df_polars = df_polars.with_columns(
            pl.col("Interaction_ID").alias("interaction_id")
        )

    # Write to Delta table
    delta_full_path = Path(delta_path)
    delta_full_path.parent.mkdir(parents=True, exist_ok=True)

    df_polars.write_delta(
        str(delta_full_path),
        mode="overwrite",
        delta_write_options={"schema_mode": "overwrite"},
    )
    logger.info(f"Wrote features to Delta table: {delta_path}")

    try:
        store = FeatureStore(repo_path=feature_repo_path)
        store.apply([])
        logger.info("Applied Feast definitions.")
    except Exception as e:
        logger.warning(f"Feast skipped: {e}")

    return {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "feature_count": df_polars.height,
        "delta_path": str(delta_path),
        "feature_repo_path": feature_repo_path,
    }
