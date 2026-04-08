
from typing import List, Optional
import logging
import pandas as pd
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False

logger = logging.getLogger(__name__)

class PiiMasker:
    def __init__(self):
        if PRESIDIO_AVAILABLE:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
        else:
            logger.warning("Microsoft Presidio not found. PII masking will be skipped.")

    def mask_text(self, text: str) -> str:
        if not PRESIDIO_AVAILABLE or not isinstance(text, str):
            return text
        
        try:
            results = self.analyzer.analyze(text=text, entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "IP_ADDRESS"], language='en')
            # Define masking operator
            operators = {
                "DEFAULT": OperatorConfig("replace", {"new_value": "<PII>"}),
                "PERSON": OperatorConfig("replace", {"new_value": "<NAME>"}),
                "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
            }
            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators=operators
            )
            return anonymized_result.text
        except Exception as e:
            logger.error(f"PII Masking failed for text: {text[:20]}... Error: {e}")
            return text

    def mask_dataframe(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        if not PRESIDIO_AVAILABLE:
            return df
        
        df_masked = df.copy()
        for col in columns:
            if col in df_masked.columns:
                logger.info(f"Masking PII in column: {col}")
                # Use apply (can be slow for large data, but safest for Presidio integration)
                df_masked[col] = df_masked[col].apply(self.mask_text)
        return df_masked
