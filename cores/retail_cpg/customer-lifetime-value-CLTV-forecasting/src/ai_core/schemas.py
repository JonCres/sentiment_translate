import pandera.polars as pa
import polars as pl
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any

# --- Pydantic Models for Configuration ---

class SkeletonMappingParams(BaseModel):
    """Configuration for skeleton mapping."""
    skeleton_mapping: Dict[str, Dict[str, str]] = Field(
        ..., description="Mapping for skeleton columns (transactional, contractual, muscle)"
    )

class PIIParams(BaseModel):
    """Configuration for PII masking."""
    pii_columns: List[str] = Field(default_factory=lambda: ["email", "phone", "name"], description="Structured PII columns to hash")
    text_columns: List[str] = Field(default_factory=list, description="Free-text columns to scan for PII")

class CleaningParams(BaseModel):
    """Configuration for data cleaning."""
    date_format: Optional[str] = Field(None, description="Format string for date parsing")

class RFMParams(BaseModel):
    """Configuration for RFM transformation."""
    observation_period_end: Optional[str] = Field(None, description="End date for observation period YYYY-MM-DD")

class FeastConfigParams(BaseModel):
    """Configuration for Feast registration."""
    feature_repo_path: str = "feature_repo"
    delta_path: str = "data/03_primary/processed_data"

class ModelParams(BaseModel):
    """Configuration for model training."""
    penalizer_coef: float = Field(0.0, description="Penalizer coefficient for the model")
    params: Dict[str, Any] = Field(default_factory=dict, description="Additional model parameters")

class PredictionParams(BaseModel):
    """Configuration for CLTV prediction."""
    prediction_horizons: List[int] = Field([12, 24, 36], description="Prediction horizons in months")
    n_bootstrap: int = Field(100, description="Number of bootstrap samples for CI")


# --- Pandera Schemas for Data Validation ---

class TransactionSkeletonSchema(pa.DataFrameModel):
    """Schema for Cleaned Transaction Skeleton Data."""
    customer_id: str
    transaction_date: pl.Datetime
    transaction_count: float = pa.Field(nullable=True)
    transaction_value: float = pa.Field(nullable=True)
    TotalValue: float = pa.Field(description="Calculated total value (quantity * price)")

    class Config:
        strict = False
        coerce = True

class RFMSchema(pa.DataFrameModel):
    """Schema for RFM (Recency, Frequency, Monetary) Data."""
    customer_id: str
    frequency: float = pa.Field(ge=0.0)
    recency: float = pa.Field(ge=0.0)
    T: float = pa.Field(ge=0.0)
    monetary_value: float = pa.Field(description="Monetary value (avg or total depending on model)")
    cohort_month: Optional[str]
    acquisition_channel: Optional[str]

    class Config:
        strict = False

class CLTVPredictionSchema(pa.DataFrameModel):
    """Schema for CLTV Predictions."""
    customer_id: str
    clv_12mo: float
    clv_95_ci_lower: float
    clv_95_ci_upper: float
    p_alive: float = pa.Field(ge=0.0, le=1.0)
    clv_segment: str
    engagement_score: Optional[float]
    
    class Config:
        strict = False