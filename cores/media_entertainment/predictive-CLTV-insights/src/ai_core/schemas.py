import pandera.polars as pa
import polars as pl
from pydantic import BaseModel, Field
from typing import Optional, List

# --- Pydantic Models for Configuration ---

class SkeletonParams(BaseModel):
    """Configuration parameters for skeleton creation."""
    observation_period_end: Optional[str] = Field(None, description="End date for observation period YYYY-MM-DD")
    customer_id_col: str = Field("customer_id", description="Name of the customer ID column")
    date_col: str = Field("transaction_dt", description="Name of the date column")
    amount_col: str = Field("amount_usd", description="Name of the amount column")

class FeastConfig(BaseModel):
    """Configuration for Feast registration."""
    feature_repo_path: str = "feature_repo"
    rfm_delta_path: str = "data/05_model_input/processed_data"
    survival_delta_path: str = "data/05_model_input/survival_data"
    behavioral_delta_path: str = "data/05_model_input/feature_store"


# --- Pandera Schemas for Data Validation ---

class TransactionSchema(pa.DataFrameModel):
    """Schema for Transaction Data."""
    customer_id: str
    transaction_id: str
    transaction_dt: pl.Datetime
    amount_usd: float = pa.Field(ge=0.0, description="Transaction amount must be non-negative")
    transaction_type: str

    class Config:
        strict = True
        coerce = True

class RawTransactionSchema(pa.DataFrameModel):
    """Schema for Raw Transaction Data (pre-cleaning)."""
    customer_id: Optional[str]
    transaction_id: str
    transaction_dt: pl.Datetime
    amount_usd: float  # Can be negative (returns)
    transaction_type: str

    class Config:
        strict = False
        coerce = True

class SubscriptionSchema(pa.DataFrameModel):
    """Schema for Subscription Data."""
    customer_id: str
    start_date: pl.Datetime
    end_date: Optional[pl.Datetime]
    status: str = pa.Field(isin=["active", "churned", "paused"])

    class Config:
        strict = False  # Allow extra columns if needed
        coerce = True

class RFMSchema(pa.DataFrameModel):
    """Schema for RFM (Recency, Frequency, Monetary) Data."""
    customer_id: str
    frequency: float = pa.Field(ge=0.0)
    recency: float = pa.Field(ge=0.0)
    T: float = pa.Field(ge=0.0)
    monetary_value: float = pa.Field(ge=0.0)

    class Config:
        strict = False
